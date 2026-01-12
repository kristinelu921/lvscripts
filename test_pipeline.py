#!/usr/bin/env python3
"""
Comprehensive test script for the video QA pipeline.
Tests on questions from downloaded_videos_questions.json with ground truth evaluation.

test_pipeline /mnt/ssh/data/longvideobench/videos_processed --output-dir run_two --mode full
"""

import json
import os
import math
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import pipeline components
from os_model import answer_question, my_model
from critic_model_os import assess_all
from critic_response import re_evaluate_low_confidence_answers


def convert_answer_format(answer, from_format='letter', to_format='letter'):
    """
    Convert between letter choices (A, B, C, D) and numeric indices (0, 1, 2, 3).

    Args:
        answer: Answer in current format (str for letter, int for index)
        from_format: Current format - 'letter' or 'index'
        to_format: Desired format - 'letter' or 'index'

    Returns:
        Converted answer in the desired format
    """
    if from_format == to_format:
        return answer

    if from_format == 'letter' and to_format == 'index':
        # Convert A->0, B->1, C->2, D->3, etc.
        if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
            return ord(answer.upper()) - ord('A')
        return answer

    elif from_format == 'index' and to_format == 'letter':
        # Convert 0->A, 1->B, 2->C, 3->D, etc.
        if isinstance(answer, int) and answer >= 0:
            return chr(ord('A') + answer)
        return answer

    return answer


class PipelineTester:
    """Test harness for video QA pipeline"""

    def __init__(self, video_folder, questions_path, output_dir='test_results', gt_format='index', query_aware=True, vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", llm_model="deepseek-ai/DeepSeek-V3.1", use_no_vlm=False, num_trials=1, video_batch_size=1):
        """
        Args:
            video_folder: Directory containing video folders
            questions_path: Path to questions JSON file
            output_dir: Directory for test results
            gt_format: Format of ground truth answers - 'letter' (A,B,C,D) or 'index' (0,1,2,3)
            query_aware: Whether to use query-aware captions (True) or regular captions (False)
            vlm_model: Vision language model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
            llm_model: Language model to use (default: deepseek-ai/DeepSeek-V3.1)
            use_no_vlm: If True, use caption search only without VLM queries (default: False)
            num_trials: Number of trials to run for majority voting (default: 1 = no majority voting)
            video_batch_size: Number of videos to process in parallel (default: 1 = sequential)
        """
        self.video_folder = Path(video_folder)
        self.questions_path = Path(questions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gt_format = gt_format  # Ground truth format
        self.query_aware = query_aware  # Caption type to use
        self.vlm_model = vlm_model  # Vision language model
        self.llm_model = llm_model  # Language model
        self.use_no_vlm = use_no_vlm  # Caption search only mode
        self.num_trials = num_trials  # Number of trials for majority voting
        self.video_batch_size = video_batch_size  # Number of videos to process in parallel

        # Load questions
        with open(self.questions_path, 'r') as f:
            self.all_questions = json.load(f)

        self.test_start_time = datetime.now()

    def get_majority_answer(self, trial_results):
        """
        Determine the majority answer from multiple trials

        Args:
            trial_results: List of answer results from different trials

        Returns:
            Dict with majority answer and vote counts
        """
        from collections import Counter

        # Extract answers from trials
        answers = []
        for trial in trial_results:
            if trial and 'answer' in trial:
                answer = str(trial['answer']).strip().upper()
                answers.append(answer)

        if not answers:
            return None

        # Count votes
        vote_counts = Counter(answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts[majority_answer]

        return {
            'majority_answer': majority_answer,
            'vote_counts': dict(vote_counts),
            'total_trials': len(answers),
            'majority_count': majority_count,
            'agreement_rate': majority_count / len(answers) if answers else 0,
            'all_trials': trial_results
        }

    def get_testable_videos(self):
        """Get list of videos that are ready for testing (have frames and captions)"""
        testable = []

        for video_id, questions in self.all_questions.items():
            video_dir = self.video_folder / video_id
            frames_dir = video_dir / 'frames'

            # Check for the appropriate caption file based on query_aware toggle
            if self.query_aware:
                captions_file = video_dir / 'captions' / 'frame_captions_query_aware.json'
                caption_type = 'query_aware'
                caption_requirement = 'query-aware captions (run caption_frames_query_aware.py first)'
            else:
                captions_file = video_dir / 'captions' / 'frame_captions.json'
                caption_type = 'regular'
                caption_requirement = 'regular captions (run caption_frames.py first)'

            embeddings_file = video_dir / 'captions' / 'frame_captions_sorted_embeddings.jsonl'

            # Check if video is ready
            if not video_dir.exists():
                print(f"⚠️  {video_id}: Video directory not found")
                continue

            if not frames_dir.exists() or not list(frames_dir.glob('*.jpg')):
                print(f"⚠️  {video_id}: No frames found")
                continue

            if not captions_file.exists():
                print(f"⚠️  {video_id}: No {caption_requirement}")
                continue

            if not embeddings_file.exists():
                print(f"⚠️  {video_id}: No embeddings found (run embed_frame_captions.py first)")
                continue

            testable.append({
                'video_id': video_id,
                'video_dir': str(video_dir),
                'questions': questions,
                'num_questions': len(questions),
                'num_frames': len(list(frames_dir.glob('*.jpg'))),
                'caption_type': caption_type
            })

        return testable

    async def test_single_video(self, video_info, mode='full'):
        """
        Test a single video with all its questions

        Args:
            video_info: Dict with video metadata
            mode: 'qa_only', 'critic_only', or 'full'
        """
        video_id = video_info['video_id']
        video_dir = video_info['video_dir']
        questions = video_info['questions']

        results = {
            'video_id': video_id,
            'num_questions': len(questions),
            'num_frames': video_info['num_frames'],
            'caption_type': video_info.get('caption_type', 'unknown'),
            'pre_critic_answers': [],  # Answers before critic
            'post_critic_results': [],  # Combined answers + critic assessment
            'accuracy_pre_critic': None,
            'accuracy_post_critic': None,  # If re-evaluation implemented
            'avg_confidence': None,
            'errors': []
        }

        print(f"\n{'='*60}")
        print(f"Testing video: {video_id}")
        print(f"Questions: {len(questions)}, Frames: {video_info['num_frames']}")
        print(f"{'='*60}")

        # Phase 1: Question Answering
        if mode in ['qa_only', 'full']:
            if self.num_trials > 1:
                print(f"\n📝 Phase 1: Running QA Pipeline with Majority Voting ({self.num_trials} trials)...")
            else:
                print(f"\n📝 Phase 1: Running QA Pipeline...")

            for i, q in enumerate(questions):
                print(f"\nQuestion {i+1}/{len(questions)}: {q['uid']}")
                print(f"Q: {q['question'][:80]}...")

                try:
                    # Run multiple trials if num_trials > 1
                    trial_results = []

                    for trial_num in range(self.num_trials):
                        if self.num_trials > 1:
                            print(f"  → Trial {trial_num + 1}/{self.num_trials}...")

                        answer = await answer_question(
                            question_uid=f"{q['uid']}_trial{trial_num}" if self.num_trials > 1 else q['uid'],
                            question=q['question'],
                            vid_folder=self.video_folder,
                            vid_num=video_id,
                            candidates=q.get('candidates'),
                            vlm_model=self.vlm_model,
                            llm_model=self.llm_model,
                            use_no_vlm=self.use_no_vlm
                        )

                        trial_results.append(answer)

                        if self.num_trials > 1 and answer:
                            print(f"     Answer: {answer.get('answer', 'N/A')}")

                    # Determine final answer based on majority vote or single trial
                    if self.num_trials > 1:
                        majority_result = self.get_majority_answer(trial_results)
                        if majority_result:
                            # Use majority answer as the final answer
                            final_answer = majority_result['majority_answer']

                            # Find the first trial with the majority answer to get full details
                            representative_trial = None
                            for trial in trial_results:
                                if trial and str(trial.get('answer', '')).strip().upper() == final_answer:
                                    representative_trial = trial
                                    break

                            # Build answer dict with majority voting info
                            answer = representative_trial if representative_trial else trial_results[0]
                            if answer:
                                answer['majority_voting'] = {
                                    'vote_counts': majority_result['vote_counts'],
                                    'agreement_rate': majority_result['agreement_rate'],
                                    'total_trials': majority_result['total_trials'],
                                    'majority_count': majority_result['majority_count']
                                }

                            print(f"  📊 Majority Vote: {final_answer} ({majority_result['majority_count']}/{self.num_trials} votes)")
                            print(f"     Vote Distribution: {majority_result['vote_counts']}")
                            print(f"     Agreement Rate: {majority_result['agreement_rate']*100:.1f}%")
                        else:
                            answer = None
                    else:
                        # Single trial, use as-is
                        answer = trial_results[0] if trial_results else None

                    if answer:
                        # Check correctness against ground truth
                        is_correct = None
                        if 'correct_choice' in q and q['correct_choice'] is not None:
                            correct_answer = q['candidates'][q['correct_choice']]

                            # Get predicted answer from model (as letter)
                            predicted_answer = str(answer.get('answer', '')).strip().upper()

                            # Check if it's a numeric answer (0-4)
                            if predicted_answer in ['0', '1', '2', '3', '4']:
                                predicted_idx = int(predicted_answer)
                            # Check if it's a letter answer (A-E)
                            elif predicted_answer in ['A', 'B', 'C', 'D', 'E']:
                                predicted_idx = convert_answer_format(predicted_answer, from_format='letter', to_format='index')
                            else:
                                # Unknown format, try to convert anyway
                                predicted_idx = convert_answer_format(predicted_answer, from_format='letter', to_format='index')

                            is_correct = (predicted_idx == q['correct_choice'])

                        # Store pre-critic answer
                        pre_critic_result = {
                            'uid': q['uid'],
                            'question': q['question'],
                            'candidates': q['candidates'],
                            'predicted_answer': str(answer.get('answer', '')).strip().upper(),
                            'correct_choice_idx': q.get('correct_choice'),
                            'correct_answer': q['candidates'][q['correct_choice']] if 'correct_choice' in q else None,
                            'is_correct': is_correct,
                            'evidence_frames': answer.get('evidence_frame_numbers', []),
                            'reasoning': answer.get('reasoning', ''),
                            'timestamp': answer.get('timestamp', None)
                        }
                        # Include criteria if present
                        if 'criteria' in answer:
                            pre_critic_result['criteria'] = answer['criteria']
                        # Include majority voting data if present
                        if 'majority_voting' in answer:
                            pre_critic_result['majority_voting'] = answer['majority_voting']
                        results['pre_critic_answers'].append(pre_critic_result)

                        status = "✅ Correct" if is_correct else "❌ Wrong" if is_correct is not None else "⚪ Unknown"
                        print(f"A: {answer.get('answer')} {status}")
                    else:
                        print(f"❌ Failed to get answer")
                        results['errors'].append(f"QA failed for {q['uid']}")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    results['errors'].append(f"Exception in QA for {q['uid']}: {str(e)}")

        # Phase 2: Critic Assessment
        if mode in ['critic_only', 'full']:
            print(f"\n🔍 Phase 2: Running Critic Assessment...")

            # Write answers to file for critic to read (only if we have answers from QA phase)
            if results['pre_critic_answers']:
                video_dir_path = self.video_folder / video_id
                answers_file = video_dir_path / f"{video_id}_answers_reformatted.json"

                # Convert pre-critic answers to the format critic expects
                answers_for_critic = []
                for ans in results['pre_critic_answers']:
                    critic_answer = {
                        'uid': ans['uid'],
                        'question': ans['question'],
                        'candidates': ans['candidates'],
                        'answer': ans['predicted_answer'],
                        'frames': ans['evidence_frames'],
                        'reasoning': ans['reasoning']
                    }
                    # Include criteria if present
                    if 'criteria' in ans:
                        critic_answer['criteria'] = ans['criteria']
                    answers_for_critic.append(critic_answer)

                # Write to file
                with open(answers_file, 'w') as f:
                    json.dump(answers_for_critic, f, indent=2)
                print(f"Wrote {len(answers_for_critic)} answers to {answers_file}")
            elif mode == 'critic_only':
                print(f"⚠️  critic_only mode expects existing answers file from previous QA run")

            try:
                critic_results = await assess_all(
                    video_dir=str(self.video_folder),
                    num=video_id,
                    llm_model=self.llm_model,
                    vlm_model=self.vlm_model
                )

                if critic_results:
                    # Merge pre-critic answers with critic assessments
                    for pre_critic in results['pre_critic_answers']:
                        # Find matching critic assessment
                        matching_critic = next(
                            (c for c in critic_results if c.get('uid') == pre_critic['uid']),
                            None
                        )

                        # Check if judge made a decision
                        judge_decision = None
                        judge_reasoning = None
                        was_reevaluated = False
                        if matching_critic and 'judge_decision' in matching_critic:
                            judge_decision = matching_critic['judge_decision']
                            judge_reasoning = matching_critic.get('judge_reasoning', '')
                            was_reevaluated = True

                        # Determine final answer
                        if was_reevaluated:
                            final_answer = judge_decision
                        else:
                            final_answer = pre_critic['predicted_answer']

                        # Create combined result
                        combined = {
                            # Pre-critic info
                            'uid': pre_critic['uid'],
                            'question': pre_critic['question'],
                            'candidates': pre_critic['candidates'],
                            'predicted_answer': pre_critic['predicted_answer'],
                            'correct_choice_idx': pre_critic['correct_choice_idx'],
                            'correct_answer': pre_critic['correct_answer'],
                            'is_correct': pre_critic['is_correct'],
                            'evidence_frames': pre_critic['evidence_frames'],
                            'reasoning': pre_critic['reasoning'],

                            # Critic assessment
                            'critic_confidence': matching_critic.get('confidence', -1) if matching_critic else -1,
                            'critic_possible_errors': matching_critic.get('possible_errors', []) if matching_critic else [],
                            'critic_suggestion': matching_critic.get('suggestion', None) if matching_critic else None,
                            'critic_answer_choice': matching_critic.get('critic_answer_choice', -1) if matching_critic else -1,

                            # Judge decision (if exists)
                            'judge_decision': judge_decision,
                            'judge_reasoning': judge_reasoning,
                            'was_reevaluated_by_judge': was_reevaluated,

                            # Post-critic answer (uses judge decision if available, otherwise original)
                            'final_answer': final_answer,
                        }

                        results['post_critic_results'].append(combined)

                    # Calculate average confidence
                    confidences = [r['critic_confidence'] for r in results['post_critic_results']
                                 if r['critic_confidence'] >= 0]
                    if confidences:
                        results['avg_confidence'] = sum(confidences) / len(confidences)
                        print(f"Average confidence: {results['avg_confidence']:.1f}%")

                        # Show confidence distribution
                        high = sum(1 for c in confidences if c >= 80)
                        medium = sum(1 for c in confidences if 50 <= c < 80)
                        low = sum(1 for c in confidences if c < 50)
                        print(f"Confidence distribution: High(≥80%):{high}, Medium(50-79%):{medium}, Low(<50%):{low}")
                else:
                    print(f"⚠️  No critic results returned")

            except Exception as e:
                print(f"❌ Critic error: {e}")
                results['errors'].append(f"Critic failed: {str(e)}")

        # Phase 3: Re-evaluation (for low confidence answers < 70%)
        if mode == 'full' and results['post_critic_results']:
            low_conf_count = sum(1 for r in results['post_critic_results'] if r['critic_confidence'] < 70 and r['critic_confidence'] >= 0)

            if low_conf_count > 0:
                print(f"\n🔄 Phase 3: Re-evaluating {low_conf_count} low-confidence answers (< 70%)...")

                try:
                    re_eval_results = await re_evaluate_low_confidence_answers(
                        vid_dir=str(self.video_folder),
                        num=video_id,
                        confidence_threshold=70,
                        llm_model=self.llm_model,
                        vlm_model=self.vlm_model
                    )

                    if re_eval_results:
                        # Update post_critic_results with re-evaluated answers
                        for re_eval in re_eval_results:
                            if re_eval.get('re_evaluated'):
                                # Find matching post_critic result and update it
                                for i, post_critic in enumerate(results['post_critic_results']):
                                    if post_critic['uid'] == re_eval.get('uid'):
                                        # Update with re-evaluated answer
                                        results['post_critic_results'][i]['final_answer'] = re_eval.get('answer')
                                        results['post_critic_results'][i]['re_evaluated'] = True
                                        results['post_critic_results'][i]['original_answer'] = re_eval.get('original_answer')
                                        results['post_critic_results'][i]['re_eval_reasoning'] = re_eval.get('reasoning', '')

                                        # Check correctness of re-evaluated answer
                                        if 'correct_choice_idx' in post_critic and post_critic['correct_choice_idx'] is not None:
                                            re_eval_answer = str(re_eval.get('answer', '')).strip().upper()

                                            # Check if it's a numeric answer (0-4)
                                            if re_eval_answer in ['0', '1', '2', '3', '4']:
                                                predicted_idx = int(re_eval_answer)
                                            # Check if it's a letter answer (A-E)
                                            elif re_eval_answer in ['A', 'B', 'C', 'D', 'E']:
                                                predicted_idx = convert_answer_format(re_eval_answer, from_format='letter', to_format='index')
                                            else:
                                                # Unknown format, try to convert anyway
                                                predicted_idx = convert_answer_format(re_eval_answer, from_format='letter', to_format='index')

                                            results['post_critic_results'][i]['is_correct_after_reeval'] = (predicted_idx == post_critic['correct_choice_idx'])

                                        break

                        print(f"Re-evaluation complete! Updated {low_conf_count} answers.")
                    else:
                        print(f"⚠️  Re-evaluation returned no results")

                except Exception as e:
                    print(f"❌ Re-evaluation error: {e}")
                    results['errors'].append(f"Re-evaluation failed: {str(e)}")
            else:
                print(f"\n✅ All answers have confidence ≥ 50%, no re-evaluation needed")

        # Calculate accuracy (pre-critic)
        if results['pre_critic_answers']:
            correct_count = sum(1 for r in results['pre_critic_answers'] if r['is_correct'])
            total_with_gt = sum(1 for r in results['pre_critic_answers'] if r['is_correct'] is not None)
            if total_with_gt > 0:
                results['accuracy_pre_critic'] = correct_count / total_with_gt
                print(f"\n📊 Pre-Critic Accuracy: {correct_count}/{total_with_gt} ({results['accuracy_pre_critic']*100:.1f}%)")

        # Calculate accuracy (post-critic) - uses final_answer which includes judge decisions
        if results['post_critic_results']:
            # Count questions that were re-evaluated by judge
            judge_reevals = sum(1 for r in results['post_critic_results'] if r['was_reevaluated_by_judge'])

            # Recalculate correctness based on final_answer (not pre-critic answer)
            for r in results['post_critic_results']:
                if r['final_answer'] in ['0', '1', '2', '3', '4']:
                    r['final_answer'] = int(r['final_answer'])
                elif r['final_answer'] in ['A', 'B', 'C', 'D', 'E']:
                    r['final_answer'] = convert_answer_format(r['final_answer'], from_format='letter', to_format='index')
                else:
                    continue
                if r['correct_choice_idx'] is not None:
                    # Check if final_answer matches ground truth
                    r['is_correct_post_critic'] = (r['final_answer'] == r['correct_choice_idx'])
                else:
                    r['is_correct_post_critic'] = None

            correct_count = sum(1 for r in results['post_critic_results'] if r.get('is_correct_post_critic'))
            total_with_gt = sum(1 for r in results['post_critic_results'] if r.get('is_correct_post_critic') is not None)
            if total_with_gt > 0:
                results['accuracy_post_critic'] = correct_count / total_with_gt
                results['num_judge_reevaluations'] = judge_reevals

                # Show if different from pre-critic or if judge made decisions
                if results['accuracy_post_critic'] != results['accuracy_pre_critic'] or judge_reevals > 0:
                    print(f"📊 Post-Critic Accuracy: {correct_count}/{total_with_gt} ({results['accuracy_post_critic']*100:.1f}%)")
                    if judge_reevals > 0:
                        print(f"   → {judge_reevals} answers re-evaluated by judge")

        return results

    async def run_tests(self, video_ids=None, max_videos=None, mode='full'):
        """
        Run tests on multiple videos

        Args:
            video_ids: List of specific video IDs to test (None = all)
            max_videos: Maximum number of videos to test
            mode: 'qa_only', 'critic_only', or 'full'
        """
        testable_videos = self.get_testable_videos()

        # Filter by video_ids if specified
        if video_ids:
            testable_videos = [v for v in testable_videos if v['video_id'] in video_ids]

        # Limit number of videos
        if max_videos:
            testable_videos = testable_videos[:max_videos]

        if not testable_videos:
            print("❌ No testable videos found!")
            return None

        caption_type_label = "Query-Aware Captions" if self.query_aware else "Regular Captions"
        vlm_mode_label = "Caption Search Only (No VLM)" if self.use_no_vlm else "Full Pipeline (with VLM)"

        print(f"\n{'='*60}")
        print(f"PIPELINE TEST SUITE")
        print(f"{'='*60}")
        print(f"Videos to test: {len(testable_videos)}")
        print(f"Total questions: {sum(v['num_questions'] for v in testable_videos)}")
        print(f"Caption type: {caption_type_label}")
        print(f"VLM mode: {vlm_mode_label}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

        all_results = []

        batch_size = getattr(self, 'video_batch_size', 1)
        num_batches = math.ceil(len(testable_videos) / batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(testable_videos))
            batch_videos = testable_videos[start_idx:end_idx]
        
            print(f"PROCESSING BATCH {batch_idx+1}/{num_batches} ({len(batch_videos)} videos)...")
            tasks = []
            for i, video_info in enumerate(batch_videos):
                abs_idx = start_idx + i
                tasks.append(self.test_single_video(video_info, mode=mode))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing video {batch_videos[i]['video_id']}: {result}")
                    all_results.append({
                        'video_id': batch_videos[i]['video_id'],
                        'errors': [f"Error processing video {batch_videos[i]['video_id']}: {result}"]
                    })
                else:
                    print(f"✅ Processed video {batch_videos[i]['video_id']}")
                    all_results.append(result)

                if i % 10 == 0 or i == len(testable_videos) - 1:
                    self.save_results(all_results, partial=True)

        # Generate final report
        report = self.generate_report(all_results)

        # Save final results
        self.save_results(all_results, partial=False)
        self.save_report(report, all_results)

        return {
            'results': all_results,
            'report': report
        }

    def generate_report(self, all_results):
        """Generate comprehensive test report"""
        report = {
            'test_metadata': {
                'start_time': self.test_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.test_start_time).total_seconds(),
                'video_folder': str(self.video_folder),
                'questions_path': str(self.questions_path),
                'caption_type': 'query_aware' if self.query_aware else 'regular',
                'use_no_vlm': self.use_no_vlm,
                'num_trials': self.num_trials,
                'majority_voting_enabled': self.num_trials > 1
            },
            'summary': {
                'total_videos': len(all_results),
                'total_questions': sum(r['num_questions'] for r in all_results),
                'videos_with_errors': sum(1 for r in all_results if r['errors']),
            },
            'qa_performance': {},
            'critic_performance': {},
            'per_video_results': []
        }

        # QA Performance (Pre-Critic)
        all_pre_critic = [qa for r in all_results for qa in r['pre_critic_answers']]
        if all_pre_critic:
            correct = sum(1 for qa in all_pre_critic if qa['is_correct'])
            with_gt = sum(1 for qa in all_pre_critic if qa['is_correct'] is not None)

            report['qa_performance_pre_critic'] = {
                'total_answered': len(all_pre_critic),
                'with_ground_truth': with_gt,
                'correct': correct,
                'accuracy': correct / with_gt if with_gt > 0 else None,
                'accuracy_percentage': f"{(correct / with_gt * 100):.2f}%" if with_gt > 0 else "N/A"
            }

            # Add majority voting statistics if enabled
            if self.num_trials > 1:
                answers_with_voting = [qa for qa in all_pre_critic if 'majority_voting' in qa]
                if answers_with_voting:
                    agreement_rates = [qa['majority_voting']['agreement_rate'] for qa in answers_with_voting]
                    avg_agreement = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0

                    report['qa_performance_pre_critic']['majority_voting'] = {
                        'avg_agreement_rate': avg_agreement,
                        'avg_agreement_percentage': f"{(avg_agreement * 100):.1f}%",
                        'unanimous_answers': sum(1 for qa in answers_with_voting if qa['majority_voting']['agreement_rate'] == 1.0),
                        'split_decisions': sum(1 for qa in answers_with_voting if qa['majority_voting']['agreement_rate'] < 0.6)
                    }

        # Critic Performance (from post_critic_results)
        all_post_critic = [c for r in all_results for c in r['post_critic_results']]
        if all_post_critic:
            confidences = [c['critic_confidence'] for c in all_post_critic if c['critic_confidence'] >= 0]

            # Calculate correlation between confidence and correctness (using post-critic correctness)
            correct_high_conf = sum(1 for c in all_post_critic if c.get('is_correct_post_critic') and c['critic_confidence'] >= 80)
            wrong_high_conf = sum(1 for c in all_post_critic if not c.get('is_correct_post_critic') and c['critic_confidence'] >= 80)
            correct_low_conf = sum(1 for c in all_post_critic if c.get('is_correct_post_critic') and c['critic_confidence'] < 50)
            wrong_low_conf = sum(1 for c in all_post_critic if not c.get('is_correct_post_critic') and c['critic_confidence'] < 50)

            # Judge statistics
            judge_reevals = [c for c in all_post_critic if c.get('was_reevaluated_by_judge')]
            total_judge_reevals = len(judge_reevals)

            # Judge improved accuracy?
            judge_corrections = sum(1 for c in judge_reevals
                                   if c.get('is_correct_post_critic') and not c['is_correct'])
            judge_made_worse = sum(1 for c in judge_reevals
                                  if not c.get('is_correct_post_critic') and c['is_correct'])

            report['critic_performance'] = {
                'total_assessed': len(all_post_critic),
                'avg_confidence': sum(confidences) / len(confidences) if confidences else None,
                'high_confidence_count': sum(1 for c in confidences if c >= 80),
                'medium_confidence_count': sum(1 for c in confidences if 50 <= c < 80),
                'low_confidence_count': sum(1 for c in confidences if c < 50),
                'calibration': {
                    'correct_with_high_confidence': correct_high_conf,
                    'wrong_with_high_confidence': wrong_high_conf,
                    'correct_with_low_confidence': correct_low_conf,
                    'wrong_with_low_confidence': wrong_low_conf
                },
                'judge_reevaluations': {
                    'total_reevaluated': total_judge_reevals,
                    'judge_corrected_errors': judge_corrections,
                    'judge_introduced_errors': judge_made_worse,
                    'net_improvement': judge_corrections - judge_made_worse
                }
            }

            # Post-critic accuracy
            post_critic_correct = sum(1 for c in all_post_critic if c.get('is_correct_post_critic'))
            post_critic_with_gt = sum(1 for c in all_post_critic if c.get('is_correct_post_critic') is not None)
            if post_critic_with_gt > 0:
                report['qa_performance_post_critic'] = {
                    'total_answered': len(all_post_critic),
                    'with_ground_truth': post_critic_with_gt,
                    'correct': post_critic_correct,
                    'accuracy': post_critic_correct / post_critic_with_gt,
                    'accuracy_percentage': f"{(post_critic_correct / post_critic_with_gt * 100):.2f}%"
                }

        # Per-video summary
        for result in all_results:
            video_summary = {
                'video_id': result['video_id'],
                'num_questions': result['num_questions'],
                'caption_type': result.get('caption_type', 'unknown'),
                'accuracy_pre_critic': result.get('accuracy_pre_critic'),
                'accuracy_post_critic': result.get('accuracy_post_critic'),
                'num_judge_reevaluations': result.get('num_judge_reevaluations', 0),
                'avg_confidence': result['avg_confidence'],
                'errors': len(result['errors'])
            }
            report['per_video_results'].append(video_summary)

        return report

    def save_results(self, results, partial=False):
        """Save detailed results to JSON"""
        suffix = '_partial' if partial else ''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results{suffix}_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if not partial:
            print(f"\n💾 Detailed results saved to: {output_path}")

            # Also save simplified results with only requested fields
            simplified_results = []
            for result in results:
                video_result = {
                    'video_id': result['video_id'],
                    'num_questions': result['num_questions'],
                    'num_frames': result.get('num_frames', 'N/A'),
                    'questions': []
                }

                for q in result.get('post_critic_results', []):
                    simplified_q = {
                        'uid': q['uid'],
                        'question': q['question'],
                        'candidates': q.get('candidates', []),
                        'pre_critic_answer': q['predicted_answer'],
                        'reasoning': q.get('reasoning', ''),
                        'correct_choice_idx': q['correct_choice_idx'],
                        'correct_answer': q['correct_answer'],
                        'is_correct': q.get('is_correct_post_critic'),
                        'evidence_frames': q['evidence_frames'],
                        'criteria': q.get('criteria', []),
                        'critic_confidence': q['critic_confidence'],
                        'critic_answer': q.get('critic_answer_choice', -1),
                        'judge_choice': q.get('judge_decision') if q.get('was_reevaluated_by_judge') else None,
                        'judge_reasoning': q.get('judge_reasoning', '') if q.get('was_reevaluated_by_judge') else None,
                        're_evaluated_with_judge': q.get('was_reevaluated_by_judge', False)
                    }
                    video_result['questions'].append(simplified_q)

                simplified_results.append(video_result)

            simplified_path = self.output_dir / f"simplified_results_{timestamp}.json"
            with open(simplified_path, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            print(f"💾 Simplified results saved to: {simplified_path}")

    def save_report(self, report, all_results):
        """Save report to JSON and text"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = self.output_dir / f"test_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Text report
        txt_path = self.output_dir / f"test_report_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VIDEO QA PIPELINE TEST REPORT\n")
            f.write("="*70 + "\n\n")

            # Metadata
            f.write("TEST METADATA\n")
            f.write("-"*70 + "\n")
            for key, value in report['test_metadata'].items():
                f.write(f"{key}: {value}\n")

            # Summary
            f.write("\n\nSUMMARY\n")
            f.write("-"*70 + "\n")
            for key, value in report['summary'].items():
                f.write(f"{key}: {value}\n")

            # QA Performance (Pre-Critic)
            if report.get('qa_performance_pre_critic'):
                f.write("\n\nQA PERFORMANCE (PRE-CRITIC)\n")
                f.write("-"*70 + "\n")
                for key, value in report['qa_performance_pre_critic'].items():
                    if key == 'majority_voting' and isinstance(value, dict):
                        f.write(f"\nMajority Voting Statistics:\n")
                        for mv_key, mv_value in value.items():
                            f.write(f"  {mv_key}: {mv_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            # QA Performance (Post-Critic)
            if report.get('qa_performance_post_critic'):
                f.write("\n\nQA PERFORMANCE (POST-CRITIC)\n")
                f.write("-"*70 + "\n")
                for key, value in report['qa_performance_post_critic'].items():
                    f.write(f"{key}: {value}\n")

            # Critic Performance
            if report['critic_performance']:
                f.write("\n\nCRITIC PERFORMANCE\n")
                f.write("-"*70 + "\n")
                for key, value in report['critic_performance'].items():
                    if key == 'judge_reevaluations' and isinstance(value, dict):
                        f.write(f"\nJudge Re-evaluations:\n")
                        for judge_key, judge_value in value.items():
                            f.write(f"  {judge_key}: {judge_value}\n")
                    elif key == 'calibration' and isinstance(value, dict):
                        f.write(f"\nCalibration:\n")
                        for cal_key, cal_value in value.items():
                            f.write(f"  {cal_key}: {cal_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            # Per-video results
            f.write("\n\nPER-VIDEO RESULTS\n")
            f.write("-"*70 + "\n")
            for video in report['per_video_results']:
                f.write(f"\n{video['video_id']} ({video.get('caption_type', 'unknown')}):\n")
                f.write(f"  Questions: {video['num_questions']}\n")
                if video.get('accuracy_pre_critic') is not None:
                    f.write(f"  Pre-Critic Accuracy: {video['accuracy_pre_critic']*100:.1f}%\n")
                if video.get('accuracy_post_critic') is not None and video['accuracy_post_critic'] != video.get('accuracy_pre_critic'):
                    f.write(f"  Post-Critic Accuracy: {video['accuracy_post_critic']*100:.1f}%\n")
                if video.get('num_judge_reevaluations', 0) > 0:
                    f.write(f"  Judge Re-evaluations: {video['num_judge_reevaluations']}\n")
                if video['avg_confidence'] is not None:
                    f.write(f"  Avg Confidence: {video['avg_confidence']:.1f}%\n")
                if video['errors']:
                    f.write(f"  Errors: {video['errors']}\n")

            # Detailed per-question results
            f.write("\n\nDETAILED PER-QUESTION RESULTS\n")
            f.write("="*70 + "\n")
            for result in all_results:
                f.write(f"\n{result['video_id']}:\n")
                f.write("-"*70 + "\n")
                for q in result['post_critic_results']:
                    # Use post-critic correctness
                    status = "✅" if q.get('is_correct_post_critic') else "❌" if q.get('is_correct_post_critic') is not None else "⚪"
                    f.write(f"\n{q['uid']}: {status}\n")
                    f.write(f"  Q: {q['question'][:100]}...\n")
                    f.write(f"  Predicted: {q['predicted_answer']} | Correct: {q['correct_answer']}\n")

                    # Show judge decision if exists
                    if q.get('was_reevaluated_by_judge'):
                        f.write(f"  ⚖️  Judge Re-evaluated: Original={q['predicted_answer']} → Judge={q['judge_decision']}\n")
                        if q.get('judge_reasoning'):
                            f.write(f"     Judge Reasoning: {q['judge_reasoning'][:100]}...\n")

                    f.write(f"  Critic Confidence: {q['critic_confidence']}%\n")
                    if q.get('critic_possible_errors'):
                        f.write(f"  Possible Errors: {', '.join(q['critic_possible_errors'])}\n")
                    if q.get('critic_suggestion'):
                        f.write(f"  Critic Suggestion: {q['critic_suggestion']}\n")
                    f.write(f"  Evidence Frames: {q['evidence_frames']}\n")

        print(f"📊 Report saved to: {json_path}")
        print(f"📄 Text report saved to: {txt_path}")

        # Print summary to console
        self.print_summary(report)

    def print_summary(self, report):
        """Print summary to console"""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")

        if report.get('qa_performance_pre_critic'):
            qa = report['qa_performance_pre_critic']
            print(f"\n📝 QA Performance:")
            print(f"   Questions answered: {qa['total_answered']}")
            print(f"   Accuracy: {qa['accuracy_percentage']}")

        if report['critic_performance']:
            critic = report['critic_performance']
            print(f"\n🔍 Critic Performance:")
            print(f"   Assessments: {critic['total_assessed']}")
            if critic['avg_confidence']:
                print(f"   Avg confidence: {critic['avg_confidence']:.1f}%")
            print(f"   High confidence (≥80%): {critic['high_confidence_count']}")
            print(f"   Medium confidence (50-79%): {critic['medium_confidence_count']}")
            print(f"   Low confidence (<50%): {critic['low_confidence_count']}")

            if critic.get('calibration'):
                cal = critic['calibration']
                print(f"\n   Calibration:")
                print(f"     Correct + High Conf: {cal['correct_with_high_confidence']}")
                print(f"     Wrong + High Conf: {cal['wrong_with_high_confidence']}")
                print(f"     Correct + Low Conf: {cal['correct_with_low_confidence']}")
                print(f"     Wrong + Low Conf: {cal['wrong_with_low_confidence']}")

        print(f"\n⏱️  Duration: {report['test_metadata']['duration_seconds']:.1f}s")
        print(f"{'='*70}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Test video QA pipeline on downloaded videos with ground truth evaluation"
    )

    parser.add_argument('video_folder', help='Directory containing video folders')
    parser.add_argument('--questions', default='/mnt/ssd/data/longvideobench/downloaded_videos_questions.json',
                       help='Path to questions JSON')
    parser.add_argument('--output-dir', default='test_results',
                       help='Directory for test results')
    parser.add_argument('--videos', nargs='+',
                       help='Specific video IDs to test (default: all)')
    parser.add_argument('--max-videos', type=int,
                       help='Maximum number of videos to test')
    parser.add_argument('--mode', choices=['qa_only', 'critic_only', 'full'], default='full',
                       help='Test mode: qa_only, critic_only, or full pipeline')
    parser.add_argument('--gt-format', choices=['letter', 'index'], default='index',
                       help='Ground truth answer format: "letter" (A,B,C,D) or "index" (0,1,2,3). Default: index')
    parser.add_argument('--query-aware', action='store_true', default=False,
                       help='Use query-aware captions instead of regular captions')
    parser.add_argument('--vlm-model', type=str, default='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                       help='Vision language model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)')
    #'Qwen/Qwen2.5-VL-72B-Instruct'
    parser.add_argument('--llm-model', type=str, default='deepseek-ai/DeepSeek-V3.1',
                       help='Language model to use (default: deepseek-ai/DeepSeek-V3.1)')
    parser.add_argument('--no-vlm', action='store_true', default=False,
                       help='Use caption search only without VLM queries (uses prompts_no_vlm.py)')
    parser.add_argument('--num-trials', type=int, default=1,
                       help='Number of trials for majority voting (default: 1 = no majority voting, e.g., 5 for 5 trials)')
    parser.add_argument('--video-batch-size', type=int, default=1,
                       help='Number of videos to process in parallel (default: 1 = sequential, e.g., 3 for 3 videos at once)')

    args = parser.parse_args()

    # Create tester
    tester = PipelineTester(
        video_folder=args.video_folder,
        questions_path=args.questions,
        output_dir=args.output_dir,
        gt_format=args.gt_format,
        query_aware=args.query_aware,
        vlm_model=args.vlm_model,
        llm_model=args.llm_model,
        use_no_vlm=args.no_vlm,
        num_trials=args.num_trials,
        video_batch_size=args.video_batch_size
    )

    # Run tests
    await tester.run_tests(
        video_ids=args.videos,
        max_videos=args.max_videos,
        mode=args.mode
    )


if __name__ == "__main__":
    asyncio.run(main())
