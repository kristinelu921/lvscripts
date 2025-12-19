#!/usr/bin/env python3
"""
Enhanced Evaluation Script with Comprehensive Tracing

Runs evaluation on a specified number of questions and saves:
1. Individual question traces (full reasoning logs)
2. Results with ground truth comparison
3. Detailed metrics and error analysis
4. Per-video and per-category breakdowns
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from os_model import Pipeline as OSPipeline, query_model_iterative_with_retry as os_query
from critic_model_os import CriticPipeline, critic_assess
from critic_response import (
    Pipeline as CriticRespPipeline,
    query_model_iterative_with_retry as critic_resp_query,
    create_enhanced_prompt,
)


def load_questions_with_groundtruth(videos_dir: str, max_questions: int = None) -> List[Dict]:
    """
    Load questions with ground truth answers from video directories.

    Args:
        videos_dir: Path to videos directory
        max_questions: Maximum number of questions to load

    Returns:
        List of question dictionaries with video_id, question, ground_truth, etc.
    """
    all_questions = []

    for video_id in sorted(os.listdir(videos_dir)):
        vid_path = os.path.join(videos_dir, video_id)
        if not os.path.isdir(vid_path):
            continue

        questions_file = os.path.join(vid_path, f'{video_id}_questions.json')
        answers_file = os.path.join(vid_path, f'{video_id}_question_answers.json')

        if not os.path.exists(questions_file) or not os.path.exists(answers_file):
            continue

        # Load questions
        with open(questions_file, 'r') as f:
            questions = json.load(f)

        # Load ground truth answers
        with open(answers_file, 'r') as f:
            ground_truth = json.load(f)
            gt_dict = {gt['uid']: gt['answer'] for gt in ground_truth}

        # Combine questions with ground truth
        for q in questions:
            uid = q.get('uid')
            if uid in gt_dict:
                all_questions.append({
                    'video_id': video_id,
                    'vid_path': vid_path,
                    'uid': uid,
                    'question': q['question'],
                    'ground_truth': gt_dict[uid]
                })

        if max_questions and len(all_questions) >= max_questions:
            break

    return all_questions[:max_questions] if max_questions else all_questions


def save_question_trace(
    output_dir: str,
    video_id: str,
    uid: str,
    question: str,
    ground_truth: str,
    os_result: Dict,
    critic_result: Dict = None,
    final_result: Dict = None,
    timing: Dict = None
):
    """Save comprehensive trace for a single question"""
    trace_dir = os.path.join(output_dir, 'traces', video_id)
    os.makedirs(trace_dir, exist_ok=True)

    trace = {
        'metadata': {
            'video_id': video_id,
            'question_id': uid,
            'timestamp': datetime.now().isoformat(),
            'timing': timing or {}
        },
        'question': {
            'text': question,
            'ground_truth': ground_truth
        },
        'os_model': os_result,
        'critic_model': critic_result,
        'final_result': final_result,
        'evaluation': {
            'os_correct': os_result.get('answer', '').strip().upper() == ground_truth.upper() if os_result else False,
            'final_correct': final_result.get('answer', '').strip().upper() == ground_truth.upper() if final_result else False,
        }
    }

    trace_file = os.path.join(trace_dir, f'{uid}_trace.json')
    with open(trace_file, 'w') as f:
        json.dump(trace, f, indent=2)

    return trace_file


async def process_single_question(
    q: Dict,
    llm_model: str,
    vlm_model: str,
    use_critic: bool,
    output_dir: str
) -> Dict[str, Any]:
    """
    Process a single question and return results with timing.
    """
    video_id = q['video_id']
    vid_path = q['vid_path']
    uid = q['uid']
    question = q['question']
    ground_truth = q['ground_truth']

    result = {
        'video_id': video_id,
        'question_id': uid,
        'question': question,
        'ground_truth': ground_truth,
        'timing': {}
    }

    try:
        # Phase 1: OS Model
        print(f"\n{'='*80}")
        print(f"[{video_id}/{uid}] Processing question...")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*80}")

        start_time = time.time()
        os_model = OSPipeline(llm_model, vlm_model)
        answers_file = os.path.join(vid_path, f"{video_id}_answers.json")

        os_answer = await os_query(os_model, question, uid, vid_path, answers_file)

        # Load answer from file if needed
        if isinstance(os_answer, str):
            with open(answers_file, "r") as f:
                answers = json.load(f)
            os_answer = next((a for a in answers if a.get("uid") == uid), None)

        result['timing']['os_model'] = time.time() - start_time

        if not os_answer:
            print(f"ERROR: Failed to get OS answer for {uid}")
            result['error'] = 'OS model failed'
            return result

        result['os_prediction'] = os_answer.get('answer', '')
        result['os_reasoning'] = os_answer.get('reasoning', '')
        result['os_frames'] = os_answer.get('evidence_frame_numbers', [])
        result['os_correct'] = result['os_prediction'].strip().upper() == ground_truth.upper()

        print(f"OS Answer: {result['os_prediction']} (Correct: {result['os_correct']})")

        # Phase 2: Critic (if enabled)
        if use_critic:
            start_time = time.time()

            try:
                critic = CriticPipeline(llm_model, vlm_model)
                critic_assessment = await critic_assess(
                    critic,
                    question,
                    uid,
                    result['os_prediction'],
                    result['os_reasoning'],
                    result['os_frames'],
                    os.path.dirname(vid_path),
                    video_id,
                )

                result['timing']['critic'] = time.time() - start_time
                result['critic_confidence'] = critic_assessment.get('confidence', 0)
                result['critic_feedback'] = critic_assessment.get('feedback', '')

                print(f"Critic Confidence: {result['critic_confidence']}%")

                # Phase 3: Re-evaluation if needed
                if critic_assessment.get('confidence', 100) < 70:
                    print(f"Low confidence, re-evaluating...")
                    start_time = time.time()

                    enhanced_question = create_enhanced_prompt(critic_assessment)
                    critic_resp_model = CriticRespPipeline(llm_model, vlm_model)

                    re_eval = await critic_resp_query(
                        critic_resp_model,
                        enhanced_question,
                        uid,
                        vid_path,
                    )

                    result['timing']['re_eval'] = time.time() - start_time
                    result['final_prediction'] = re_eval.get('answer', '')
                    result['final_reasoning'] = re_eval.get('reasoning', '')
                    result['final_correct'] = result['final_prediction'].strip().upper() == ground_truth.upper()

                    print(f"Re-eval Answer: {result['final_prediction']} (Correct: {result['final_correct']})")
                else:
                    result['final_prediction'] = result['os_prediction']
                    result['final_correct'] = result['os_correct']

            except Exception as e:
                print(f"Critic failed: {e}")
                result['critic_error'] = str(e)
                result['final_prediction'] = result['os_prediction']
                result['final_correct'] = result['os_correct']
        else:
            result['final_prediction'] = result['os_prediction']
            result['final_correct'] = result['os_correct']

        # Save trace
        trace_file = save_question_trace(
            output_dir,
            video_id,
            uid,
            question,
            ground_truth,
            os_answer,
            result.get('critic_feedback'),
            {'answer': result.get('final_prediction'), 'reasoning': result.get('final_reasoning', '')},
            result['timing']
        )

        result['trace_file'] = trace_file
        print(f"Trace saved: {trace_file}")
        print(f"Total time: {sum(result['timing'].values()):.2f}s")

    except Exception as e:
        print(f"ERROR processing question {uid}: {e}")
        result['error'] = str(e)
        result['final_correct'] = False

    return result


async def run_evaluation(
    videos_dir: str,
    output_dir: str,
    max_questions: int = 300,
    llm_model: str = "deepseek-ai/DeepSeek-V3.1",
    vlm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    use_critic: bool = True,
    start_from: int = 0
):
    """
    Run evaluation on questions with comprehensive tracing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load questions
    print(f"Loading questions from {videos_dir}...")
    questions = load_questions_with_groundtruth(videos_dir, max_questions + start_from)

    if start_from > 0:
        questions = questions[start_from:]
        print(f"Starting from question {start_from}")

    print(f"Loaded {len(questions)} questions with ground truth")

    if not questions:
        print("ERROR: No questions found")
        return

    # Process questions
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION")
    print(f"Questions: {len(questions)}")
    print(f"LLM: {llm_model}")
    print(f"VLM: {vlm_model}")
    print(f"Critic: {use_critic}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    results = []
    start_time = time.time()

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing {q['video_id']}/{q['uid']}...")

        result = await process_single_question(
            q,
            llm_model,
            vlm_model,
            use_critic,
            output_dir
        )

        results.append(result)

        # Save intermediate results every 10 questions
        if i % 10 == 0:
            intermediate_file = os.path.join(output_dir, f'results_checkpoint_{i}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nCheckpoint saved: {intermediate_file}")

    total_time = time.time() - start_time

    # Calculate metrics
    print(f"\n{'='*80}")
    print("CALCULATING METRICS")
    print(f"{'='*80}\n")

    os_correct = sum(1 for r in results if r.get('os_correct'))
    final_correct = sum(1 for r in results if r.get('final_correct'))
    errors = sum(1 for r in results if 'error' in r)

    metrics = {
        'summary': {
            'total_questions': len(results),
            'total_time_seconds': total_time,
            'avg_time_per_question': total_time / len(results),
            'errors': errors,
        },
        'accuracy': {
            'os_correct': os_correct,
            'os_accuracy': os_correct / len(results) * 100,
            'final_correct': final_correct,
            'final_accuracy': final_correct / len(results) * 100,
            'improvement': (final_correct - os_correct) / len(results) * 100,
        },
        'per_video': {}
    }

    # Per-video breakdown
    videos = set(r['video_id'] for r in results)
    for video_id in videos:
        video_results = [r for r in results if r['video_id'] == video_id]
        video_correct = sum(1 for r in video_results if r.get('final_correct'))

        metrics['per_video'][video_id] = {
            'total': len(video_results),
            'correct': video_correct,
            'accuracy': video_correct / len(video_results) * 100 if video_results else 0
        }

    # Save final results
    final_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'videos_dir': videos_dir,
            'llm_model': llm_model,
            'vlm_model': vlm_model,
            'use_critic': use_critic,
        },
        'metrics': metrics,
        'results': results
    }

    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total questions: {len(results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Errors: {errors}")
    print(f"\nOS Model Accuracy: {metrics['accuracy']['os_accuracy']:.2f}% ({os_correct}/{len(results)})")
    print(f"Final Accuracy: {metrics['accuracy']['final_accuracy']:.2f}% ({final_correct}/{len(results)})")
    print(f"Improvement: {metrics['accuracy']['improvement']:+.2f}%")
    print(f"\nResults: {results_file}")
    print(f"Traces: {output_dir}/traces/")
    print(f"{'='*80}\n")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with comprehensive tracing")
    parser.add_argument(
        '--videos_dir',
        type=str,
        default='/resource/scripts/videos',
        help='Directory containing videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/resource/evaluation_results',
        help='Output directory for results and traces'
    )
    parser.add_argument(
        '--max_questions',
        type=int,
        default=300,
        help='Maximum number of questions to evaluate'
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        default='deepseek-ai/DeepSeek-V3.1',
        help='LLM model name'
    )
    parser.add_argument(
        '--vlm_model',
        type=str,
        default='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        help='VLM model name'
    )
    parser.add_argument(
        '--no_critic',
        action='store_true',
        help='Disable critic module'
    )
    parser.add_argument(
        '--start_from',
        type=int,
        default=0,
        help='Start from question N (for resuming)'
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
        use_critic=not args.no_critic,
        start_from=args.start_from
    ))


if __name__ == '__main__':
    main()
