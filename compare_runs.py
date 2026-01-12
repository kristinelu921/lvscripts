#!/usr/bin/env python3
"""
Compare answers between two test runs
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def convert_answer_to_int(answer):
    """Convert answer to int, handling various formats"""
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.isdigit():
            return int(answer)
        if len(answer) == 1 and answer.upper() in 'ABCD':
            return ord(answer.upper()) - ord('A')
    return None

def load_results(filepath):
    """Load test results and extract answers by uid"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    answers = {}
    for video in data:
        video_id = video.get('video_id')

        # Get pre-critic answers
        if 'pre_critic_answers' in video:
            for qa in video['pre_critic_answers']:
                uid = qa.get('uid')
                predicted = qa.get('predicted_answer')
                correct_idx = qa.get('correct_choice_idx')

                answers[uid] = {
                    'video_id': video_id,
                    'question': qa.get('question', '')[:100] + '...',
                    'predicted': convert_answer_to_int(predicted),
                    'correct': correct_idx,
                    'is_correct': convert_answer_to_int(predicted) == correct_idx,
                    'raw_answer': predicted
                }

    return answers

def compare_runs(run1_file, run2_file):
    """Compare answers between two runs"""

    print("Loading results...")
    run1 = load_results(run1_file)
    run2 = load_results(run2_file)

    print(f"Run 1: {len(run1)} questions")
    print(f"Run 2: {len(run2)} questions")

    # Find common questions
    common_uids = set(run1.keys()) & set(run2.keys())
    run1_only = set(run1.keys()) - set(run2.keys())
    run2_only = set(run2.keys()) - set(run1.keys())

    print(f"\nCommon questions: {len(common_uids)}")
    print(f"Only in Run 1: {len(run1_only)}")
    print(f"Only in Run 2: {len(run2_only)}")

    # Calculate accuracies
    run1_correct = sum(1 for uid in common_uids if run1[uid]['is_correct'])
    run2_correct = sum(1 for uid in common_uids if run2[uid]['is_correct'])

    run1_accuracy = (run1_correct / len(common_uids) * 100) if common_uids else 0
    run2_accuracy = (run2_correct / len(common_uids) * 100) if common_uids else 0

    print("\n" + "="*80)
    print("ACCURACY COMPARISON (on common questions)")
    print("="*80)
    print(f"Run 1 Accuracy: {run1_correct}/{len(common_uids)} = {run1_accuracy:.2f}%")
    print(f"Run 2 Accuracy: {run2_correct}/{len(common_uids)} = {run2_accuracy:.2f}%")
    print(f"Difference: {run2_accuracy - run1_accuracy:+.2f}%")

    # Categorize changes
    both_correct = []
    both_wrong = []
    run1_correct_run2_wrong = []  # Regression
    run1_wrong_run2_correct = []  # Improvement
    answer_changed = []

    for uid in common_uids:
        r1 = run1[uid]
        r2 = run2[uid]

        if r1['is_correct'] and r2['is_correct']:
            both_correct.append(uid)
        elif not r1['is_correct'] and not r2['is_correct']:
            both_wrong.append(uid)
        elif r1['is_correct'] and not r2['is_correct']:
            run1_correct_run2_wrong.append(uid)
        elif not r1['is_correct'] and r2['is_correct']:
            run1_wrong_run2_correct.append(uid)

        if r1['predicted'] != r2['predicted']:
            answer_changed.append(uid)

    print("\n" + "="*80)
    print("ANSWER COMPARISON")
    print("="*80)
    print(f"Both runs correct: {len(both_correct)}")
    print(f"Both runs wrong: {len(both_wrong)}")
    print(f"Run 1 correct → Run 2 wrong (REGRESSION): {len(run1_correct_run2_wrong)}")
    print(f"Run 1 wrong → Run 2 correct (IMPROVEMENT): {len(run1_wrong_run2_correct)}")
    print(f"Answer changed: {len(answer_changed)}")

    # Show regressions
    if run1_correct_run2_wrong:
        print("\n" + "="*80)
        print(f"REGRESSIONS (Run 1 correct → Run 2 wrong): {len(run1_correct_run2_wrong)}")
        print("="*80)
        for uid in run1_correct_run2_wrong[:10]:
            r1 = run1[uid]
            r2 = run2[uid]
            print(f"\n{uid}")
            print(f"  Video: {r1['video_id']}")
            print(f"  Question: {r1['question']}")
            print(f"  Correct answer: {r1['correct']}")
            print(f"  Run 1: {r1['predicted']} ✓")
            print(f"  Run 2: {r2['predicted']} ✗")
        if len(run1_correct_run2_wrong) > 10:
            print(f"\n  ... and {len(run1_correct_run2_wrong) - 10} more")

    # Show improvements
    if run1_wrong_run2_correct:
        print("\n" + "="*80)
        print(f"IMPROVEMENTS (Run 1 wrong → Run 2 correct): {len(run1_wrong_run2_correct)}")
        print("="*80)
        for uid in run1_wrong_run2_correct[:10]:
            r1 = run1[uid]
            r2 = run2[uid]
            print(f"\n{uid}")
            print(f"  Video: {r1['video_id']}")
            print(f"  Question: {r1['question']}")
            print(f"  Correct answer: {r1['correct']}")
            print(f"  Run 1: {r1['predicted']} ✗")
            print(f"  Run 2: {r2['predicted']} ✓")
        if len(run1_wrong_run2_correct) > 10:
            print(f"\n  ... and {len(run1_wrong_run2_correct) - 10} more")

    # Show some examples where both got it wrong but changed answer
    both_wrong_changed = [uid for uid in both_wrong if uid in answer_changed]
    if both_wrong_changed:
        print("\n" + "="*80)
        print(f"BOTH WRONG BUT ANSWER CHANGED: {len(both_wrong_changed)}")
        print("="*80)
        for uid in both_wrong_changed[:5]:
            r1 = run1[uid]
            r2 = run2[uid]
            print(f"\n{uid}")
            print(f"  Video: {r1['video_id']}")
            print(f"  Question: {r1['question']}")
            print(f"  Correct answer: {r1['correct']}")
            print(f"  Run 1: {r1['predicted']} ✗")
            print(f"  Run 2: {r2['predicted']} ✗")
        if len(both_wrong_changed) > 5:
            print(f"\n  ... and {len(both_wrong_changed) - 5} more")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Net change: {len(run1_wrong_run2_correct) - len(run1_correct_run2_wrong):+d} questions")
    print(f"Accuracy change: {run2_accuracy - run1_accuracy:+.2f}%")

    if run2_accuracy > run1_accuracy:
        print(f"\n✅ Run 2 performed BETTER")
    elif run2_accuracy < run1_accuracy:
        print(f"\n❌ Run 2 performed WORSE")
    else:
        print(f"\n➖ No change in accuracy")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <run1_results.json> <run2_results.json>")
        print("\nExample:")
        print("  python compare_runs.py my_results/test_results_20260102_171915.json run_two/test_results_20260103_202314.json")
        sys.exit(1)

    run1_file = sys.argv[1]
    run2_file = sys.argv[2]

    if not Path(run1_file).exists():
        print(f"Error: File {run1_file} not found")
        sys.exit(1)

    if not Path(run2_file).exists():
        print(f"Error: File {run2_file} not found")
        sys.exit(1)

    compare_runs(run1_file, run2_file)