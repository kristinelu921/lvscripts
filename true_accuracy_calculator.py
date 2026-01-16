#!/usr/bin/env python3
"""
True Accuracy Calculator
Calculates pre-critic vs post-critic accuracy without selection bias.

Pre-critic: predicted_answer == correct_choice_idx (from pre_critic_answers)
Post-critic: final_answer == correct_choice_idx (from post_critic_results)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def calculate_true_accuracy(file_path):
    """Calculate true pre-critic and post-critic accuracy."""

    with open(file_path, 'r') as f:
        data = json.load(f)

    print("="*80)
    print("TRUE ACCURACY ANALYSIS")
    print("="*80)
    print(f"File: {Path(file_path).name}")
    print(f"Total videos: {len(data)}")
    print()

    # Pre-critic accuracy
    pre_total = 0
    pre_correct = 0

    # Post-critic accuracy
    post_total = 0
    post_correct = 0

    # Detailed tracking
    critic_changes = {
        'kept_correct': 0,      # Pre correct, post correct
        'changed_wrong_to_correct': 0,  # Pre wrong, post correct
        'changed_correct_to_wrong': 0,  # Pre correct, post wrong
        'kept_wrong': 0,        # Pre wrong, post wrong
    }

    # Judge involvement
    judge_involved = 0
    judge_improved = 0
    judge_worsened = 0
    judge_no_change = 0

    # Track by confidence (5 buckets of 20 points each)
    confidence_buckets = defaultdict(lambda: {'total': 0, 'pre_correct': 0, 'post_correct': 0})

    # Process each video
    for video in data:
        video_id = video.get('video_id')

        # Get pre-critic answers
        pre_critic_answers = {q['uid']: q for q in video.get('pre_critic_answers', [])}

        # Get post-critic results
        post_critic_results = {q['uid']: q for q in video.get('post_critic_results', [])}

        # Calculate accuracies
        for uid, pre_q in pre_critic_answers.items():
            pre_total += 1

            # Pre-critic correctness
            pre_ans = str(pre_q.get('predicted_answer', '')).strip()
            correct_idx = str(pre_q.get('correct_choice_idx', '')).strip()
            pre_is_correct = (pre_ans == correct_idx)

            if pre_is_correct:
                pre_correct += 1

            # Post-critic correctness
            post_q = post_critic_results.get(uid)
            if post_q:
                post_total += 1

                final_ans = post_q.get('final_answer')
                # Handle both string and integer
                final_ans_str = str(final_ans).strip() if final_ans is not None else ''
                post_is_correct = (final_ans_str == correct_idx)

                if post_is_correct:
                    post_correct += 1

                # Track changes
                if pre_is_correct and post_is_correct:
                    critic_changes['kept_correct'] += 1
                elif not pre_is_correct and post_is_correct:
                    critic_changes['changed_wrong_to_correct'] += 1
                elif pre_is_correct and not post_is_correct:
                    critic_changes['changed_correct_to_wrong'] += 1
                else:
                    critic_changes['kept_wrong'] += 1

                # Judge tracking
                if post_q.get('judge_decision') is not None:
                    judge_involved += 1
                    if post_is_correct and not pre_is_correct:
                        judge_improved += 1
                    elif not post_is_correct and pre_is_correct:
                        judge_worsened += 1
                    else:
                        judge_no_change += 1

                # Confidence tracking (5 buckets: 0-19, 20-39, 40-59, 60-79, 80-100)
                confidence = post_q.get('critic_confidence', -1)
                if confidence >= 0:
                    bucket = (confidence // 20) * 20
                    confidence_buckets[bucket]['total'] += 1
                    if pre_is_correct:
                        confidence_buckets[bucket]['pre_correct'] += 1
                    if post_is_correct:
                        confidence_buckets[bucket]['post_correct'] += 1

    # Calculate percentages
    pre_accuracy = (pre_correct / pre_total * 100) if pre_total > 0 else 0
    post_accuracy = (post_correct / post_total * 100) if post_total > 0 else 0
    net_change = post_correct - pre_correct
    net_change_pct = (net_change / pre_total * 100) if pre_total > 0 else 0

    # Print results
    print("="*80)
    print("OVERALL ACCURACY")
    print("="*80)
    print(f"Total questions: {pre_total}")
    print()
    print(f"Pre-critic accuracy:  {pre_correct}/{pre_total} ({pre_accuracy:.2f}%)")
    print(f"Post-critic accuracy: {post_correct}/{post_total} ({post_accuracy:.2f}%)")
    print(f"Net change: {net_change:+d} questions ({net_change_pct:+.2f} percentage points)")
    print()

    # Critic impact
    print("="*80)
    print("CRITIC IMPACT")
    print("="*80)
    print(f"Kept correct:              {critic_changes['kept_correct']:4d}")
    print(f"Changed wrong → correct:   {critic_changes['changed_wrong_to_correct']:4d} ✓")
    print(f"Changed correct → wrong:   {critic_changes['changed_correct_to_wrong']:4d} ✗")
    print(f"Kept wrong:                {critic_changes['kept_wrong']:4d}")
    print()
    net_fixes = critic_changes['changed_wrong_to_correct'] - critic_changes['changed_correct_to_wrong']
    print(f"Net improvement: {net_fixes:+d} questions")
    print()

    # Judge impact
    if judge_involved > 0:
        print("="*80)
        print("JUDGE IMPACT")
        print("="*80)
        print(f"Times judge involved:    {judge_involved}/{post_total} ({judge_involved/post_total*100:.1f}%)")
        print(f"  Improved (wrong→correct): {judge_improved}")
        print(f"  Worsened (correct→wrong): {judge_worsened}")
        print(f"  No change in correctness: {judge_no_change}")
        print()

    # Pre-critic correct answers by confidence
    print("="*80)
    print("PRE-CRITIC CORRECT BY CONFIDENCE")
    print("="*80)
    print(f"{'Confidence':<15} {'Total':>8} {'Pre-Correct':>12} {'Pre-Acc %':>12}")
    print("-"*80)

    for conf in sorted(confidence_buckets.keys()):
        bucket = confidence_buckets[conf]
        pre_acc = (bucket['pre_correct'] / bucket['total'] * 100) if bucket['total'] > 0 else 0

        print(f"{conf:3d}-{conf+19:3d}        {bucket['total']:8d} {bucket['pre_correct']:12d} {pre_acc:11.2f}%")

    print()

    # Confidence analysis
    print("="*80)
    print("CONFIDENCE vs ACCURACY")
    print("="*80)
    print(f"{'Confidence':<15} {'Count':>6} {'Pre-Acc':>10} {'Post-Acc':>10} {'Change':>10}")
    print("-"*80)

    for conf in sorted(confidence_buckets.keys()):
        bucket = confidence_buckets[conf]
        pre_acc = (bucket['pre_correct'] / bucket['total'] * 100) if bucket['total'] > 0 else 0
        post_acc = (bucket['post_correct'] / bucket['total'] * 100) if bucket['total'] > 0 else 0
        change = post_acc - pre_acc

        print(f"{conf:3d}-{conf+19:3d}        {bucket['total']:6d}   {pre_acc:6.2f}%    {post_acc:6.2f}%   {change:+6.2f}%")

    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if net_change > 0:
        print(f"✓ Critic/Judge pipeline IMPROVED accuracy by {net_change_pct:.2f} percentage points")
    elif net_change < 0:
        print(f"✗ Critic/Judge pipeline WORSENED accuracy by {net_change_pct:.2f} percentage points")
    else:
        print(f"= Critic/Judge pipeline had NO NET EFFECT on accuracy")

    print()
    print(f"The critic:")
    print(f"  - Fixed {critic_changes['changed_wrong_to_correct']} errors")
    print(f"  - Introduced {critic_changes['changed_correct_to_wrong']} errors")
    print(f"  - Net: {net_fixes:+d} questions")

    return {
        'pre_total': pre_total,
        'pre_correct': pre_correct,
        'pre_accuracy': pre_accuracy,
        'post_total': post_total,
        'post_correct': post_correct,
        'post_accuracy': post_accuracy,
        'net_change': net_change,
        'net_change_pct': net_change_pct,
        'critic_changes': critic_changes,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python true_accuracy_calculator.py <path_to_results.json>")
        print("\nExample:")
        print("  python true_accuracy_calculator.py test_results_partial_20260116_075954.json")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    results = calculate_true_accuracy(file_path)

    return results


if __name__ == "__main__":
    main()
