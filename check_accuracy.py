#!/usr/bin/env python3
"""
Check accuracy in results file, handling string/int conversions
"""
import json
import sys

def check_accuracy(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    total_correct_pre = 0
    total_questions_pre = 0
    total_correct_post = 0
    total_questions_post = 0

    # Track mismatches between is_correct field and actual correctness
    mismatches = []

    for video in data:
        video_id = video.get('video_id', 'unknown')

        # Check pre-critic answers
        for answer in video.get('pre_critic_answers', []):
            uid = answer.get('uid')
            predicted = answer.get('predicted_answer')
            correct_idx = answer.get('correct_choice_idx')
            is_correct_field = answer.get('is_correct')

            # Convert predicted to int if it's a string digit
            if isinstance(predicted, str) and predicted.isdigit():
                predicted_int = int(predicted)
            elif isinstance(predicted, int):
                predicted_int = predicted
            else:
                print(f"Warning: {uid} has non-numeric predicted_answer: {predicted}")
                continue

            # Check if actually correct
            actual_correct = (predicted_int == correct_idx)

            if actual_correct:
                total_correct_pre += 1
            total_questions_pre += 1

            # Check for mismatch
            if actual_correct != is_correct_field:
                mismatches.append({
                    'uid': uid,
                    'predicted': predicted,
                    'predicted_int': predicted_int,
                    'correct_idx': correct_idx,
                    'is_correct_field': is_correct_field,
                    'actual_correct': actual_correct
                })

        # Check post-critic answers
        for answer in video.get('post_critic_results', []):
            uid = answer.get('uid')
            predicted = answer.get('answer')
            correct_idx = answer.get('correct_choice_idx')

            # Convert predicted to int if it's a string digit
            if isinstance(predicted, str) and predicted.isdigit():
                predicted_int = int(predicted)
            elif isinstance(predicted, int):
                predicted_int = predicted
            else:
                continue

            # Check if correct
            if predicted_int == correct_idx:
                total_correct_post += 1
            total_questions_post += 1

    # Calculate accuracies
    accuracy_pre = (total_correct_pre / total_questions_pre * 100) if total_questions_pre > 0 else 0
    accuracy_post = (total_correct_post / total_questions_post * 100) if total_questions_post > 0 else 0

    print(f"\n{'='*60}")
    print(f"ACCURACY REPORT")
    print(f"{'='*60}\n")

    print(f"Pre-Critic Results:")
    print(f"  Total Questions: {total_questions_pre}")
    print(f"  Correct Answers: {total_correct_pre}")
    print(f"  Accuracy: {accuracy_pre:.2f}%\n")

    if total_questions_post > 0:
        print(f"Post-Critic Results:")
        print(f"  Total Questions: {total_questions_post}")
        print(f"  Correct Answers: {total_correct_post}")
        print(f"  Accuracy: {accuracy_post:.2f}%\n")
    else:
        print(f"Post-Critic Results: No data available\n")

    if mismatches:
        print(f"{'='*60}")
        print(f"MISMATCHES FOUND: {len(mismatches)} questions")
        print(f"{'='*60}\n")
        print("The 'is_correct' field doesn't match actual correctness for:")
        for m in mismatches[:10]:  # Show first 10
            print(f"\n  UID: {m['uid']}")
            print(f"    Predicted (raw): {m['predicted']} (type: {type(m['predicted']).__name__})")
            print(f"    Predicted (int): {m['predicted_int']}")
            print(f"    Correct Index: {m['correct_idx']}")
            print(f"    is_correct field: {m['is_correct_field']}")
            print(f"    Actually correct: {m['actual_correct']}")
        if len(mismatches) > 10:
            print(f"\n  ... and {len(mismatches) - 10} more")
    else:
        print("✓ No mismatches found - all 'is_correct' fields are accurate")

    return {
        'pre_critic': {
            'total': total_questions_pre,
            'correct': total_correct_pre,
            'accuracy': accuracy_pre
        },
        'post_critic': {
            'total': total_questions_post,
            'correct': total_correct_post,
            'accuracy': accuracy_post
        },
        'mismatches': len(mismatches)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_accuracy.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    check_accuracy(results_file)