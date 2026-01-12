#!/usr/bin/env python3
"""
Calculate true accuracy from test results, handling string/int conversions
"""
import json
import sys
from pathlib import Path

def convert_answer_to_int(answer):
    """Convert answer to int, handling various formats"""
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        # Try to convert string to int
        answer = answer.strip()
        if answer.isdigit():
            return int(answer)
        # Handle letter answers (A=0, B=1, C=2, D=3)
        if len(answer) == 1 and answer.upper() in 'ABCD':
            return ord(answer.upper()) - ord('A')
    return None

def calculate_is_correct(predicted, correct_idx):
    """Calculate if prediction is correct"""
    predicted_int = convert_answer_to_int(predicted)
    if predicted_int is None:
        return False
    return predicted_int == correct_idx

def process_results(results_file):
    """Process results file and calculate accuracies"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Statistics
    pre_critic_correct = 0
    critic_correct = 0
    num_discrepancies = 0
    judge_correct = 0
    total_questions = 0
    judge_critic_correct = 0
    judge_pre_critic_correct = 0
    judge_incorrect = 0
    post_critic_correct = 0
    critic_fails = 0
    total_judge_questions = 0
    re_eval_critic_correct = 0
    re_eval_pre_critic_correct = 0
    right_stays_right = 0
    right_turns_wrong = 0
    wrong_stays_wrong = 0
    wrong_turns_right = 0

    re_eval_original_correct = 0
    re_eval_original_incorrect = 0

    for video in data:
        video_id = video.get('video_id', 'unknown')
        for question in video['questions']:
            total_questions += 1
            pre_critic_answer = convert_answer_to_int(question.get('pre_critic_answer', None))
            critic_answer = convert_answer_to_int(question.get('critic_answer', None))
            correct_answer = question.get('correct_choice_idx', None)
            re_evaluated_with_judge = question.get('re_evaluated_with_judge', False)
            if re_evaluated_with_judge:
                judge_choice = convert_answer_to_int(question.get('judge_choice', None))
            

            if pre_critic_answer == correct_answer:
                pre_critic_correct += 1
            if critic_answer == correct_answer:
                critic_correct += 1
            if critic_answer not in [0, 1, 2, 3, 4]:
                critic_fails += 1
            if pre_critic_answer != critic_answer:
                num_discrepancies += 1
            if re_evaluated_with_judge:
                if pre_critic_answer == correct_answer:
                    re_eval_original_correct += 1
                    if judge_choice == correct_answer:
                        right_stays_right += 1
                    else:
                        right_turns_wrong += 1
                if pre_critic_answer != correct_answer:
                    re_eval_original_incorrect += 1
                    if judge_choice == correct_answer:
                        wrong_turns_right += 1
                    else:
                        wrong_stays_wrong += 1

                if judge_choice == correct_answer and judge_choice == critic_answer:
                    judge_critic_correct += 1
                    post_critic_correct += 1
                elif judge_choice == correct_answer and judge_choice == pre_critic_answer:
                    judge_pre_critic_correct += 1
                    post_critic_correct += 1
                else:
                    judge_incorrect += 1
            if re_evaluated_with_judge is False and pre_critic_answer == correct_answer:
                post_critic_correct += 1
            if re_evaluated_with_judge:
                total_judge_questions += 1
                if critic_answer == correct_answer:
                    re_eval_critic_correct += 1
                if pre_critic_answer == correct_answer:
                    re_eval_pre_critic_correct += 1 
            
    pre_critic_acc = pre_critic_correct / total_questions * 100
    critic_acc = critic_correct / total_questions * 100
    critic_fails_acc = critic_fails / total_questions * 100
    post_critic_acc = post_critic_correct / total_questions * 100
    judge_acc = (judge_critic_correct + judge_pre_critic_correct) / total_judge_questions * 100
    judge_critic_acc = judge_critic_correct / total_judge_questions * 100
    judge_pre_critic_acc = judge_pre_critic_correct / total_judge_questions * 100
    judge_incorrect_acc = judge_incorrect / total_judge_questions * 100
    re_eval_critic_acc = re_eval_critic_correct / total_judge_questions * 100
    re_eval_pre_critic_acc = re_eval_pre_critic_correct / total_judge_questions * 100
    re_eval_original_acc = re_eval_original_correct / total_judge_questions * 100
    re_eval_original_incorrect_acc = re_eval_original_incorrect / total_judge_questions * 100
    right_stays_right_acc = right_stays_right / total_judge_questions * 100
    right_turns_wrong_acc = right_turns_wrong / total_judge_questions * 100
    wrong_stays_wrong_acc = wrong_stays_wrong / total_judge_questions * 100
    wrong_turns_right_acc = wrong_turns_right / total_judge_questions * 100
    print(f"Right Stays Right: {right_stays_right}/{total_judge_questions} = {right_stays_right_acc:.2f}%")
    print(f"Right Turns Wrong: {right_turns_wrong}/{total_judge_questions} = {right_turns_wrong_acc:.2f}%")
    print(f"Wrong Stays Wrong: {wrong_stays_wrong}/{total_judge_questions} = {wrong_stays_wrong_acc:.2f}%")
    print(f"Wrong Turns Right: {wrong_turns_right}/{total_judge_questions} = {wrong_turns_right_acc:.2f}%")
    print(f"Re-Eval Original Accuracy: {re_eval_original_correct}/{total_judge_questions} = {re_eval_original_acc:.2f}%")
    print(f"Re-Eval Original Incorrect Accuracy: {re_eval_original_incorrect}/{total_judge_questions} = {re_eval_original_incorrect_acc:.2f}%")
    print(f"Pre-Critic Accuracy:  {pre_critic_correct}/{total_questions} = {pre_critic_acc:.2f}%")
    print(f"Critic Accuracy: {critic_correct}/{total_questions} = {critic_acc:.2f}%")
    print(f"Post-Critic Accuracy: {post_critic_correct}/{total_questions} = {post_critic_acc:.2f}%")
    print(f"Judge Accuracy: {judge_critic_correct + judge_pre_critic_correct}/{total_judge_questions} = {judge_acc:.2f}%")
    print(f"Critic Fails: {critic_fails}/{total_questions} = {critic_fails_acc:.2f}%")
    print(f"Re-Eval Critic Accuracy: {re_eval_critic_correct}/{total_judge_questions} = {re_eval_critic_acc:.2f}%")
    print(f"Re-Eval Pre-Critic Accuracy: {re_eval_pre_critic_correct}/{total_judge_questions} = {re_eval_pre_critic_acc:.2f}%")
    print(f"Judge Critic Accuracy: {judge_critic_correct}/{total_judge_questions} = {judge_critic_acc:.2f}%")
    print(f"Judge Pre-Critic Accuracy: {judge_pre_critic_correct}/{total_judge_questions} = {judge_pre_critic_acc:.2f}%")
    print(f"Judge Incorrect Accuracy: {judge_incorrect}/{total_judge_questions} = {judge_incorrect_acc:.2f}%")

    stats = {}
    stats['total_questions'] = total_questions
    stats['pre_critic_correct'] = pre_critic_correct
    stats['critic_correct'] = critic_correct
    stats['post_critic_correct'] = post_critic_correct
    stats['judge_correct'] = judge_correct
    stats['judge_critic_correct'] = judge_critic_correct
    stats['judge_pre_critic_correct'] = judge_pre_critic_correct
    stats['judge_incorrect'] = judge_incorrect
    stats['pre_critic_acc'] = pre_critic_acc
    stats['critic_acc'] = critic_acc
    stats['post_critic_acc'] = post_critic_acc
    stats['judge_acc'] = judge_acc
    stats['judge_critic_acc'] = judge_critic_acc
    stats['judge_pre_critic_acc'] = judge_pre_critic_acc
    stats['judge_incorrect_acc'] = judge_incorrect_acc

    with open(results_file.replace('.json', '_accuracy.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    return {
        'total_questions': total_questions,
        'pre_critic_correct': pre_critic_correct,
        'critic_correct': critic_correct,
        'post_critic_correct': post_critic_correct,
        'judge_correct': judge_correct,
        'judge_critic_correct': judge_critic_correct,
        'judge_pre_critic_correct': judge_pre_critic_correct,
        'judge_incorrect': judge_incorrect,
        'pre_critic_acc': pre_critic_acc,
        'critic_acc': critic_acc,
        'post_critic_acc': post_critic_acc,
        'judge_acc': judge_acc,
        'judge_critic_acc': judge_critic_acc,
        'judge_pre_critic_acc': judge_pre_critic_acc,
        'judge_incorrect_acc': judge_incorrect_acc,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_accuracy.py <test_results.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)

    stats = process_results(results_file)
