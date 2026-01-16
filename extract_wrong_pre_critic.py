#!/usr/bin/env python3
"""
Extract all questions where pre-critic was WRONG from lvbench results.
Includes evidence frames, reasoning trace, answers, and question details.
"""

import json
import sys
from pathlib import Path


def extract_wrong_pre_critic(file_path):
    """Extract questions where pre-critic was wrong."""

    with open(file_path, 'r') as f:
        data = json.load(f)

    wrong_questions = []

    for video in data:
        video_id = video.get('video_id')

        # Get pre-critic answers
        pre_critic_answers = video.get('pre_critic_answers', [])

        for pre_q in pre_critic_answers:
            # Check if pre-critic was wrong
            pre_ans = str(pre_q.get('predicted_answer', '')).strip()
            correct_idx = str(pre_q.get('correct_choice_idx', '')).strip()
            is_correct = pre_q.get('is_correct', False)

            # Filter for wrong answers
            if not is_correct or pre_ans != correct_idx:
                # Extract requested fields
                wrong_q = {
                    'video_id': video_id,
                    'uid': pre_q.get('uid'),
                    'question': pre_q.get('question'),
                    'candidates': pre_q.get('candidates', []),
                    'pre_critic_answer': pre_ans,
                    'pre_critic_reasoning': pre_q.get('reasoning', ''),
                    'correct_choice_idx': correct_idx,
                    'correct_answer': pre_q.get('correct_answer', ''),
                    'evidence_frames': pre_q.get('evidence_frames', []),
                }

                wrong_questions.append(wrong_q)

    return wrong_questions


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_wrong_pre_critic.py <path_to_results.json> [output.json]")
        print("\nExample:")
        print("  python extract_wrong_pre_critic.py test_results_partial_20260116_075954.json wrong_precritic.json")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Determine output file
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = f"wrong_precritic_{Path(file_path).stem}.json"

    print(f"Reading from: {file_path}")

    wrong_questions = extract_wrong_pre_critic(file_path)

    print(f"\nFound {len(wrong_questions)} questions where pre-critic was WRONG")

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(wrong_questions, f, indent=2)

    print(f"Saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total wrong pre-critic answers: {len(wrong_questions)}")

    # Count unique videos
    unique_videos = len(set(q['video_id'] for q in wrong_questions))
    print(f"Across {unique_videos} videos")

    return wrong_questions


if __name__ == "__main__":
    main()
