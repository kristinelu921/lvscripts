#!/usr/bin/env python3
"""
Compare original answers with re-evaluated answers
"""
import json
import os
from pathlib import Path

def convert_answer(answer):
    """Convert answer to comparable format"""
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        answer = answer.strip().upper()
        if answer.isdigit():
            return int(answer)
        # Convert A->0, B->1, etc.
        if len(answer) == 1 and answer in 'ABCDEFGH':
            return ord(answer) - ord('A')
    return str(answer)

def compare_video_answers(video_dir):
    """Compare original and re-evaluated answers for a video"""
    video_id = os.path.basename(video_dir)

    # Find answer files
    original_file = os.path.join(video_dir, f"{video_id}_answers_reformatted.json")
    reevaluated_file = os.path.join(video_dir, f"{video_id}_re_evaluated.json")

    if not os.path.exists(original_file):
        return None
    if not os.path.exists(reevaluated_file):
        return None

    # Load files
    try:
        with open(original_file, 'r') as f:
            original = json.load(f)
        with open(reevaluated_file, 'r') as f:
            reevaluated = json.load(f)
    except Exception as e:
        print(f"Error loading {video_id}: {e}")
        return None

    # Compare answers
    changes = []
    for orig in original:
        uid = orig.get('uid')
        orig_answer = convert_answer(orig.get('answer'))

        # Find matching re-evaluated
        reeval = next((r for r in reevaluated if r.get('uid') == uid), None)
        if reeval:
            reeval_answer = convert_answer(reeval.get('answer'))

            if orig_answer != reeval_answer:
                changes.append({
                    'uid': uid,
                    'video_id': video_id,
                    'original': orig_answer,
                    'reevaluated': reeval_answer,
                    'question': orig.get('question', '')[:80] + '...'
                })

    return changes

def main():
    videos_dir = '/mnt/ssh/data/longvideobench/videos_processed'

    all_changes = []
    total_questions = 0
    videos_compared = 0

    print("Comparing original answers with re-evaluated answers...")

    for video_dir in sorted(Path(videos_dir).iterdir()):
        if not video_dir.is_dir():
            continue

        changes = compare_video_answers(str(video_dir))
        if changes is not None:
            videos_compared += 1

            # Count questions
            video_id = video_dir.name
            original_file = video_dir / f"{video_id}_answers_reformatted.json"
            try:
                with open(original_file, 'r') as f:
                    total_questions += len(json.load(f))
            except:
                pass

            if changes:
                all_changes.extend(changes)

    # Print results
    print("\n" + "="*80)
    print("RE-EVALUATION COMPARISON")
    print("="*80)
    print(f"Videos compared: {videos_compared}")
    print(f"Total questions: {total_questions}")
    print(f"Answers changed: {len(all_changes)}")

    if total_questions > 0:
        change_pct = (len(all_changes) / total_questions * 100)
        print(f"Change rate: {change_pct:.1f}%")

    if all_changes:
        print("\n" + "="*80)
        print(f"CHANGED ANSWERS (showing first 20 of {len(all_changes)})")
        print("="*80)

        for change in all_changes[:20]:
            print(f"\n{change['uid']}")
            print(f"  Video: {change['video_id']}")
            print(f"  Question: {change['question']}")
            print(f"  Original: {change['original']}")
            print(f"  Re-evaluated: {change['reevaluated']}")

        if len(all_changes) > 20:
            print(f"\n... and {len(all_changes) - 20} more changes")
    else:
        print("\n✓ No answers were changed by re-evaluation")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
