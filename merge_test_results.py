#!/usr/bin/env python3
"""
Merge multiple test_results_partial JSON files into a single combined result.

Usage:
    python3 merge_test_results.py forward_run/test_results_partial_*.json reverse_run/test_results_partial_*.json -o combined_results.json
"""

import json
import argparse
from pathlib import Path

def merge_results(file_paths, output_path):
    """
    Merge multiple test result JSON files into one.

    Args:
        file_paths: List of paths to test result JSON files
        output_path: Path to save merged results
    """
    all_results = []
    video_ids_seen = set()

    print(f"Merging {len(file_paths)} files...")

    for file_path in file_paths:
        print(f"  Loading: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # data should be a list of video results
            if not isinstance(data, list):
                print(f"  ⚠️  Skipping {file_path}: not a list")
                continue

            # Add videos that haven't been seen yet
            for video_result in data:
                video_id = video_result.get('video_id')
                if video_id and video_id not in video_ids_seen:
                    all_results.append(video_result)
                    video_ids_seen.add(video_id)
                elif video_id:
                    print(f"  ⚠️  Duplicate video {video_id} - keeping first occurrence")

        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            continue

    # Sort by video_id for consistency
    all_results = sorted(all_results, key=lambda v: v.get('video_id', ''))

    # Calculate summary statistics
    total_videos = len(all_results)
    total_questions = sum(v.get('num_questions', 0) for v in all_results)

    print(f"\n{'='*60}")
    print(f"MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"Total questions: {total_questions}")

    # Calculate accuracies if available
    pre_correct = 0
    post_correct = 0
    total_q = 0

    for video in all_results:
        if 'accuracy_pre_critic' in video and 'accuracy_post_critic' in video:
            num_q = video.get('num_questions', 0)
            pre_correct += video['accuracy_pre_critic'] * num_q
            post_correct += video['accuracy_post_critic'] * num_q
            total_q += num_q

    if total_q > 0:
        print(f"\nAccuracy Summary:")
        print(f"  Pre-critic:  {pre_correct:.0f}/{total_q} ({pre_correct/total_q*100:.2f}%)")
        print(f"  Post-critic: {post_correct:.0f}/{total_q} ({post_correct/total_q*100:.2f}%)")
        print(f"  Improvement: {(post_correct - pre_correct)/total_q*100:+.2f}%")

    print(f"{'='*60}")

    # Save merged results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Merged results saved to: {output_path}")
    print(f"   {total_videos} videos, {total_questions} questions")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple test_results_partial JSON files"
    )

    parser.add_argument('files', nargs='+', help='JSON files to merge (supports wildcards)')
    parser.add_argument('-o', '--output', default='merged_results.json',
                       help='Output file path (default: merged_results.json)')

    args = parser.parse_args()

    # Expand file paths (handle wildcards)
    import glob
    file_paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            file_paths.extend(matches)
        else:
            print(f"⚠️  No files found matching: {pattern}")

    if not file_paths:
        print("❌ No input files found!")
        return 1

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in file_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    merge_results(unique_paths, args.output)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
