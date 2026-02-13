#!/usr/bin/env python3
"""
Setup script for Kimi dataset processing.
Creates folder structure and extracts 100 questions from each dataset.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_longvideobench_questions(json_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load and filter longvideobench questions."""
    with open(json_path, 'r') as f:
        all_questions = json.load(f)

    # Take first 100 questions
    selected = all_questions[:limit]

    # Extract unique video IDs
    video_ids = list(set([q['video_id'] for q in selected]))

    print(f"Longvideobench: {len(selected)} questions from {len(video_ids)} videos")
    return selected, video_ids


def load_lvbench_questions(json_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load and filter lvbench questions."""
    with open(json_path, 'r') as f:
        content = f.read()

    # Fix malformed JSON (starts with { instead of [)
    if content.strip().startswith('{'):
        content = '[' + content.strip()[1:-1] + ']'

    all_questions = json.loads(content)

    # Take first 100 questions
    selected = all_questions[:limit]

    # Extract unique video IDs
    video_ids = list(set([q['video_id'] for q in selected]))

    print(f"LVBench: {len(selected)} questions from {len(video_ids)} videos")
    return selected, video_ids


def load_videomme_questions(json_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load and filter videomme questions."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # The file is a dict with video_id keys and question lists as values
    all_questions = []
    for video_id, questions in data.items():
        for q in questions:
            q['video_id'] = video_id
            all_questions.append(q)

    # Take first 100 questions
    selected = all_questions[:limit]

    # Extract unique video IDs
    video_ids = list(set([q['video_id'] for q in selected]))

    print(f"VideoMME: {len(selected)} questions from {len(video_ids)} videos")
    return selected, video_ids


def find_video_in_processed_dirs(video_id: str, base_dir: Path) -> Path:
    """Find video folder in videos_processed_* directories."""
    # Check all videos_processed_* and lvbench_videos_processed* directories
    for pattern in ["videos_processed*", "lvbench_videos_processed*"]:
        for processed_dir in base_dir.glob(pattern):
            video_dir = processed_dir / video_id
            if video_dir.exists():
                return video_dir

    return None


def setup_dataset(dataset_name: str, questions: List[Dict], video_ids: List[str],
                  kimi_base: Path, source_base: Path, use_symlink: bool = True):
    """Setup folder structure and copy/link videos for a dataset."""

    dataset_dir = kimi_base / dataset_name
    videos_dir = dataset_dir / "videos"
    video_files_dir = dataset_dir / "video_files"

    # Create directories
    dataset_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    video_files_dir.mkdir(exist_ok=True)

    print(f"\nSetting up {dataset_name}...")
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  Processing {len(video_ids)} videos...")

    # Track successful video copies
    found_videos = []
    missing_videos = []

    for i, video_id in enumerate(video_ids, 1):
        # Find source video directory
        source_video_dir = find_video_in_processed_dirs(video_id, source_base)

        if source_video_dir is None:
            print(f"  [{i}/{len(video_ids)}] Warning: Could not find {video_id}")
            missing_videos.append(video_id)
            continue

        # Create target directory
        target_video_dir = video_files_dir / video_id
        target_video_dir.mkdir(exist_ok=True)

        # Find the actual video file (.mp4)
        video_files = list(source_video_dir.glob("*.mp4"))
        if not video_files:
            # Try parent directories (videos/ and root)
            video_files = list(source_base.glob(f"videos/{video_id}.mp4"))
        if not video_files:
            video_files = list(source_base.glob(f"*/{video_id}.mp4"))

        if video_files:
            source_video_file = video_files[0]
            target_video_file = videos_dir / source_video_file.name

            # Copy or symlink video file
            if use_symlink and not target_video_file.exists():
                target_video_file.symlink_to(source_video_file.absolute())
                print(f"  [{i}/{len(video_ids)}] Linked {video_id}")
            elif not target_video_file.exists():
                shutil.copy2(source_video_file, target_video_file)
                print(f"  [{i}/{len(video_ids)}] Copied {video_id}")
            else:
                print(f"  [{i}/{len(video_ids)}] Exists {video_id}")

            found_videos.append(video_id)
        else:
            print(f"  [{i}/{len(video_ids)}] Warning: No .mp4 file for {video_id}")
            missing_videos.append(video_id)

    # Filter questions to only include videos we found
    filtered_questions = [q for q in questions if q['video_id'] in found_videos]

    # Save filtered questions
    questions_file = dataset_dir / "downloaded_questions.json"
    with open(questions_file, 'w') as f:
        json.dump(filtered_questions, f, indent=2)

    print(f"\n{dataset_name} Summary:")
    print(f"  Found: {len(found_videos)} videos")
    print(f"  Missing: {len(missing_videos)} videos")
    print(f"  Questions saved: {len(filtered_questions)}")
    print(f"  Output: {questions_file}")

    if missing_videos:
        print(f"  Missing video IDs: {missing_videos[:10]}{'...' if len(missing_videos) > 10 else ''}")

    return len(found_videos), len(missing_videos)


def main():
    parser = argparse.ArgumentParser(description='Setup Kimi dataset processing folders')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory containing datasets')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of questions to extract per dataset')
    parser.add_argument('--copy', action='store_true',
                        help='Copy videos instead of symlinking')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    kimi_base = base_dir / "kimi"

    print("="*60)
    print("Kimi Dataset Setup")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Questions per dataset: {args.limit}")
    print(f"Mode: {'Copy' if args.copy else 'Symlink'}")
    print()

    # Create main kimi directory
    kimi_base.mkdir(exist_ok=True)

    # Process each dataset
    datasets = [
        {
            'name': 'longvideobench',
            'questions_path': base_dir / 'longvideobench' / 'lvb_val.json',
            'source_base': base_dir / 'longvideobench',
            'loader': load_longvideobench_questions
        },
        {
            'name': 'lvbench',
            'questions_path': base_dir / 'lvbench' / 'lvbench_questions.json',
            'source_base': base_dir / 'lvbench',
            'loader': load_lvbench_questions
        },
        {
            'name': 'videomme',
            'questions_path': base_dir / 'videomme' / 'videomme_questions.json',
            'source_base': base_dir / 'videomme',
            'loader': load_videomme_questions
        }
    ]

    total_found = 0
    total_missing = 0

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset['name'].upper()}")
        print(f"{'='*60}")

        # Load questions
        questions, video_ids = dataset['loader'](str(dataset['questions_path']), args.limit)

        # Setup dataset
        found, missing = setup_dataset(
            dataset['name'],
            questions,
            video_ids,
            kimi_base,
            dataset['source_base'],
            use_symlink=not args.copy
        )

        total_found += found
        total_missing += missing

    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Total videos found: {total_found}")
    print(f"Total videos missing: {total_missing}")
    print(f"Success rate: {100*total_found/(total_found+total_missing):.1f}%")
    print()
    print("Next steps:")
    print("  1. Run extract_clips_and_frames.py on kimi/*/video_files/")
    print("  2. Run caption_clips_kimi.py to generate clip captions")
    print("  3. Run embed_clip_captions.py to generate embeddings")
    print("="*60)


if __name__ == '__main__':
    main()
