#!/usr/bin/env python3
"""
Generate CES_logs and global_summary from clip_frame_captions.json using LLM.
Processes captions in kimi/{dataset}/video_files/{video_id}/clip_frame_captions.json

Usage:
    python generate_summaries_from_frame_captions.py --dataset lvbench
    python generate_summaries_from_frame_captions.py --dataset all
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from together import AsyncTogether


# Import prompt templates
sys.path.insert(0, str(Path(__file__).parent))
from prompts import CES_log_prompt, global_summary_prompt


def load_env():
    """Load API keys from env.json."""
    env_path = Path(__file__).parent / 'env.json'
    with open(env_path, 'r') as f:
        return json.load(f)


async def generate_summary_with_llm(client, prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """Generate summary using Together AI."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    Error generating summary: {e}")
        return None


async def generate_summaries_for_video(video_id, dataset_dir, client, model):
    """Generate CES_logs and global_summary for a single video from clip_frame_captions."""
    video_files_dir = dataset_dir / 'video_files' / video_id
    captions_dir = video_files_dir / 'captions'
    captions_dir.mkdir(exist_ok=True)

    # Check if clip_frame_captions exist
    frame_captions_file = video_files_dir / 'clip_frame_captions.json'
    if not frame_captions_file.exists():
        print(f"  ✗ {video_id}: No clip_frame_captions.json found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No frame captions file'}

    # Check if summaries already exist
    ces_log_file = captions_dir / 'CES_logs.txt'
    global_summary_file = captions_dir / 'global_summary.txt'

    if ces_log_file.exists() and global_summary_file.exists():
        print(f"  ✓ {video_id}: Summaries already exist")
        return {'video_id': video_id, 'status': 'skipped'}

    # Load frame captions
    with open(frame_captions_file, 'r') as f:
        frame_captions = json.load(f)

    if not frame_captions:
        print(f"  ✗ {video_id}: No captions in file")
        return {'video_id': video_id, 'status': 'error', 'error': 'Empty captions file'}

    print(f"  Processing {video_id}: {len(frame_captions)} frame clips")

    # Format captions with timestamps for prompts
    captions_data = []
    for clip in frame_captions:
        start = int(clip['start'])
        end = int(clip['end'])
        captions_data.append(f"[{start}s-{end}s] {clip['caption']}")

    captions_text = "\n".join(captions_data)

    # Generate CES logs
    print(f"    Generating CES logs...")
    ces_prompt = CES_log_prompt(captions_text)
    ces_logs = await generate_summary_with_llm(client, ces_prompt, model)

    if not ces_logs:
        print(f"  ✗ {video_id}: Failed to generate CES logs")
        return {'video_id': video_id, 'status': 'error', 'error': 'CES generation failed'}

    # Generate global summary
    print(f"    Generating global summary...")
    summary_prompt = global_summary_prompt(captions_text)
    global_summary = await generate_summary_with_llm(client, summary_prompt, model)

    if not global_summary:
        print(f"  ✗ {video_id}: Failed to generate global summary")
        return {'video_id': video_id, 'status': 'error', 'error': 'Summary generation failed'}

    # Save summaries
    with open(ces_log_file, 'w') as f:
        f.write(ces_logs)

    with open(global_summary_file, 'w') as f:
        f.write(global_summary)

    print(f"  ✓ {video_id}: Generated summaries")

    return {
        'video_id': video_id,
        'status': 'success'
    }


async def process_dataset(dataset_name, base_dir, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """Process all videos in a dataset."""
    dataset_dir = Path(base_dir) / 'kimi' / dataset_name
    video_files_dir = dataset_dir / 'video_files'

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    # Get list of video IDs
    video_ids = sorted([d.name for d in video_files_dir.iterdir() if d.is_dir()])

    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    # Load API key
    env = load_env()
    client = AsyncTogether(api_key=env['together_key'])

    results = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}]")
        result = await generate_summaries_for_video(video_id, dataset_dir, client, model)
        results.append(result)

    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


async def main():
    parser = argparse.ArgumentParser(description='Generate summaries from clip_frame_captions')
    parser.add_argument('--dataset', choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                       default='all', help='Dataset to process')
    parser.add_argument('--base-dir', default='/mnt/ssd/data',
                       help='Base directory (default: /mnt/ssd/data)')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                       help='LLM model for summaries')

    args = parser.parse_args()

    print("=" * 60)
    print("Summary Generation from Clip Frame Captions")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model}")
    print()

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        await process_dataset(dataset, args.base_dir, args.model)

    print("\n" + "=" * 60)
    print("SUMMARY GENERATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
