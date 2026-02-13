#!/usr/bin/env python3
"""
Generate CES_logs and global_summary from clip captions using Kimi.
Processes captions in kimi/{dataset}/video_files/{video_id}/captions/clip_captions.json

Usage:
    python generate_kimi_summaries.py --dataset longvideobench
    python generate_kimi_summaries.py --dataset all
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


async def generate_summary_with_kimi(client, prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
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
    """Generate CES_logs and global_summary for a single video."""
    video_files_dir = dataset_dir / 'video_files' / video_id
    captions_dir = video_files_dir / 'captions'

    # Check if captions exist
    captions_file = captions_dir / 'clip_captions.json'
    if not captions_file.exists():
        print(f"  ✗ {video_id}: No clip_captions.json found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No captions file'}

    # Check if summaries already exist
    ces_log_file = captions_dir / 'CES_logs.txt'
    global_summary_file = captions_dir / 'global_summary.txt'

    if ces_log_file.exists() and global_summary_file.exists():
        print(f"  ✓ {video_id}: Summaries already exist")
        return {'video_id': video_id, 'status': 'skipped'}

    # Load captions
    with open(captions_file, 'r') as f:
        clip_captions = json.load(f)

    if not clip_captions:
        print(f"  ✗ {video_id}: No captions in file")
        return {'video_id': video_id, 'status': 'error', 'error': 'Empty captions file'}

    print(f"  Processing {video_id}: {len(clip_captions)} captions")

    # Format captions data with timestamps
    captions_data = []
    for clip in clip_captions:
        start_sec = int(clip['start'])
        end_sec = int(clip['end'])
        captions_data.append(f"[{start_sec}s-{end_sec}s]: {clip['caption']}")

    captions_text = "\n".join(captions_data)

    # Generate CES logs
    print(f"    Generating CES logs...")
    ces_prompt = CES_log_prompt(captions_text)
    ces_log = await generate_summary_with_kimi(client, ces_prompt, model)

    if ces_log:
        with open(ces_log_file, 'w') as f:
            f.write(ces_log)
        print(f"    ✓ CES_logs.txt generated")
    else:
        print(f"    ✗ Failed to generate CES logs")

    # Generate global summary
    print(f"    Generating global summary...")
    summary_prompt = global_summary_prompt(captions_text)
    global_summary = await generate_summary_with_kimi(client, summary_prompt, model)

    if global_summary:
        with open(global_summary_file, 'w') as f:
            f.write(global_summary)
        print(f"    ✓ global_summary.txt generated")
    else:
        print(f"    ✗ Failed to generate global summary")

    if ces_log and global_summary:
        print(f"  ✓ {video_id}: Summaries generated")
        return {'video_id': video_id, 'status': 'success'}
    else:
        return {'video_id': video_id, 'status': 'partial', 'error': 'Some summaries failed'}


async def process_dataset(dataset_name, base_dir, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """Process all videos in a dataset."""
    dataset_dir = Path(base_dir) / 'kimi' / dataset_name
    video_files_dir = dataset_dir / 'video_files'

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    # Get list of video IDs
    video_ids = [d.name for d in video_files_dir.iterdir() if d.is_dir()]

    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    # Load API key and create client
    env = load_env()
    client = AsyncTogether(api_key=env['together_key'])

    # Process videos sequentially (to avoid rate limits)
    results = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}]")
        result = await generate_summaries_for_video(video_id, dataset_dir, client, model)
        results.append(result)
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = len(results) - successful - skipped

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description='Generate summaries from clip captions')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                        help='Model endpoint for summary generation')
    args = parser.parse_args()

    print("="*60)
    print("Kimi Summary Generation")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model}")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        asyncio.run(process_dataset(dataset, args.base_dir, args.model))

    print("\n" + "="*60)
    print("SUMMARY GENERATION COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Modify pipeline for clip caption search")
    print("  2. Test pipeline on single video")
    print("="*60)


if __name__ == '__main__':
    main()
