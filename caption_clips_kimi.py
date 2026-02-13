#!/usr/bin/env python3
"""
Generate captions for video clips using Kimi API via Together AI.
Processes clips in kimi/{dataset}/video_files/{video_id}/clips/

Usage:
    python caption_clips_kimi.py --dataset longvideobench
    python caption_clips_kimi.py --dataset all --max-concurrent 10
"""

import os
import sys
import json
import base64
import asyncio
import argparse
from pathlib import Path
import aiohttp


def load_env():
    """Load API keys from env.json."""
    env_path = Path(__file__).parent / 'env.json'
    with open(env_path, 'r') as f:
        return json.load(f)


async def encode_video_base64(video_path):
    """Encode video file to base64."""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
        return base64.b64encode(video_bytes).decode('utf-8')


async def caption_clip_with_kimi(kimi_api_key, video_path):
    """
    Generate caption for a video clip using direct Kimi/Moonshot API.

    Args:
        kimi_api_key: Kimi API key
        video_path: Path to video clip

    Returns:
        Caption string
    """
    import aiohttp

    # Encode video to base64
    video_b64 = await encode_video_base64(video_path)

    # Create prompt for detailed caption
    prompt = "Describe what happens in this video clip. Be detailed but concise, covering actions, people/objects, setting, any visible text, and mood."

    # Prepare API request
    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kimi_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "kimi-k2.5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
            ]
        }],
        "temperature": 1.0,
        "max_tokens": 2048
    }

    # Call Kimi API directly
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    result = await response.json()
                    caption = result['choices'][0]['message']['content']
                    return caption
                else:
                    error = await response.text()
                    print(f"      Error {response.status}: {error[:200]}")
                    return None
    except Exception as e:
        print(f"      Error captioning: {e}")
        return None


async def caption_video_clips(video_id, dataset_dir, kimi_api_key, semaphore):
    """Caption all clips for a single video."""
    video_files_dir = dataset_dir / 'video_files' / video_id
    clips_dir = video_files_dir / 'clips'
    captions_dir = video_files_dir / 'captions'
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Check if clips exist
    if not clips_dir.exists():
        print(f"  ✗ {video_id}: No clips directory found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No clips directory'}

    # Check if captions already exist
    captions_file = captions_dir / 'clip_captions.json'
    if captions_file.exists():
        with open(captions_file, 'r') as f:
            existing_captions = json.load(f)
        print(f"  ✓ {video_id}: Captions already exist ({len(existing_captions)} clips)")
        return {'video_id': video_id, 'status': 'skipped', 'captions_generated': len(existing_captions)}

    # Load clips metadata
    metadata_file = clips_dir / 'clips_metadata.json'
    if not metadata_file.exists():
        print(f"  ✗ {video_id}: No clips_metadata.json found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No clips metadata'}

    with open(metadata_file, 'r') as f:
        clips_metadata = json.load(f)

    print(f"  Processing {video_id}: {len(clips_metadata)} clips")

    # Caption each clip with rate limiting
    clip_captions = []
    for i, clip_info in enumerate(clips_metadata, 1):
        clip_path = clips_dir / clip_info['filename']

        if not clip_path.exists():
            print(f"    [{i}/{len(clips_metadata)}] Warning: {clip_info['filename']} not found")
            continue

        print(f"    [{i}/{len(clips_metadata)}] Captioning {clip_info['filename']}...")

        async with semaphore:
            caption = await caption_clip_with_kimi(kimi_api_key, clip_path)

        if caption:
            clip_captions.append({
                'filename': clip_info['filename'],
                'start': clip_info['start'],
                'end': clip_info['end'],
                'duration': clip_info['duration'],
                'caption': caption
            })

    # Save captions
    with open(captions_file, 'w') as f:
        json.dump(clip_captions, f, indent=2)

    print(f"  ✓ {video_id}: Generated {len(clip_captions)} captions")

    return {
        'video_id': video_id,
        'status': 'success',
        'captions_generated': len(clip_captions)
    }


async def process_dataset(dataset_name, base_dir, max_concurrent=10):
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

    # Load API key
    env = load_env()
    kimi_api_key = env['kimi_api_key']

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process all videos concurrently
    tasks = [caption_video_clips(vid_id, dataset_dir, kimi_api_key, semaphore) for vid_id in video_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'skipped')
    errors = len(results) - successful - skipped

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description='Caption video clips using direct Kimi/Moonshot API')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Maximum concurrent API requests (default: 5 for Kimi API)')
    args = parser.parse_args()

    print("="*60)
    print("Kimi Clip Captioning (Direct Moonshot API)")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Model: kimi-k2.5")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        asyncio.run(process_dataset(dataset, args.base_dir, args.max_concurrent))

    print("\n" + "="*60)
    print("CAPTIONING COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Run embed_clip_captions.py to generate embeddings")
    print("  2. Run generate_kimi_summaries.py to generate summaries")
    print("="*60)


if __name__ == '__main__':
    main()
