#!/usr/bin/env python3
"""
Generate captions for frame sequences using Together API with Kimi model.
Groups frames by scene-change timestamps and sends to kjadel/moonshotai/Kimi-K2.5-9b8c5484.

This creates "fake clips" from frames at 1 FPS and captions them using Together API.

Usage:
    python caption_frames_together.py --dataset longvideobench
    python caption_frames_together.py --dataset all --max-concurrent 5
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
    """Load API keys from env_kristine.json."""
    env_path = Path(__file__).parent / 'env_kristine.json'
    with open(env_path, 'r') as f:
        return json.load(f)


async def encode_image_base64(image_path):
    """Encode image file to base64."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')


async def caption_frames_with_together_kimi(together_api_key, frame_paths, start_time, end_time, max_retries=3, max_frames=15):
    """
    Generate caption for a sequence of frames using Together API with Kimi model.

    Args:
        together_api_key: Together API key
        frame_paths: List of frame file paths in order
        start_time: Start time in seconds
        end_time: End time in seconds
        max_retries: Maximum number of retry attempts
        max_frames: Maximum number of frames to send (default: 15)

    Returns:
        Caption string or None on failure
    """

    # Limit number of frames to avoid overwhelming the API
    if len(frame_paths) > max_frames:
        # Sample frames evenly across the sequence
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    # Create prompt for detailed caption
    prompt = f"You are viewing {len(frame_paths)} frames from a video segment (from {start_time:.1f}s to {end_time:.1f}s). Describe what happens in this sequence of frames. Be detailed but concise, covering actions, people/objects, setting, any visible text, and mood."

    # Prepare API request to Together AI
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }

    # Build content array with frames in order with timestamps
    content = [{"type": "text", "text": prompt}]

    for frame_path in frame_paths:
        try:
            # Encode image to base64
            image_b64 = await encode_image_base64(frame_path)

            # Extract frame number from filename (e.g., frame_0001.jpg -> 1)
            frame_name = Path(frame_path).stem
            frame_num = int(frame_name.split('_')[-1])

            # Calculate approximate timestamp for this frame (frames are at 1 fps)
            # Frame number corresponds to seconds into the video
            frame_time = frame_num

            # Add frame with timestamp label
            content.append({
                "type": "text",
                "text": f"\n--- Frame at {frame_time}s ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        except Exception as e:
            print(f"      Warning: Could not encode frame {frame_path}: {e}")
            continue

    payload = {
        "model": "moonshotai/Kimi-K2.5",
        "messages": [{
            "role": "user",
            "content": content
        }],
        "temperature": 1.0,
        "max_tokens": 2048
    }

    # Call Together API with retries
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
                    if response.status == 200:
                        result = await response.json()
                        caption = result['choices'][0]['message']['content']
                        return caption
                    else:
                        error = await response.text()
                        print(f"      Error {response.status}: {error[:200]}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"      Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            return None
        except Exception as e:
            print(f"      Error captioning frames: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"      Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                return None

    return None


async def caption_video_frame_clips(video_id, dataset_dir, together_api_key, semaphore):
    """Caption frame sequences for a single video based on clip timestamps."""
    video_files_dir = dataset_dir / 'video_files' / video_id
    clips_dir = video_files_dir / 'clips'
    frames_dir = video_files_dir / 'frames'

    # Check if frames exist
    if not frames_dir.exists():
        print(f"  ✗ {video_id}: No frames directory found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No frames directory'}

    # Check if clips metadata exists (to get scene timestamps)
    metadata_file = clips_dir / 'clips_metadata.json'
    if not metadata_file.exists():
        print(f"  ✗ {video_id}: No clips_metadata.json found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No clips metadata'}

    # Check if captions already exist
    captions_file = video_files_dir / 'clip_frame_captions.json'
    if captions_file.exists():
        with open(captions_file, 'r') as f:
            existing_captions = json.load(f)
        print(f"  ✓ {video_id}: Frame captions already exist ({len(existing_captions)} clips)")
        return {'video_id': video_id, 'status': 'skipped', 'captions_generated': len(existing_captions)}

    # Load clips metadata to get scene timestamps
    with open(metadata_file, 'r') as f:
        clips_metadata = json.load(f)

    # Get all frame files sorted by name
    all_frames = sorted(frames_dir.glob('frame_*.jpg'))
    if not all_frames:
        print(f"  ✗ {video_id}: No frames found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No frames found'}

    print(f"  Processing {video_id}: {len(clips_metadata)} clip sequences, {len(all_frames)} frames")

    # Caption each "fake clip" (frames grouped by scene timestamps) with rate limiting
    clip_frame_captions = []
    for i, clip_info in enumerate(clips_metadata, 1):
        start_sec = int(clip_info['start'])
        end_sec = int(clip_info['end'])

        # Collect frames that fall within this time range (frames are at 1 fps)
        # Frame N corresponds to second N (frame_0001.jpg = 1 second)
        clip_frames = []
        for frame_path in all_frames:
            frame_name = frame_path.stem
            try:
                frame_num = int(frame_name.split('_')[-1])
                # Check if this frame falls within the clip time range
                if start_sec <= frame_num <= end_sec:
                    clip_frames.append(str(frame_path))
            except (ValueError, IndexError):
                continue

        if not clip_frames:
            print(f"    [{i}/{len(clips_metadata)}] No frames found for clip {start_sec}-{end_sec}s, skipping")
            continue

        print(f"    [{i}/{len(clips_metadata)}] Captioning {len(clip_frames)} frames for clip {start_sec}-{end_sec}s...")

        async with semaphore:
            caption = await caption_frames_with_together_kimi(
                together_api_key,
                clip_frames,
                clip_info['start'],
                clip_info['end']
            )

        if caption:
            clip_frame_captions.append({
                'start': clip_info['start'],
                'end': clip_info['end'],
                'duration': clip_info['duration'],
                'num_frames': len(clip_frames),
                'caption': caption
            })
            print(f"    [{i}/{len(clips_metadata)}] ✓ Generated caption")
        else:
            print(f"    [{i}/{len(clips_metadata)}] ✗ Failed to generate caption")

    # Save captions
    with open(captions_file, 'w') as f:
        json.dump(clip_frame_captions, f, indent=2)

    print(f"  ✓ {video_id}: Generated {len(clip_frame_captions)} frame-based captions")

    return {
        'video_id': video_id,
        'status': 'success',
        'captions_generated': len(clip_frame_captions)
    }


async def process_dataset(dataset_name, base_dir, max_concurrent=3):
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
    together_api_key = env['together_key']

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process all videos concurrently
    tasks = [caption_video_frame_clips(vid_id, dataset_dir, together_api_key, semaphore) for vid_id in video_ids]
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
    parser = argparse.ArgumentParser(description='Caption frame sequences using Together API with Kimi model')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory')
    parser.add_argument('--max-concurrent', type=int, default=3,
                        help='Maximum concurrent API requests (default: 3)')
    args = parser.parse_args()

    print("="*60)
    print("Frame-Based Clip Captioning (Together API + Kimi)")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Model: moonshotai/Kimi-K2.5")
    print(f"Frame rate: 1 FPS (grouped by scene timestamps)")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        asyncio.run(process_dataset(dataset, args.base_dir, args.max_concurrent))

    print("\n" + "="*60)
    print("FRAME CAPTIONING COMPLETE")
    print("="*60)
    print("Output: clip_frame_captions.json in each video folder")
    print("Next step: Embed these captions for retrieval")
    print("="*60)


if __name__ == '__main__':
    main()
