#!/usr/bin/env python3
"""
Generate captions for frame sequences or clip videos using Kimi-style models.

Frame mode (default):
  - Caption frame groups between scene timestamps from clips_metadata.json
  - Produces clip_frame_captions.json in each video folder

Clip mode (--clip):
  - Caption actual extracted clip videos (clips/*.mp4)
  - Produces captions/clip_captions.json in each video folder
"""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import aiohttp


TOGETHER_KIMI_MODEL = "moonshotai/Kimi-K2.5"


def load_together_env():
    """Load Together API keys from env_kristine.json."""
    env_path = Path(__file__).parent / "env_kristine.json"
    with open(env_path, "r") as f:
        return json.load(f)


def load_kimi_env():
    """Load Kimi API keys from env.json."""
    env_path = Path(__file__).parent / "env.json"
    with open(env_path, "r") as f:
        return json.load(f)


async def encode_image_base64(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")


async def encode_video_base64(video_path):
    """Encode video file to base64."""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        return base64.b64encode(video_bytes).decode("utf-8")


async def caption_frames_with_together_kimi(together_api_key, frame_paths, start_time, end_time, max_retries=3, max_frames=15):
    """
    Generate caption for a sequence of frames using Together AI with Kimi model.
    """

    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    prompt = (
        f"You are viewing {len(frame_paths)} frames from a video segment "
        f"(from {start_time:.1f}s to {end_time:.1f}s). Describe what happens in this sequence of frames. "
        "Be detailed but concise, covering actions, people/objects, setting, any visible text, and mood."
    )

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json",
    }

    content = [{"type": "text", "text": prompt}]
    for frame_path in frame_paths:
        try:
            image_b64 = await encode_image_base64(frame_path)
            frame_name = Path(frame_path).stem
            frame_num = int(frame_name.split("_")[-1])
            frame_time = frame_num

            content.append({"type": "text", "text": f"\n--- Frame at {frame_time}s ---"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        except Exception as e:
            print(f"      Warning: Could not encode frame {frame_path}: {e}")
            continue

    payload = {
        "model": TOGETHER_KIMI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 1.0,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
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


async def caption_clip_with_kimi(kimi_api_key, video_path):
    """Generate caption for a clip using Moonshot/Kimi direct API."""

    video_b64 = await encode_video_base64(video_path)
    prompt = (
        "Describe what happens in this video clip. Be very detailed, covering all actions, "
        "people/objects, what people are wearing, doing, saying, setting, any visible text, "
        "and mood. Describe the scene as well."
    )

    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kimi_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "kimi-k2.5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ],
        }],
        "temperature": 1.0,
        "max_tokens": 2048,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                error = await response.text()
                print(f"      Error {response.status}: {error[:200]}")
                return None
    except Exception as e:
        print(f"      Error captioning clip: {e}")
        return None


async def caption_video_frame_clips(video_id, dataset_dir, together_api_key, semaphore):
    """Caption grouped frame sequences for one video."""
    video_files_dir = dataset_dir / "video_files" / video_id
    clips_dir = video_files_dir / "clips"
    frames_dir = video_files_dir / "frames"
    captions_file = video_files_dir / "clip_frame_captions.json"

    if not frames_dir.exists():
        print(f"  ✗ {video_id}: No frames directory found")
        return {"video_id": video_id, "status": "error", "error": "No frames directory"}

    metadata_file = clips_dir / "clips_metadata.json"
    if not metadata_file.exists():
        print(f"  ✗ {video_id}: No clips_metadata.json found")
        return {"video_id": video_id, "status": "error", "error": "No clips metadata"}

    if captions_file.exists():
        with open(captions_file, "r") as f:
            existing = json.load(f)
        print(f"  ✓ {video_id}: Frame captions already exist ({len(existing)} clips)")
        return {"video_id": video_id, "status": "skipped", "captions_generated": len(existing)}

    with open(metadata_file, "r") as f:
        clips_metadata = json.load(f)

    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not all_frames:
        print(f"  ✗ {video_id}: No frames found")
        return {"video_id": video_id, "status": "error", "error": "No frames found"}

    print(f"  Processing {video_id}: {len(clips_metadata)} clip sequences, {len(all_frames)} frames")

    clip_frame_captions = []
    for i, clip_info in enumerate(clips_metadata, 1):
        start_sec = int(clip_info["start"])
        end_sec = int(clip_info["end"])

        clip_frames = []
        for frame_path in all_frames:
            frame_name = frame_path.stem
            try:
                frame_num = int(frame_name.split("_")[-1])
                if start_sec <= frame_num <= end_sec:
                    clip_frames.append(str(frame_path))
            except (ValueError, IndexError):
                continue

        if not clip_frames:
            print(f"    [{i}/{len(clips_metadata)}] No frames found for clip {start_sec}-{end_sec}s, skipping")
            continue

        print(f"    [{i}/{len(clips_metadata)}] Captioning {len(clip_frames)} frames for clip {start_sec}-{end_sec}s...")
        async with semaphore:
            caption = await caption_frames_with_together_kimi(together_api_key, clip_frames, clip_info["start"], clip_info["end"])

        if caption:
            clip_frame_captions.append({
                "start": clip_info["start"],
                "end": clip_info["end"],
                "duration": clip_info["duration"],
                "num_frames": len(clip_frames),
                "caption": caption,
            })
            print(f"    [{i}/{len(clips_metadata)}] ✓ Generated caption")
        else:
            print(f"    [{i}/{len(clips_metadata)}] ✗ Failed to generate caption")

    with open(captions_file, "w") as f:
        json.dump(clip_frame_captions, f, indent=2)

    print(f"  ✓ {video_id}: Generated {len(clip_frame_captions)} frame-based captions")
    return {"video_id": video_id, "status": "success", "captions_generated": len(clip_frame_captions)}


async def caption_video_clips(video_id, dataset_dir, kimi_api_key, semaphore):
    """Caption extracted clips for one video."""
    video_files_dir = dataset_dir / "video_files" / video_id
    clips_dir = video_files_dir / "clips"
    captions_dir = video_files_dir / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)

    if not clips_dir.exists():
        print(f"  ✗ {video_id}: No clips directory found")
        return {"video_id": video_id, "status": "error", "error": "No clips directory"}

    captions_file = captions_dir / "clip_captions.json"
    if captions_file.exists():
        with open(captions_file, "r") as f:
            existing = json.load(f)
        print(f"  ✓ {video_id}: Captions already exist ({len(existing)} clips)")
        return {"video_id": video_id, "status": "skipped", "captions_generated": len(existing)}

    metadata_file = clips_dir / "clips_metadata.json"
    if not metadata_file.exists():
        print(f"  ✗ {video_id}: No clips_metadata.json found")
        return {"video_id": video_id, "status": "error", "error": "No clips metadata"}

    with open(metadata_file, "r") as f:
        clips_metadata = json.load(f)

    print(f"  Processing {video_id}: {len(clips_metadata)} clips")
    clip_captions = []
    for i, clip_info in enumerate(clips_metadata, 1):
        clip_path = clips_dir / clip_info["filename"]

        if not clip_path.exists():
            print(f"    [{i}/{len(clips_metadata)}] Warning: {clip_info['filename']} not found")
            continue

        print(f"    [{i}/{len(clips_metadata)}] Captioning {clip_info['filename']}...")
        async with semaphore:
            caption = await caption_clip_with_kimi(kimi_api_key, clip_path)

        if caption:
            clip_captions.append({
                "filename": clip_info["filename"],
                "start": clip_info["start"],
                "end": clip_info["end"],
                "duration": clip_info["duration"],
                "caption": caption,
            })

    with open(captions_file, "w") as f:
        json.dump(clip_captions, f, indent=2)

    print(f"  ✓ {video_id}: Generated {len(clip_captions)} captions")
    return {"video_id": video_id, "status": "success", "captions_generated": len(clip_captions)}


def _summarize_results(results):
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    errors = len(results) - successful - skipped
    return successful, skipped, errors


async def process_dataset(dataset_name, base_dir, max_concurrent=3):
    """Process all videos in a dataset using frame-mode captions."""
    dataset_dir = Path(base_dir) / "kimi" / dataset_name
    video_files_dir = dataset_dir / "video_files"

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    video_ids = [d.name for d in video_files_dir.iterdir() if d.is_dir()]
    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    env = load_together_env()
    together_api_key = env["together_key"]

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [caption_video_frame_clips(vid_id, dataset_dir, together_api_key, semaphore) for vid_id in video_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful, skipped, errors = _summarize_results(results)

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


async def process_clip_dataset(dataset_name, base_dir, max_concurrent=5):
    """Process all videos in a dataset using clip-mode captions."""
    dataset_dir = Path(base_dir) / "kimi" / dataset_name
    video_files_dir = dataset_dir / "video_files"

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    video_ids = [d.name for d in video_files_dir.iterdir() if d.is_dir()]
    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    env = load_kimi_env()
    kimi_api_key = env["kimi_api_key"]

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [caption_video_clips(vid_id, dataset_dir, kimi_api_key, semaphore) for vid_id in video_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful, skipped, errors = _summarize_results(results)

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Caption frame sequences using Kimi models")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "longvideobench", "lvbench", "videomme"],
                        help="Dataset to process")
    parser.add_argument("--base-dir", type=str, default="/mnt/ssd/data", help="Base directory")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Maximum concurrent API requests (default: 3 for frame mode, 5 for clip mode)")
    parser.add_argument("--clip", action="store_true", help="Caption extracted clips instead of frame groups")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Kimi Captioning (Frames + Clips)")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Max concurrent: {args.max_concurrent}")

    datasets = ["longvideobench", "lvbench", "videomme"] if args.dataset == "all" else [args.dataset]

    if args.clip:
        print("Mode: clip captioning")
        print(f"Model: kimi-k2.5")
        for dataset in datasets:
            asyncio.run(process_clip_dataset(dataset, args.base_dir, args.max_concurrent))
        print("\n" + "=" * 60)
        print("CLIP CAPTIONING COMPLETE")
        print("=" * 60)
        print("Output: captions/clip_captions.json in each video folder")
        print("Next step: Run embed_frame_captions.py --caption-style clip")
        print("=" * 60)
        return

    print("Mode: frame captioning")
    print(f"Model: {TOGETHER_KIMI_MODEL}")
    print(f"Frame rate: 1 FPS (grouped by scene timestamps)")

    for dataset in datasets:
        asyncio.run(process_dataset(dataset, args.base_dir, args.max_concurrent))

    print("\n" + "=" * 60)
    print("FRAME CAPTIONING COMPLETE")
    print("=" * 60)
    print("Output: clip_frame_captions.json in each video folder")
    print("Next step: Embed these captions for retrieval")
    print("=" * 60)


if __name__ == "__main__":
    main()
