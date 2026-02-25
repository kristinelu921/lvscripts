#!/usr/bin/env python3
"""Compatibility wrapper for clip summaries.

This delegates to :mod:`generate_kimi_summaries` so there is one summary implementation.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from together import AsyncTogether

# Ensure parent directory scripts are importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_kimi_summaries import generate_summaries_for_video, load_env


async def _run_video(video_dir: Path, model: str, client):
    dataset_dir = video_dir.parent.parent
    return await generate_summaries_for_video(video_dir.name, dataset_dir, client, model, caption_style="clip")


async def _run_parent(parent_dir: Path, model: str, client):
    video_dirs = [
        item
        for item in parent_dir.iterdir()
        if item.is_dir() and (item / "captions" / "clip_captions.json").exists()
    ]
    if not video_dirs:
        print(f"⚠️  No videos with clip captions found in {parent_dir}")
        return

    success = 0
    for video_dir in video_dirs:
        try:
            result = await _run_video(video_dir, model, client)
            if result.get("status") == "success":
                success += 1
        except Exception as e:
            print(f"❌ Error processing {video_dir.name}: {e}")

    print(f"\n✅ Successfully processed {success}/{len(video_dirs)} videos")


async def main_async(video_path: Path, model: str):
    env = load_env()
    client = AsyncTogether(api_key=env['together_key'])

    if (video_path / "captions" / "clip_captions.json").exists():
        await _run_video(video_path, model, client)
    else:
        await _run_parent(video_path, model, client)


def main():
    parser = argparse.ArgumentParser(description="Generate CES_logs.txt and global_summary.txt from clip captions")
    parser.add_argument("video_path", help="Path to video directory or parent directory containing video folders")
    parser.add_argument("--llm-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                        help="LLM model to use")

    args = parser.parse_args()
    asyncio.run(main_async(Path(args.video_path), args.llm_model))


if __name__ == "__main__":
    main()
