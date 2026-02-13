#!/usr/bin/env python3
"""
Generate CES_logs.txt and global_summary.txt from clip captions using an LLM.
This is needed for the QA pipeline when using clip-based captions.
"""

import json
import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_example_query import query_llm_async
from prompts import CES_log_prompt, global_summary_prompt


async def generate_summaries_for_video(video_dir, llm_model="kimi-k2.5"):
    """
    Generate CES_logs.txt and global_summary.txt from clip captions.

    Args:
        video_dir: Path to video directory (should contain captions/clip_captions.json)
        llm_model: LLM model to use for generation
    """
    video_dir = Path(video_dir)
    captions_dir = video_dir / "captions"
    clip_captions_file = captions_dir / "clip_captions.json"

    if not clip_captions_file.exists():
        print(f"⚠️  No clip captions found at {clip_captions_file}")
        return False

    # Load clip captions
    with open(clip_captions_file) as f:
        clip_data = json.load(f)

    # Format captions for the prompts
    captions_text = []
    for clip in clip_data:
        clip_name = clip.get('clip_filename', 'unknown')
        start_time = clip.get('start', 0)
        end_time = clip.get('end', 0)

        captions_text.append(f"\n[Clip: {clip_name}, Time: {start_time:.1f}s - {end_time:.1f}s]")

        for caption_type, caption_content in clip['captions'].items():
            captions_text.append(f"\n{caption_type.upper()}:")
            captions_text.append(caption_content[:1000])  # Truncate if too long

    captions_data = "\n".join(captions_text)

    print(f"Generating summaries for {video_dir.name} using {llm_model}...")

    # Generate CES logs
    print("  Generating CES logs (Character/Event/Scene)...")
    ces_prompt = CES_log_prompt(captions_data)
    ces_logs = await query_llm_async(ces_prompt, model_choice=llm_model, temperature=0.3)

    ces_logs_path = captions_dir / "CES_logs.txt"
    with open(ces_logs_path, 'w') as f:
        f.write(ces_logs)
    print(f"  ✓ Saved: {ces_logs_path}")

    # Generate global summary
    print("  Generating global summary...")
    summary_prompt = global_summary_prompt(captions_data)
    global_summary = await query_llm_async(summary_prompt, model_choice=llm_model, temperature=0.3)

    global_summary_path = captions_dir / "global_summary.txt"
    with open(global_summary_path, 'w') as f:
        f.write(global_summary)
    print(f"  ✓ Saved: {global_summary_path}")

    return True


async def generate_summaries_for_all_videos(parent_dir, llm_model="kimi-k2.5"):
    """
    Generate summaries for all videos in a parent directory.

    Args:
        parent_dir: Directory containing video subdirectories
        llm_model: LLM model to use
    """
    parent_dir = Path(parent_dir)

    if not parent_dir.exists():
        print(f"❌ Directory not found: {parent_dir}")
        return

    # Find all video directories (those with captions/clip_captions.json)
    video_dirs = []
    for item in parent_dir.iterdir():
        if item.is_dir():
            clip_captions = item / "captions" / "clip_captions.json"
            if clip_captions.exists():
                video_dirs.append(item)

    if not video_dirs:
        print(f"⚠️  No videos with clip captions found in {parent_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Generating summaries for {len(video_dirs)} videos")
    print(f"{'='*80}\n")

    success_count = 0
    for video_dir in video_dirs:
        try:
            success = await generate_summaries_for_video(video_dir, llm_model)
            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing {video_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"✅ Successfully processed {success_count}/{len(video_dirs)} videos")
    print(f"{'='*80}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CES_logs.txt and global_summary.txt from clip captions"
    )
    parser.add_argument(
        "video_path",
        help="Path to video directory or parent directory containing video folders"
    )
    parser.add_argument(
        "--llm-model",
        default="kimi-k2.5",
        help="LLM model to use (default: kimi-k2.5)"
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)

    # Check if this is a single video directory or parent directory
    clip_captions = video_path / "captions" / "clip_captions.json"
    if clip_captions.exists():
        # Single video directory
        await generate_summaries_for_video(video_path, args.llm_model)
    else:
        # Parent directory with multiple videos
        await generate_summaries_for_all_videos(video_path, args.llm_model)


if __name__ == "__main__":
    asyncio.run(main())
