#!/usr/bin/env python3
"""Compatibility wrapper for clip captioning in ``caption_frames_together.py``."""

import argparse
import asyncio
from pathlib import Path

from caption_frames_together import process_clip_dataset


def main():
    parser = argparse.ArgumentParser(description="Caption video clips using Kimi direct API")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "longvideobench", "lvbench", "videomme"],
                        help="Dataset to process")
    parser.add_argument("--base-dir", type=str, default="/mnt/ssd/data", help="Base directory")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Maximum concurrent API requests (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("Kimi Clip Captioning (Direct Moonshot API)")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("Model: kimi-k2.5")

    datasets = ["longvideobench", "lvbench", "videomme"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        asyncio.run(process_clip_dataset(dataset, args.base_dir, args.max_concurrent))

    print("\n" + "=" * 60)
    print("CAPTIONING COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Run embed_frame_captions.py --caption-style clip")
    print("  2. Run generate_kimi_summaries.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
