#!/usr/bin/env python3
"""Compatibility wrapper for clip-frame caption embeddings."""

import argparse
import asyncio
from pathlib import Path

from preprocessing import embed_frame_captions


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for clip_frame_captions.json')
    parser.add_argument('--dataset', choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                       default='all', help='Dataset to process')
    parser.add_argument('--base-dir', default='/mnt/ssd/data',
                       help='Base directory (default: /mnt/ssd/data)')
    parser.add_argument('--model', default=None,
                       help='Embedding model')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing clip-frame embeddings files')
    args = parser.parse_args()

    print("=" * 60)
    print("Clip Frame Caption Embedding Generation")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model or 'nvidia/NV-Embed-v2'}")
    print()

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        video_dir = Path(args.base_dir) / 'kimi' / dataset / 'video_files'
        asyncio.run(
            embed_frame_captions.embed_many(
                str(video_dir),
                batch_size=10,
                caption_style='clip-frame',
                provider='together',
                model=args.model,
                force=args.force,
            )
        )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Generate summaries using generate_summaries_from_frame_captions.py")
    print("  2. Run pipeline with --query-aware or equivalent frame-caption path")


if __name__ == '__main__':
    main()
