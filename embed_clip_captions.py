#!/usr/bin/env python3
"""Compatibility wrapper for clip caption embeddings."""

import argparse
import asyncio
from pathlib import Path

from preprocessing import embed_frame_captions


def main():
    parser = argparse.ArgumentParser(description='Embed clip captions using Together API')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data', help='Base directory')
    parser.add_argument('--model', type=str, default=None, help='Embedding model')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing multiple videos')
    args = parser.parse_args()

    print("=" * 60)
    print("Clip Caption Embedding Generation")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model or 'Alibaba-NLP/gte-modernbert-base'}")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]
    for dataset in datasets:
        video_dir = Path(args.base_dir) / 'kimi' / dataset / 'video_files'
        asyncio.run(
            embed_frame_captions.embed_many(
                str(video_dir),
                batch_size=args.batch_size,
                caption_style='clip',
                provider='together',
                model=args.model,
            )
        )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Run generate_kimi_summaries.py")
    print("  2. Run pipeline with clip captions")
    print("=" * 60)


if __name__ == '__main__':
    main()
