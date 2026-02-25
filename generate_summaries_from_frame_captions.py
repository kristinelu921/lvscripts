#!/usr/bin/env python3
"""Compatibility wrapper for frame-style summaries.

Historically this script generated summaries from ``clip_frame_captions.json``.
It now delegates to :mod:`generate_kimi_summaries` with ``--caption-style frame``.
"""

import argparse
import asyncio

from generate_kimi_summaries import process_dataset as generate_summaries_dataset


async def _run(dataset, base_dir, model):
    await generate_summaries_dataset(dataset, base_dir, model=model, caption_style="frame")


def main():
    parser = argparse.ArgumentParser(description="Generate CES_logs.txt and global_summary.txt from frame captions")
    parser.add_argument('--dataset', choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                       default='all', help='Dataset to process')
    parser.add_argument('--base-dir', default='/mnt/ssd/data',
                       help='Base directory (default: /mnt/ssd/data)')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                       help='LLM model for summaries')

    args = parser.parse_args()

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]
    async def runner():
        for dataset in datasets:
            await _run(dataset, args.base_dir, args.model)

    asyncio.run(runner())


if __name__ == '__main__':
    main()
