#!/usr/bin/env python3
"""
Quick test script for VideoMME integration.

This script runs a minimal test on 1 video with 2 questions to verify
that the VideoMME integration is working correctly.

Usage:
    python test_videomme_quick.py
"""

import asyncio
import sys
from run_videomme import run_videomme_evaluation


async def quick_test():
    """Run a quick test with 1 video and 2 questions."""
    print("="*80)
    print("VIDEOMME QUICK TEST")
    print("="*80)
    print()
    print("This will:")
    print("1. Load the VideoMME validation dataset")
    print("2. Process 1 video with 2 questions")
    print("3. Test the complete pipeline (download, caption, embed, answer)")
    print()
    print("Expected time: ~20-30 minutes")
    print("="*80)
    print()

    try:
        await run_videomme_evaluation(
            dataset_subset="validation",
            output_dir="./videomme_test_quick",
            llm_model="deepseek-ai/DeepSeek-V3.1",
            vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            max_videos=1,
            max_questions_per_video=2,
            use_critic=True,
            skip_existing=True
        )

        print()
        print("="*80)
        print("QUICK TEST COMPLETE!")
        print("="*80)
        print()
        print("Results saved to: ./videomme_test_quick/")
        print()
        print("To run a full evaluation:")
        print("  python run_videomme.py --subset validation --output_dir ./full_results")
        print()

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(quick_test())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
