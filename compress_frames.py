#!/usr/bin/env python3
"""
Compress all frames in videos_processed directories to have shorter side = 224px
Uses PIL for efficient batch processing with multiprocessing
"""
import os
import sys
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse


def resize_frame(args):
    """Resize a single frame to have shorter side = 224px

    Args:
        args: tuple of (frame_path, target_short_side, quality)

    Returns:
        tuple: (success, frame_path, old_size, new_size) or (False, frame_path, error)
    """
    frame_path, target_short_side, quality = args

    try:
        # Open image
        img = Image.open(frame_path)
        original_size = img.size

        # Check if already correct size
        short_side = min(img.size)
        if short_side == target_short_side:
            return (True, frame_path, original_size, original_size, "already_correct")

        # Calculate new dimensions maintaining aspect ratio
        width, height = img.size
        if width < height:
            # Width is shorter side
            new_width = target_short_side
            new_height = int(height * (target_short_side / width))
        else:
            # Height is shorter side
            new_height = target_short_side
            new_width = int(width * (target_short_side / height))

        # Resize image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Save with specified quality
        img_resized.save(frame_path, "JPEG", quality=quality, optimize=True)

        return (True, frame_path, original_size, (new_width, new_height), "resized")

    except Exception as e:
        return (False, frame_path, str(e))


def get_all_frames(base_dirs, check_size=True):
    """Get all frame paths that need resizing

    Args:
        base_dirs: List of directories to process
        check_size: If True, only return frames that aren't already 224px on short side

    Returns:
        List of frame paths
    """
    frames_to_process = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: {base_dir} does not exist, skipping")
            continue

        print(f"Scanning {base_dir}...")

        # Find all frames
        for frames_dir in base_path.glob("*/frames"):
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))

            if check_size and frame_files:
                # Check first frame to see if directory needs processing
                try:
                    img = Image.open(frame_files[0])
                    short_side = min(img.size)
                    if short_side == 224:
                        # Already correct size, skip entire directory
                        continue
                except Exception as e:
                    print(f"Warning: Could not check {frame_files[0]}: {e}")

            frames_to_process.extend([str(f) for f in frame_files])

    return frames_to_process


def main():
    parser = argparse.ArgumentParser(
        description="Compress frames to have shorter side = 224px"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "/mnt/ssd/data/longvideobench/videos_processed_1",
            "/mnt/ssd/data/longvideobench/videos_processed_2",
            "/mnt/ssd/data/longvideobench/videos_processed_3"
        ],
        help="Directories to process (default: all videos_processed_* dirs)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target size for shorter side (default: 224)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality (1-95, default: 85)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Don't check if frames are already correct size (process all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually resize, just show what would be done"
    )

    args = parser.parse_args()

    workers = args.workers if args.workers else cpu_count()

    print("="*70)
    print("FRAME COMPRESSION TOOL")
    print("="*70)
    print(f"Target shorter side: {args.target_size}px")
    print(f"JPEG quality: {args.quality}")
    print(f"Worker processes: {workers}")
    print(f"Dry run: {args.dry_run}")
    print("="*70)

    # Get all frames to process
    print("\nScanning directories for frames...")
    frames = get_all_frames(args.dirs, check_size=not args.no_check)

    if not frames:
        print("\n✓ No frames need resizing! All frames are already at target size.")
        return 0

    print(f"\nFound {len(frames):,} frames to process")

    if args.dry_run:
        print("\nDRY RUN - Would process:")
        # Sample first 5 frames
        for frame in frames[:5]:
            print(f"  {frame}")
        if len(frames) > 5:
            print(f"  ... and {len(frames)-5:,} more")
        return 0

    # Prepare arguments for multiprocessing
    process_args = [(frame, args.target_size, args.quality) for frame in frames]

    # Process frames with progress bar
    print("\nProcessing frames...")
    success_count = 0
    error_count = 0
    already_correct = 0
    resized_count = 0

    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(resize_frame, process_args),
            total=len(frames),
            desc="Compressing",
            unit="frame"
        ):
            if result[0]:  # success
                success_count += 1
                if result[4] == "already_correct":
                    already_correct += 1
                else:
                    resized_count += 1
            else:
                error_count += 1
                print(f"\n✗ Error processing {result[1]}: {result[2]}")

    # Summary
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Total frames: {len(frames):,}")
    print(f"Successfully processed: {success_count:,}")
    print(f"  - Already correct size: {already_correct:,}")
    print(f"  - Resized: {resized_count:,}")
    print(f"Errors: {error_count:,}")
    print("="*70)

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
