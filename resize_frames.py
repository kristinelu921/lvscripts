#!/usr/bin/env python3
"""
Resize all video frames to have shortest side = 224 pixels
Maintains aspect ratio with parallel processing
"""

import os
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def resize_frame(image_path, target_short_side=224, overwrite=True):
    """
    Resize image so shortest side is target_short_side, maintaining aspect ratio

    Args:
        image_path: Path to image file
        target_short_side: Target size for shortest dimension (default: 224)
        overwrite: If True, overwrite original. If False, save to *_resized.jpg

    Returns:
        (success: bool, path: str, message: str)
    """
    try:
        image_path = Path(image_path)
        with Image.open(image_path) as img:
            # Get current dimensions
            width, height = img.size

            # Skip if already at target size
            if min(width, height) == target_short_side:
                return (True, str(image_path), "Already correct size")

            # Calculate scaling factor based on shortest side
            if width < height:
                # Width is shorter
                scale = target_short_side / width
                new_width = target_short_side
                new_height = int(height * scale)
            else:
                # Height is shorter (or square)
                scale = target_short_side / height
                new_height = target_short_side
                new_width = int(width * scale)

            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save
            if overwrite:
                resized_img.save(image_path, quality=95)
            else:
                # Save with _resized suffix
                new_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"
                resized_img.save(new_path, quality=95)

            return (True, str(image_path), f"Resized to {new_width}x{new_height}")
    except Exception as e:
        return (False, str(image_path), f"Error: {e}")

def process_video_directory(video_dir, target_short_side=224, overwrite=True,
                           frame_pattern="frame_*.jpg", num_workers=None):
    """
    Process all frames in a video directory with parallel processing

    Args:
        video_dir: Path to video directory
        target_short_side: Target size for shortest dimension
        overwrite: Whether to overwrite original files
        frame_pattern: Glob pattern to match frame files
        num_workers: Number of parallel workers (default: cpu_count())
    """
    video_path = Path(video_dir)

    # Check if frames are in a 'frames' subdirectory
    frames_subdir = video_path / "frames"
    if frames_subdir.exists() and frames_subdir.is_dir():
        search_path = frames_subdir
    else:
        search_path = video_path

    # Find all frame files
    frame_files = sorted(search_path.glob(frame_pattern))

    if not frame_files:
        print(f"No frames found in {video_dir} matching pattern {frame_pattern}")
        return 0

    print(f"Processing {len(frame_files)} frames in {video_dir}")

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} workers")

    # Create partial function with fixed parameters
    resize_func = partial(resize_frame, target_short_side=target_short_side, overwrite=overwrite)

    # Process in parallel
    success_count = 0
    with Pool(num_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(resize_func, frame_files),
            total=len(frame_files),
            desc=f"Resizing {video_path.name}"
        ))

        # Count successes
        success_count = sum(1 for success, _, _ in results if success)

        # Print any errors
        errors = [(path, msg) for success, path, msg in results if not success]
        if errors:
            print(f"\nErrors encountered:")
            for path, msg in errors[:10]:  # Show first 10 errors
                print(f"  {path}: {msg}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

    print(f"Successfully resized {success_count}/{len(frame_files)} frames")
    return success_count

def process_all_videos(root_dir, target_short_side=224, overwrite=True,
                      frame_pattern="frame_*.jpg", num_workers=None):
    """
    Process all video directories under root_dir

    Args:
        root_dir: Root directory containing video subdirectories
        target_short_side: Target size for shortest dimension
        overwrite: Whether to overwrite original files
        frame_pattern: Glob pattern to match frame files
        num_workers: Number of parallel workers (default: cpu_count())
    """
    root_path = Path(root_dir)

    # Find all subdirectories (video folders)
    video_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    print(f"Found {len(video_dirs)} video directories")

    total_frames = 0
    for video_dir in video_dirs:
        frames_processed = process_video_directory(
            video_dir,
            target_short_side,
            overwrite,
            frame_pattern,
            num_workers
        )
        total_frames += frames_processed
        print()

    print(f"Total frames processed: {total_frames}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize video frames to have shortest side = 224px (with parallel processing)"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing video folders (each with frame images)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target size for shortest side (default: 224)"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite originals, create new files with _resized suffix"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.jpg",
        help="Glob pattern to match frame files (default: frame_*.jpg)"
    )
    parser.add_argument(
        "--single-video",
        action="store_true",
        help="Process directory as single video (not a parent directory)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {cpu_count()})"
    )

    args = parser.parse_args()

    if args.single_video:
        # Process single video directory
        process_video_directory(
            args.directory,
            args.target_size,
            not args.no_overwrite,
            args.pattern,
            args.workers
        )
    else:
        # Process all video directories under root
        process_all_videos(
            args.directory,
            args.target_size,
            not args.no_overwrite,
            args.pattern,
            args.workers
        )
