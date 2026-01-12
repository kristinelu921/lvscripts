#!/usr/bin/env python3
"""
Extract frames from videos at 1fps and organize by video ID.

Usage:
    python extract_frames.py <video_source_dir> <output_base_dir>

Example:
    python extract_frames.py /mnt/ssh/data/longvideobench/videos /mnt/ssh/data/processed_videos
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import json


def get_video_id_from_filename(video_path):
    """Extract video ID from filename (filename without extension)"""
    return Path(video_path).stem


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"  Warning: Could not get duration: {e}")
        return None


def extract_frames(video_path, output_dir, fps=1, overwrite=False):
    """
    Extract frames from video at specified fps using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (default: 1)
        overwrite: Whether to overwrite existing frames

    Returns:
        Dictionary with extraction results
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing_frames = list(output_dir.glob('frame_*.jpg'))
    if existing_frames and not overwrite:
        print(f"  Frames already exist ({len(existing_frames)} frames), skipping...")
        return {
            'status': 'skipped',
            'frames_extracted': len(existing_frames),
            'message': 'Frames already exist'
        }

    # Get video duration for progress estimation
    duration = get_video_duration(video_path)
    if duration:
        estimated_frames = int(duration * fps)
        print(f"  Video duration: {duration:.1f}s, estimated frames: {estimated_frames}")

    # FFmpeg command to extract frames at 1fps
    # -vf fps=1: Extract 1 frame per second
    # frame_%04d.jpg: Output format with 4-digit zero-padded numbers
    output_pattern = str(output_dir / 'frame_%04d.jpg')

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality JPEG
        output_pattern,
        '-y' if overwrite else '-n'  # Overwrite or skip existing
    ]

    try:
        print(f"  Extracting frames at {fps}fps...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Count extracted frames
        frames = list(output_dir.glob('frame_*.jpg'))
        num_frames = len(frames)

        print(f"  ✓ Extracted {num_frames} frames")

        return {
            'status': 'success',
            'frames_extracted': num_frames,
            'output_dir': str(output_dir)
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"  ✗ FFmpeg error: {error_msg[:200]}")
        return {
            'status': 'error',
            'frames_extracted': 0,
            'error': error_msg[:500]
        }
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            'status': 'error',
            'frames_extracted': 0,
            'error': str(e)
        }


def process_videos(video_source_dir, output_base_dir, fps=1, overwrite=False, video_ext=None):
    """
    Process all videos in source directory.

    Args:
        video_source_dir: Directory containing video files
        output_base_dir: Base directory for output (will create subdirs per video)
        fps: Frames per second to extract
        overwrite: Whether to overwrite existing frames
        video_ext: List of video extensions to process (default: common formats)

    Returns:
        Dictionary with processing results
    """
    if video_ext is None:
        video_ext = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    video_source_dir = Path(video_source_dir)
    output_base_dir = Path(output_base_dir)

    if not video_source_dir.exists():
        print(f"Error: Video source directory not found: {video_source_dir}")
        return None

    # Find all video files
    video_files = []
    for ext in video_ext:
        video_files.extend(video_source_dir.glob(f'*{ext}'))

    if not video_files:
        print(f"Error: No video files found in {video_source_dir}")
        print(f"Looking for extensions: {video_ext}")
        return None

    print(f"\n{'='*70}")
    print(f"FRAME EXTRACTION")
    print(f"{'='*70}")
    print(f"Source directory: {video_source_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Videos found: {len(video_files)}")
    print(f"FPS: {fps}")
    print(f"{'='*70}\n")

    results = {
        'total_videos': len(video_files),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_frames': 0,
        'videos': []
    }

    for i, video_path in enumerate(sorted(video_files), 1):
        # Skip empty files
        if video_path.stat().st_size == 0:
            print(f"[{i}/{len(video_files)}] {video_path.name} - EMPTY FILE, skipping")
            results['skipped'] += 1
            continue

        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Get video ID from filename
        video_id = get_video_id_from_filename(video_path)
        print(f"  Video ID: {video_id}")

        # Create video directory structure
        video_dir = output_base_dir / video_id
        frames_dir = video_dir / 'frames'

        # Extract frames
        extraction_result = extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            fps=fps,
            overwrite=overwrite
        )

        # Update results
        video_result = {
            'video_id': video_id,
            'video_path': str(video_path),
            'frames_dir': str(frames_dir),
            'status': extraction_result['status'],
            'frames_extracted': extraction_result['frames_extracted']
        }

        if extraction_result['status'] == 'success':
            results['processed'] += 1
            results['total_frames'] += extraction_result['frames_extracted']
        elif extraction_result['status'] == 'skipped':
            results['skipped'] += 1
            results['total_frames'] += extraction_result['frames_extracted']
        else:
            results['errors'] += 1
            video_result['error'] = extraction_result.get('error', 'Unknown error')

        results['videos'].append(video_result)

    return results


def save_results(results, output_dir):
    """Save extraction results to JSON"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path(output_dir) / f'frame_extraction_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")


def print_summary(results):
    """Print summary of extraction results"""
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos: {results['total_videos']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Skipped (already exists): {results['skipped']}")
    print(f"Errors: {results['errors']}")
    print(f"Total frames extracted: {results['total_frames']}")
    print(f"{'='*70}\n")

    # Show errors if any
    if results['errors'] > 0:
        print("Videos with errors:")
        for video in results['videos']:
            if video['status'] == 'error':
                print(f"  - {video['video_id']}: {video.get('error', 'Unknown error')[:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos at 1fps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from all videos in directory
  python extract_frames.py /path/to/videos /path/to/output

  # Overwrite existing frames
  python extract_frames.py /path/to/videos /path/to/output --overwrite

  # Extract at different fps
  python extract_frames.py /path/to/videos /path/to/output --fps 2

  # Process only specific video extensions
  python extract_frames.py /path/to/videos /path/to/output --ext .mp4 .avi
        """
    )

    parser.add_argument('video_source_dir',
                       help='Directory containing video files')
    parser.add_argument('output_base_dir',
                       help='Base directory for output (subdirs created per video)')
    parser.add_argument('--fps', type=int, default=1,
                       help='Frames per second to extract (default: 1)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing frames')
    parser.add_argument('--ext', nargs='+',
                       help='Video file extensions to process (default: .mp4 .avi .mov .mkv .flv .wmv .webm)')
    parser.add_argument('--check-ffmpeg', action='store_true',
                       help='Check if ffmpeg is installed and exit')

    args = parser.parse_args()

    # Check ffmpeg availability
    if args.check_ffmpeg or True:  # Always check
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            ffmpeg_version = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg found: {ffmpeg_version}")
            if args.check_ffmpeg:
                return 0
        except FileNotFoundError:
            print("✗ FFmpeg not found!")
            print("\nPlease install ffmpeg:")
            print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  macOS: brew install ffmpeg")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            return 1

    # Process videos
    results = process_videos(
        video_source_dir=args.video_source_dir,
        output_base_dir=args.output_base_dir,
        fps=args.fps,
        overwrite=args.overwrite,
        video_ext=args.ext
    )

    if results is None:
        return 1

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output_base_dir)

    return 0 if results['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
