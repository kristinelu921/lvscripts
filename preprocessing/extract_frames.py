#!/usr/bin/env python3
"""
Extract frames from videos at 1fps and organize by video ID.
Optionally extract video clips based on scene changes.

Usage:
    python extract_frames.py <video_source_dir> <output_base_dir>

Example:
    python extract_frames.py /mnt/ssh/data/longvideobench/videos /mnt/ssh/data/processed_videos
    python extract_frames.py /mnt/ssh/data/longvideobench/videos /mnt/ssh/data/processed_videos --clip
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import json
from multiprocessing import Pool, cpu_count
from functools import partial


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


def detect_scenes(video_path, threshold=0.3):
    """
    Detect scene changes in video using ffmpeg scene detection.

    Args:
        video_path: Path to video file
        threshold: Scene change threshold (0.0-1.0, default 0.3)

    Returns:
        List of scene change timestamps in seconds
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-filter:v', f'select=\'gt(scene,{threshold})\',showinfo',
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Parse timestamps from showinfo output (in stderr which is redirected to stdout)
        timestamps = []
        output = result.stdout if result.stdout else ""
        for line in output.split('\n'):
            if 'pts_time:' in line:
                try:
                    # Extract timestamp from line like "pts_time:45.123"
                    pts_time = line.split('pts_time:')[1].split()[0]
                    timestamps.append(float(pts_time))
                except (IndexError, ValueError):
                    continue

        # Add start timestamp if not present
        if not timestamps or timestamps[0] > 0:
            timestamps.insert(0, 0.0)

        return timestamps
    except Exception as e:
        print(f"  Warning: Scene detection failed: {e}")
        return [0.0]


def extract_clips(video_path, output_dir, scene_timestamps, max_clip_duration=120, fps=5, overwrite=False):
    """
    Extract video clips based on scene timestamps.

    Args:
        video_path: Path to video file
        output_dir: Directory to save clips
        scene_timestamps: List of scene change timestamps
        max_clip_duration: Maximum clip duration in seconds (default 120 = 2 minutes)
        fps: Frame rate for clips (default 5)
        overwrite: Whether to overwrite existing clips

    Returns:
        Dictionary with extraction results
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if clips already exist
    existing_clips = list(output_dir.glob('clip_*.mp4'))
    if existing_clips and not overwrite:
        print(f"  Clips already exist ({len(existing_clips)} clips), skipping...")
        return {
            'status': 'skipped',
            'clips_extracted': len(existing_clips),
            'message': 'Clips already exist'
        }

    # Get video duration
    duration = get_video_duration(video_path)
    if not duration:
        return {
            'status': 'error',
            'clips_extracted': 0,
            'error': 'Could not get video duration'
        }

    # Create clip segments (ensure no clip exceeds max_clip_duration)
    clips_info = []
    for i, start_time in enumerate(scene_timestamps):
        # Determine end time
        if i + 1 < len(scene_timestamps):
            end_time = scene_timestamps[i + 1]
        else:
            end_time = duration

        # Split long segments into max_clip_duration chunks
        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + max_clip_duration, end_time)
            clips_info.append({
                'index': len(clips_info),
                'start': current_start,
                'end': current_end,
                'duration': current_end - current_start
            })
            current_start = current_end

    print(f"  Extracting {len(clips_info)} clips from {len(scene_timestamps)} scenes...")

    # Extract each clip
    extracted_clips = []
    for clip_info in clips_info:
        clip_filename = f"clip_{clip_info['index']:04d}.mp4"
        clip_path = output_dir / clip_filename

        # FFmpeg command to extract clip with scaling and frame rate
        # Scale to 256 on shorter side: scale=256:-1 for portrait, scale=-1:256 for landscape
        cmd = [
            'ffmpeg',
            '-y' if overwrite else '-n',
            '-ss', str(clip_info['start']),
            '-i', str(video_path),
            '-t', str(clip_info['duration']),
            '-vf', f'fps={fps},scale=256:-1',  # Simplified: scale width to 256, height proportional
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-an',  # No audio
            str(clip_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            extracted_clips.append({
                'filename': clip_filename,
                'start': clip_info['start'],
                'end': clip_info['end'],
                'duration': clip_info['duration']
            })
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to extract clip {clip_filename}: {e}")
            continue

    print(f"  ✓ Extracted {len(extracted_clips)} clips")

    # Save clips metadata
    metadata_file = output_dir / 'clips_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(extracted_clips, f, indent=2)

    return {
        'status': 'success',
        'clips_extracted': len(extracted_clips),
        'clips_info': extracted_clips,
        'output_dir': str(output_dir)
    }


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
        '-threads', '0',  # Use all available CPU threads
        '-i', str(video_path),
        '-vf', f"fps={fps},scale='if(gte(iw,ih),-1,224)':'if(gte(iw,ih),224,-1)'",
        '-q:v', '2',  # High quality JPEG
        '-threads', '0',  # Thread count for encoder
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


def process_single_video(args):
    """
    Process a single video (used for parallel processing).

    Args:
        args: Tuple of (video_path, output_base_dir, fps, overwrite, extract_clips_flag, index, total)

    Returns:
        Dictionary with video processing result
    """
    video_path, output_base_dir, fps, overwrite, extract_clips_flag, index, total = args
    video_path = Path(video_path)
    output_base_dir = Path(output_base_dir)

    # Skip empty files
    if video_path.stat().st_size == 0:
        print(f"[{index}/{total}] {video_path.name} - EMPTY FILE, skipping")
        return {
            'video_id': get_video_id_from_filename(video_path),
            'video_path': str(video_path),
            'status': 'skipped',
            'frames_extracted': 0,
            'message': 'Empty file'
        }

    print(f"\n[{index}/{total}] Processing: {video_path.name}")

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

    # Build result
    video_result = {
        'video_id': video_id,
        'video_path': str(video_path),
        'frames_dir': str(frames_dir),
        'status': extraction_result['status'],
        'frames_extracted': extraction_result['frames_extracted']
    }

    if extraction_result['status'] == 'error':
        video_result['error'] = extraction_result.get('error', 'Unknown error')

    # Extract clips if requested
    if extract_clips_flag:
        clips_dir = video_dir / 'clips'
        print(f"  Detecting scenes for clip extraction...")

        # Detect scenes
        scene_timestamps = detect_scenes(video_path)
        print(f"  Found {len(scene_timestamps)} scene changes")

        # Extract clips
        clips_result = extract_clips(
            video_path=video_path,
            output_dir=clips_dir,
            scene_timestamps=scene_timestamps,
            max_clip_duration=120,
            fps=5,
            overwrite=overwrite
        )

        video_result['clips_dir'] = str(clips_dir)
        video_result['clips_extracted'] = clips_result.get('clips_extracted', 0)
        video_result['clips_status'] = clips_result.get('status', 'unknown')

        if clips_result['status'] == 'error':
            video_result['clips_error'] = clips_result.get('error', 'Unknown error')

    return video_result


def process_videos(video_source_dir, output_base_dir, fps=1, overwrite=False, video_ext=None, parallel_jobs=1, extract_clips=False):
    """
    Process all videos in source directory.

    Args:
        video_source_dir: Directory containing video files
        output_base_dir: Base directory for output (will create subdirs per video)
        fps: Frames per second to extract
        overwrite: Whether to overwrite existing frames
        video_ext: List of video extensions to process (default: common formats)
        parallel_jobs: Number of parallel jobs (1 = sequential, 0 = auto)
        extract_clips: Whether to extract clips based on scene detection

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

    # Determine number of parallel jobs
    if parallel_jobs == 0:
        parallel_jobs = cpu_count()

    print(f"\n{'='*70}")
    print(f"FRAME EXTRACTION")
    print(f"{'='*70}")
    print(f"Source directory: {video_source_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Videos found: {len(video_files)}")
    print(f"FPS: {fps}")
    print(f"Extract clips: {extract_clips}")
    print(f"Parallel jobs: {parallel_jobs}")
    print(f"{'='*70}\n")

    results = {
        'total_videos': len(video_files),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_frames': 0,
        'total_clips': 0,
        'videos': []
    }

    # Prepare arguments for parallel processing
    sorted_videos = sorted(video_files)
    video_args = [
        (video_path, output_base_dir, fps, overwrite, extract_clips, i, len(video_files))
        for i, video_path in enumerate(sorted_videos, 1)
    ]

    # Process videos (parallel or sequential)
    if parallel_jobs > 1:
        with Pool(processes=parallel_jobs) as pool:
            video_results = pool.map(process_single_video, video_args)
    else:
        video_results = [process_single_video(args) for args in video_args]

    # Aggregate results
    for video_result in video_results:
        if video_result['status'] == 'success':
            results['processed'] += 1
            results['total_frames'] += video_result['frames_extracted']
        elif video_result['status'] == 'skipped':
            results['skipped'] += 1
            results['total_frames'] += video_result.get('frames_extracted', 0)
        else:
            results['errors'] += 1

        if 'clips_extracted' in video_result:
            results['total_clips'] += video_result['clips_extracted']

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
    if results.get('total_clips', 0) > 0:
        print(f"Total clips extracted: {results['total_clips']}")
    print(f"{'='*70}\n")

    # Show errors if any
    if results['errors'] > 0:
        print("Videos with errors:")
        for video in results['videos']:
            if video['status'] == 'error':
                print(f"  - {video['video_id']}: {video.get('error', 'Unknown error')[:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos at 1fps and optionally extract clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from all videos in directory
  python extract_frames.py /path/to/videos /path/to/output

  # Extract frames and clips with scene detection
  python extract_frames.py /path/to/videos /path/to/output --clip

  # Process videos in parallel (4 jobs)
  python extract_frames.py /path/to/videos /path/to/output -j 4

  # Auto-detect CPU count and use all cores
  python extract_frames.py /path/to/videos /path/to/output -j 0

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
    parser.add_argument('--clip', action='store_true',
                       help='Extract video clips based on scene detection (5fps, max 2min, 256px shorter side)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing frames and clips')
    parser.add_argument('--ext', nargs='+',
                       help='Video file extensions to process (default: .mp4 .avi .mov .mkv .flv .wmv .webm)')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                       help='Number of parallel jobs (0 = auto-detect CPU count, default: 1)')
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
        video_ext=args.ext,
        parallel_jobs=args.jobs,
        extract_clips=args.clip
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
