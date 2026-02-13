#!/usr/bin/env python3
"""
Fine-grained frame extraction tool - Extract frames at higher FPS (up to 10 FPS)
for specific moments that require detailed analysis (hand motions, fast actions, etc.)

This tool extracts additional frames on-demand for the LLM to analyze fine details.
Uses ffmpeg for fast, efficient frame extraction.
"""
import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def extract_frames_at_fps(
    video_path: str,
    start_second: float,
    end_second: float,
    fps: int,
    output_dir: str,
    prefix: str = "detailed"
) -> List[str]:
    """Extract frames from video at specified FPS between start and end seconds using ffmpeg

    Args:
        video_path: Path to video file
        start_second: Start time in seconds
        end_second: End time in seconds
        fps: Frames per second to extract (1-10)
        output_dir: Directory to save extracted frames
        prefix: Prefix for frame filenames (default: "detailed")

    Returns:
        List of paths to extracted frames

    Example:
        >>> # Extract frames from 45.0s to 47.0s at 5 FPS
        >>> frames = extract_frames_at_fps(
        ...     "/path/to/video.mp4",
        ...     45.0, 47.0, 5,
        ...     "/path/to/output"
        ... )
        >>> print(frames)
        ['/path/to/output/detailed_frame_0045.00.jpg',
         '/path/to/output/detailed_frame_0045.20.jpg',
         '/path/to/output/detailed_frame_0045.40.jpg',
         ...]
    """
    if fps < 1 or fps > 10:
        raise ValueError("FPS must be between 1 and 10")

    if start_second >= end_second:
        raise ValueError("start_second must be less than end_second")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Temp output pattern for ffmpeg (uses frame counter)
    temp_pattern = os.path.join(output_dir, f"{prefix}_temp_%04d.jpg")

    # Build ffmpeg command
    # -ss: start time, -to: end time
    # -vf fps={fps}: extract at specified FPS
    # -q:v 2: high quality JPEG (scale 2-31, lower is better)
    # -start_number 0: start numbering from 0
    cmd = [
        'ffmpeg',
        '-ss', str(start_second),
        '-to', str(end_second),
        '-i', video_path,
        '-vf', f"fps={fps},scale='if(gte(iw,ih),-1,224)':'if(gte(iw,ih),224,-1)'",
        '-q:v', '2',
        '-start_number', '0',
        '-y',  # overwrite output files
        temp_pattern
    ]

    try:
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise on non-zero exit, check manually
        )

        # Check if ffmpeg failed
        if result.returncode != 0:
            # Common ffmpeg errors
            stderr = result.stderr.lower()
            if "no such file" in stderr or "does not exist" in stderr:
                raise RuntimeError(f"Video file not found or inaccessible: {video_path}")
            elif "invalid" in stderr or "could not find" in stderr:
                raise RuntimeError(f"Invalid video format or codec issue: {stderr[:200]}")
            else:
                raise RuntimeError(f"ffmpeg failed (exit code {result.returncode}): {result.stderr[:500]}")

    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg")

    # Find all extracted temp frames
    temp_frames = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith(f"{prefix}_temp_") and f.endswith('.jpg')
    ])

    if not temp_frames:
        # Get video duration to provide better error message
        duration_msg = ""
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                video_duration = float(probe_result.stdout.strip())
                duration_msg = f" (video duration: {video_duration:.1f}s)"
        except:
            pass

        raise RuntimeError(
            f"No frames were extracted from {start_second}s to {end_second}s{duration_msg}. "
            f"Possible reasons: (1) Time range exceeds video duration, "
            f"(2) Video has no frames in this range, (3) Invalid time format. "
            f"ffmpeg stderr: {result.stderr[:300] if result.stderr else 'none'}"
        )

    # Rename frames to include timestamps
    extracted_frames = []
    frame_interval = 1.0 / fps

    for i, temp_filename in enumerate(temp_frames):
        # Calculate timestamp for this frame
        timestamp = start_second + (i * frame_interval)

        # Create final filename with timestamp
        final_filename = f"{prefix}_frame_{timestamp:07.2f}.jpg"

        temp_path = os.path.join(output_dir, temp_filename)
        final_path = os.path.join(output_dir, final_filename)

        # Rename temp file to final name
        os.rename(temp_path, final_path)
        extracted_frames.append(final_path)

        print(f"Extracted frame at {timestamp:.2f}s -> {final_filename}")

    return extracted_frames


def extract_frame_window(
    video_path: str,
    center_second: float,
    window_seconds: float,
    fps: int,
    output_dir: str
) -> List[str]:
    """Extract frames in a window around a center time

    Args:
        video_path: Path to video file
        center_second: Center time in seconds
        window_seconds: Window size (±window_seconds around center)
        fps: Frames per second to extract
        output_dir: Directory to save frames

    Returns:
        List of extracted frame paths
    """
    start = max(0, center_second - window_seconds)
    end = center_second + window_seconds

    return extract_frames_at_fps(
        video_path, start, end, fps, output_dir,
        prefix=f"window_{center_second:.0f}s"
    )


def extract_frames_for_list(
    video_path: str,
    time_points: List[float],
    fps: int,
    output_dir: str,
    window: float = 0.5
) -> dict:
    """Extract frames around multiple time points

    Args:
        video_path: Path to video file
        time_points: List of time points in seconds
        fps: Frames per second to extract
        output_dir: Directory to save frames
        window: Window size around each time point (default: ±0.5s)

    Returns:
        Dictionary mapping time_point -> list of frame paths
    """
    results = {}

    for time_point in time_points:
        start = max(0, time_point - window)
        end = time_point + window

        frames = extract_frames_at_fps(
            video_path, start, end, fps, output_dir,
            prefix=f"detail_{time_point:.0f}s"
        )

        results[time_point] = frames

    return results


def get_video_path_from_id(video_id: str, videos_dir: str = "/mnt/ssd/data/longvideobench/videos") -> Optional[str]:
    """Get video file path from video ID

    Args:
        video_id: Video identifier (e.g., "Y0IaijKNGX8")
        videos_dir: Directory containing video files (can be videos_processed_X, will auto-convert to videos/)

    Returns:
        Path to video file, or None if not found
    """
    # If videos_dir points to videos_processed_X, convert to videos/
    # e.g., /mnt/ssd/data/lvbench/videos_processed_1 -> /mnt/ssd/data/lvbench/videos
    if 'videos_processed' in videos_dir:
        videos_dir = videos_dir.rsplit('/', 1)[0] + '/videos'

    videos_path = Path(videos_dir)

    # Try exact match first
    video_file = videos_path / f"{video_id}.mp4"
    if video_file.exists():
        return str(video_file)

    # Try alternative extensions
    for ext in ['.mkv', '.webm', '.avi']:
        alt_file = videos_path / f"{video_id}{ext}"
        if alt_file.exists():
            return str(alt_file)

    # Try to find files with format codes (e.g., T1yhBv1ytzw.f399.mp4)
    # This handles YouTube-dl downloaded videos with format codes
    if videos_path.exists():
        for video_file in videos_path.glob(f"{video_id}.*"):
            if video_file.suffix in ['.mp4', '.mkv', '.webm', '.avi']:
                return str(video_file)

    return None


def extract_fine_grained_for_pipeline(
    video_id: str,
    start_second: float,
    end_second: float,
    fps: int = 5,
    videos_dir: str = "/mnt/ssd/data/longvideobench/videos",
    output_base: str = "/mnt/ssd/data/longvideobench",
    vid_path: str = None
) -> List[str]:
    """Extract fine-grained frames for use in pipeline

    Args:
        video_id: Video identifier
        start_second: Start time
        end_second: End time
        fps: Frames per second (default: 5)
        videos_dir: Directory containing videos
        output_base: Base directory for output

    Returns:
        List of extracted frame paths (relative to video folder for pipeline use)

    Example for LLM usage:
        >>> # When LLM needs detailed frames from 45s to 47s at 5 FPS
        >>> frames = extract_fine_grained_for_pipeline("Y0IaijKNGX8", 45.0, 47.0, fps=5)
        >>> # Returns: ['frames/detailed_frame_0045.0.jpg', 'frames/detailed_frame_0045.2.jpg', ...]
        >>> # LLM can then use these with VLM_QUERY tool
    """
    # Get video path
    video_path = get_video_path_from_id(video_id, videos_dir)
    if not video_path:
        raise FileNotFoundError(f"Video not found for ID: {video_id}")

    # Determine output directory
    if vid_path:
        # Use the provided vid_path directly
        output_dir = Path(vid_path) / "frames"
    else:
        # Fallback to old behavior
        output_dir = Path(output_base) / f"videos_processed_1/{video_id}/frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames = extract_frames_at_fps(
        video_path,
        start_second,
        end_second,
        fps,
        str(output_dir),
        prefix="detailed"
    )

    # Convert to relative paths for pipeline
    relative_frames = []
    for frame_path in frames:
        frame_name = Path(frame_path).name
        relative_frames.append(f"frames/{frame_name}")

    return relative_frames


def main():
    """CLI for fine-grained frame extraction"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract fine-grained frames at higher FPS for detailed analysis"
    )

    parser.add_argument("video_path", help="Path to video file or video ID")
    parser.add_argument("start", type=float, help="Start time in seconds")
    parser.add_argument("end", type=float, help="End time in seconds")
    parser.add_argument("--fps", type=int, default=5, choices=range(1, 11),
                       help="Frames per second to extract (1-10, default: 5)")
    parser.add_argument("--output", default="./fine_grained_frames",
                       help="Output directory")
    parser.add_argument("--prefix", default="detailed",
                       help="Prefix for frame filenames")

    args = parser.parse_args()

    # Check if video_path is a video ID or actual path
    if not os.path.exists(args.video_path) and '/' not in args.video_path:
        # Treat as video ID
        print(f"Looking for video ID: {args.video_path}")
        video_path = get_video_path_from_id(args.video_path)
        if not video_path:
            print(f"Error: Video not found for ID: {args.video_path}")
            return 1
    else:
        video_path = args.video_path

    print(f"Video: {video_path}")
    print(f"Extracting frames from {args.start}s to {args.end}s at {args.fps} FPS")
    print(f"Output directory: {args.output}\n")

    try:
        frames = extract_frames_at_fps(
            video_path,
            args.start,
            args.end,
            args.fps,
            args.output,
            args.prefix
        )

        print(f"\n✓ Successfully extracted {len(frames)} frames!")
        print(f"  Output: {args.output}")
        print(f"\nExample frames:")
        for frame in frames[:3]:
            print(f"  - {frame}")
        if len(frames) > 3:
            print(f"  ... and {len(frames) - 3} more")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
