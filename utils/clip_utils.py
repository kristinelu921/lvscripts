#!/usr/bin/env python3
"""
Utility functions for extracting and working with video clips.
"""

import os
import subprocess
from pathlib import Path
import tempfile


def extract_clip_from_video(video_path, start_frame, end_frame, output_path=None, fps=1):
    """
    Extract a clip from a video between start_frame and end_frame.

    Args:
        video_path: Path to source video file
        start_frame: Starting frame number (frame numbers are timestamps in seconds for 1fps videos)
        end_frame: Ending frame number
        output_path: Optional output path for clip. If None, uses temp file.
        fps: Frame rate of the source video (default: 1)

    Returns:
        Path to extracted clip file
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Calculate start and duration in seconds
    start_time = start_frame / fps if isinstance(start_frame, int) else start_frame
    end_time = end_frame / fps if isinstance(end_frame, int) else end_frame
    duration = end_time - start_time

    if duration <= 0:
        raise ValueError(f"Invalid clip duration: start={start_time}s, end={end_time}s")

    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"clip_{start_frame}_{end_frame}.mp4")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract clip
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',  # No audio
        str(output_path),
        '-y'  # Overwrite
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Extracted clip: {start_time}s to {end_time}s -> {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"Failed to extract clip: {error_msg}")


async def query_clip(video_path, start_frame, end_frame, fps=1):
    """
    Extract a clip from a video for querying/analysis.

    Args:
        video_path: Path to source video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Frame rate of source video (default: 1)

    Returns:
        Path to extracted clip file
    """
    try:
        clip_path = extract_clip_from_video(
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps
        )
        return {
            'status': 'success',
            'clip_path': clip_path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration': (end_frame - start_frame) / fps
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'start_frame': start_frame,
            'end_frame': end_frame
        }
