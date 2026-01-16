#!/usr/bin/env python3
"""
Direct subtitle search - searches subtitle JSON files using text matching.
Returns exact frame timestamps and second conversions.

Simple, fast, no embeddings required.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm timestamp to seconds

    Args:
        timestamp: Time in format "HH:MM:SS.mmm" or "MM:SS.mmm"

    Returns:
        Float seconds
    """
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        hours = int(hours)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        return 0.0

    minutes = int(minutes)
    seconds = float(seconds)

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_frame(seconds: float, fps: float = 1.0) -> int:
    """Convert seconds to frame number

    Args:
        seconds: Time in seconds
        fps: Frames per second (default 1.0 for standard extraction)

    Returns:
        Frame number (integer)
    """
    return int(seconds * fps)


def format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def search_subtitle_text(
    video_id: str,
    search_query: str,
    subtitles_dir: str = "/mnt/ssd/data/longvideobench/subtitles",
    case_sensitive: bool = False,
    fps: float = 1.0
) -> List[Dict]:
    """Search subtitle text directly using simple text matching

    Args:
        video_id: Video identifier (e.g., "Y0IaijKNGX8")
        search_query: Text to search for (can be substring)
        subtitles_dir: Directory containing subtitle JSON files
        case_sensitive: Whether search should be case-sensitive
        fps: Frames per second for frame number calculation (default 1.0)

    Returns:
        List of matching subtitle entries with frame info, sorted by time
        Each entry contains:
            - line: The subtitle text
            - start: Start timestamp (HH:MM:SS.mmm)
            - end: End timestamp (HH:MM:SS.mmm)
            - start_seconds: Start time in seconds (float)
            - end_seconds: End time in seconds (float)
            - start_frame: Start frame number (int)
            - end_frame: End frame number (int)
            - time_formatted: Human-readable time (M:SS or H:MM:SS)

    Example:
        >>> results = search_subtitle_text("Y0IaijKNGX8", "Covenant")
        >>> print(results[0])
        {
            'line': 'in the year 2552 the Covenant launched',
            'start': '00:00:02.790',
            'end': '00:00:02.800',
            'start_seconds': 2.79,
            'end_seconds': 2.8,
            'start_frame': 2,
            'end_frame': 2,
            'time_formatted': '0:02'
        }
    """
    # Construct subtitle file path
    subtitle_file = Path(subtitles_dir) / f"{video_id}_en.json"

    if not subtitle_file.exists():
        raise FileNotFoundError(
            f"Subtitle file not found: {subtitle_file}\n"
            f"Available videos: {list(Path(subtitles_dir).glob('*_en.json'))[:5]}"
        )

    # Load subtitles
    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    # Prepare search query
    if not case_sensitive:
        search_query = search_query.lower()

    # Search through subtitles
    matches = []

    for entry in subtitles:
        if not isinstance(entry, dict) or 'line' not in entry:
            continue

        subtitle_text = entry['line']
        search_text = subtitle_text if case_sensitive else subtitle_text.lower()

        # Check if query matches (substring search)
        if search_query in search_text:
            start_sec = timestamp_to_seconds(entry['start'])
            end_sec = timestamp_to_seconds(entry['end'])

            match = {
                'line': subtitle_text,
                'start': entry['start'],
                'end': entry['end'],
                'start_seconds': start_sec,
                'end_seconds': end_sec,
                'start_frame': seconds_to_frame(start_sec, fps),
                'end_frame': seconds_to_frame(end_sec, fps),
                'time_formatted': format_seconds(start_sec)
            }
            matches.append(match)

    # Sort by start time
    matches.sort(key=lambda x: x['start_seconds'])

    return matches


def search_subtitle_regex(
    video_id: str,
    pattern: str,
    subtitles_dir: str = "/mnt/ssd/data/longvideobench/subtitles",
    fps: float = 1.0
) -> List[Dict]:
    """Search subtitle text using regex pattern

    Args:
        video_id: Video identifier
        pattern: Regex pattern to search for
        subtitles_dir: Directory containing subtitle JSON files
        fps: Frames per second for frame number calculation

    Returns:
        List of matching subtitle entries (same format as search_subtitle_text)
    """
    subtitle_file = Path(subtitles_dir) / f"{video_id}_en.json"

    if not subtitle_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_file}")

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    # Compile regex pattern
    regex = re.compile(pattern, re.IGNORECASE)

    matches = []

    for entry in subtitles:
        if not isinstance(entry, dict) or 'line' not in entry:
            continue

        subtitle_text = entry['line']

        # Check if pattern matches
        if regex.search(subtitle_text):
            start_sec = timestamp_to_seconds(entry['start'])
            end_sec = timestamp_to_seconds(entry['end'])

            match = {
                'line': subtitle_text,
                'start': entry['start'],
                'end': entry['end'],
                'start_seconds': start_sec,
                'end_seconds': end_sec,
                'start_frame': seconds_to_frame(start_sec, fps),
                'end_frame': seconds_to_frame(end_sec, fps),
                'time_formatted': format_seconds(start_sec)
            }
            matches.append(match)

    matches.sort(key=lambda x: x['start_seconds'])

    return matches


def get_subtitles_at_time(
    video_id: str,
    time_seconds: float,
    window_seconds: float = 5.0,
    subtitles_dir: str = "/mnt/ssd/data/longvideobench/subtitles"
) -> List[Dict]:
    """Get all subtitles within a time window

    Args:
        video_id: Video identifier
        time_seconds: Center time in seconds
        window_seconds: Window size (±window_seconds around center)
        subtitles_dir: Directory containing subtitle JSON files

    Returns:
        List of subtitle entries within the time window
    """
    subtitle_file = Path(subtitles_dir) / f"{video_id}_en.json"

    if not subtitle_file.exists():
        return []

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    start_window = time_seconds - window_seconds
    end_window = time_seconds + window_seconds

    matches = []

    for entry in subtitles:
        if not isinstance(entry, dict) or 'line' not in entry:
            continue

        start_sec = timestamp_to_seconds(entry['start'])

        if start_window <= start_sec <= end_window:
            end_sec = timestamp_to_seconds(entry['end'])

            match = {
                'line': entry['line'],
                'start': entry['start'],
                'end': entry['end'],
                'start_seconds': start_sec,
                'end_seconds': end_sec,
                'start_frame': seconds_to_frame(start_sec, 1.0),
                'end_frame': seconds_to_frame(end_sec, 1.0),
                'time_formatted': format_seconds(start_sec)
            }
            matches.append(match)

    matches.sort(key=lambda x: x['start_seconds'])

    return matches


def format_results_simple(results: List[Dict]) -> str:
    """Format search results as simple text

    Args:
        results: List of subtitle search results

    Returns:
        Formatted string
    """
    if not results:
        return "No matching subtitles found."

    lines = [f"Found {len(results)} matching subtitle(s):\n"]

    for i, res in enumerate(results, 1):
        lines.append(f"[{i}] Frame {res['start_frame']}-{res['end_frame']} ({res['time_formatted']})")
        lines.append(f"    Text: {res['line']}")
        lines.append(f"    Exact time: {res['start_seconds']:.2f}s - {res['end_seconds']:.2f}s")
        lines.append("")

    return "\n".join(lines)


def main():
    """Test the subtitle search function"""
    import argparse

    parser = argparse.ArgumentParser(description="Search subtitle text directly")
    parser.add_argument("video_id", help="Video ID (e.g., Y0IaijKNGX8)")
    parser.add_argument("query", help="Text to search for")
    parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive search")
    parser.add_argument("--regex", action="store_true", help="Use regex pattern matching")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for frame calculation")

    args = parser.parse_args()

    print(f"\nSearching subtitles for video: {args.video_id}")
    print(f"Query: {args.query}")
    if args.regex:
        print("Mode: Regex")
    else:
        print(f"Mode: Text search ({'case-sensitive' if args.case_sensitive else 'case-insensitive'})")
    print()

    try:
        if args.regex:
            results = search_subtitle_regex(args.video_id, args.query, fps=args.fps)
        else:
            results = search_subtitle_text(
                args.video_id, args.query,
                case_sensitive=args.case_sensitive,
                fps=args.fps
            )

        print(format_results_simple(results))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
