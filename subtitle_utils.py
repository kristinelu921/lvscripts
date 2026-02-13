#!/usr/bin/env python3
"""
Utility functions for working with LongVideoBench subtitles.

Provides:
- Time-based subtitle search
- Frame-based subtitle retrieval
- Text extraction for specific time ranges
- Subtitle statistics
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def parse_timestamp(ts: str) -> float:
    """Convert timestamp string (HH:MM:SS.mmm) to seconds"""
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp string (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class SubtitleLoader:
    """Load and search subtitles for a video"""

    def __init__(self, subtitle_file: str):
        """
        Args:
            subtitle_file: Path to subtitle JSON file
        """
        self.subtitle_file = Path(subtitle_file)
        self.subtitles = self._load_subtitles()
        self._parse_timestamps()

    def _load_subtitles(self) -> List[Dict]:
        """Load subtitles from file"""
        with open(self.subtitle_file, 'r') as f:
            return json.load(f)

    def _parse_timestamps(self):
        """Parse all timestamps to seconds for faster searching"""
        for sub in self.subtitles:
            sub['start_seconds'] = parse_timestamp(sub['start'])
            sub['end_seconds'] = parse_timestamp(sub['end'])

    def get_subtitles_in_range(
        self,
        start_time: float,
        end_time: float,
        overlap_threshold: float = 0.0
    ) -> List[Dict]:
        """Get all subtitles within a time range

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            overlap_threshold: Minimum overlap in seconds (default: 0 = any overlap)

        Returns:
            List of subtitle entries within the time range
        """
        result = []
        for sub in self.subtitles:
            # Calculate overlap
            overlap_start = max(start_time, sub['start_seconds'])
            overlap_end = min(end_time, sub['end_seconds'])
            overlap = max(0, overlap_end - overlap_start)

            if overlap >= overlap_threshold:
                result.append(sub.copy())

        return result

    def get_text_in_range(
        self,
        start_time: float,
        end_time: float,
        separator: str = ' '
    ) -> str:
        """Get concatenated subtitle text for a time range

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            separator: String to join subtitle lines (default: ' ')

        Returns:
            Concatenated subtitle text
        """
        subs = self.get_subtitles_in_range(start_time, end_time)
        return separator.join(sub['line'] for sub in subs)

    def get_subtitle_at_time(self, time: float) -> Optional[Dict]:
        """Get the subtitle active at a specific time

        Args:
            time: Time in seconds

        Returns:
            Subtitle entry or None if no subtitle at that time
        """
        for sub in self.subtitles:
            if sub['start_seconds'] <= time <= sub['end_seconds']:
                return sub.copy()
        return None

    def search_text(self, query: str, case_sensitive: bool = False) -> List[Dict]:
        """Search for subtitles containing specific text

        Args:
            query: Text to search for
            case_sensitive: Whether to do case-sensitive search (default: False)

        Returns:
            List of matching subtitle entries
        """
        if not case_sensitive:
            query = query.lower()

        results = []
        for sub in self.subtitles:
            text = sub['line'] if case_sensitive else sub['line'].lower()
            if query in text:
                results.append(sub.copy())

        return results

    def get_statistics(self) -> Dict:
        """Get statistics about the subtitles

        Returns:
            Dictionary with subtitle statistics
        """
        if not self.subtitles:
            return {'count': 0}

        durations = [sub['end_seconds'] - sub['start_seconds'] for sub in self.subtitles]
        word_counts = [len(sub['line'].split()) for sub in self.subtitles]
        char_counts = [len(sub['line']) for sub in self.subtitles]

        return {
            'count': len(self.subtitles),
            'total_duration': self.subtitles[-1]['end_seconds'],
            'avg_subtitle_duration': sum(durations) / len(durations),
            'avg_words_per_subtitle': sum(word_counts) / len(word_counts),
            'avg_chars_per_subtitle': sum(char_counts) / len(char_counts),
            'total_words': sum(word_counts),
            'total_chars': sum(char_counts)
        }

    def export_timerange_to_srt(
        self,
        start_time: float,
        end_time: float,
        output_file: str
    ):
        """Export a time range of subtitles to SRT format

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_file: Output SRT file path
        """
        subs = self.get_subtitles_in_range(start_time, end_time)

        with open(output_file, 'w') as f:
            for i, sub in enumerate(subs, 1):
                # Convert to SRT format (with comma instead of period)
                start_srt = sub['start'].replace('.', ',')
                end_srt = sub['end'].replace('.', ',')

                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{sub['line']}\n")
                f.write("\n")


def get_subtitle_for_frame(
    subtitle_file: str,
    frame_number: int,
    fps: float = 0.5
) -> Optional[Dict]:
    """Get subtitle for a specific frame number

    Args:
        subtitle_file: Path to subtitle JSON file
        frame_number: Frame number
        fps: Frames per second used for frame extraction (default: 0.5)

    Returns:
        Subtitle entry or None
    """
    loader = SubtitleLoader(subtitle_file)
    time_seconds = frame_number / fps
    return loader.get_subtitle_at_time(time_seconds)


def get_subtitles_for_frames(
    subtitle_file: str,
    frame_numbers: List[int],
    fps: float = 0.5,
    window_seconds: float = 2.0
) -> Dict[int, List[Dict]]:
    """Get subtitles for multiple frames with a time window

    Args:
        subtitle_file: Path to subtitle JSON file
        frame_numbers: List of frame numbers
        fps: Frames per second (default: 0.5)
        window_seconds: Time window around each frame (default: ±2s)

    Returns:
        Dictionary mapping frame_number -> list of subtitles
    """
    loader = SubtitleLoader(subtitle_file)
    results = {}

    for frame_num in frame_numbers:
        center_time = frame_num / fps
        start_time = max(0, center_time - window_seconds)
        end_time = center_time + window_seconds

        results[frame_num] = loader.get_subtitles_in_range(start_time, end_time)

    return results


def create_subtitle_context_for_question(
    subtitle_file: str,
    question: str,
    top_k: int = 5
) -> str:
    """Create subtitle context by searching for question-relevant subtitles

    Args:
        subtitle_file: Path to subtitle JSON file
        question: Question text
        top_k: Number of relevant subtitles to include

    Returns:
        Formatted subtitle context string
    """
    loader = SubtitleLoader(subtitle_file)

    # Extract key terms from question (simple approach)
    question_lower = question.lower()
    words = question_lower.split()
    # Remove common question words
    stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    keywords = [w for w in words if w not in stop_words and len(w) > 3]

    # Search for each keyword
    matches = []
    for keyword in keywords:
        results = loader.search_text(keyword, case_sensitive=False)
        matches.extend(results)

    # Remove duplicates and sort by time
    seen = set()
    unique_matches = []
    for match in matches:
        key = (match['start'], match['line'])
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    unique_matches.sort(key=lambda x: x['start_seconds'])

    # Take top_k
    selected = unique_matches[:top_k]

    # Format output
    if not selected:
        return "No relevant subtitles found."

    context = "Relevant subtitles:\n"
    for sub in selected:
        context += f"[{sub['start']} - {sub['end']}] {sub['line']}\n"

    return context


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python subtitle_utils.py <subtitle_file> [start_time] [end_time]")
        print("\nExample:")
        print("  python subtitle_utils.py /path/to/video_id_en.json 10.0 20.0")
        sys.exit(1)

    subtitle_file = sys.argv[1]
    loader = SubtitleLoader(subtitle_file)

    if len(sys.argv) >= 4:
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3])
        print(f"Subtitles from {start_time}s to {end_time}s:\n")
        text = loader.get_text_in_range(start_time, end_time)
        print(text)
    else:
        # Show statistics
        stats = loader.get_statistics()
        print("Subtitle Statistics:")
        print(f"  Total subtitles: {stats['count']}")
        print(f"  Total duration: {stats['total_duration']:.1f}s")
        print(f"  Avg subtitle duration: {stats['avg_subtitle_duration']:.2f}s")
        print(f"  Avg words per subtitle: {stats['avg_words_per_subtitle']:.1f}")
        print(f"  Total words: {stats['total_words']}")
