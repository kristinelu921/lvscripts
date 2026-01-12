"""
Subtitle search tool for LLM integration

Provides a simple interface for searching subtitle content during video QA pipeline.
Can be used as a tool that the LLM calls when questions require subtitle information.
"""
import json
import os
from pathlib import Path
from typing import Dict, List
from search_subtitles import search_subtitles


def search_video_subtitles(
    video_id: str,
    query: str,
    topk: int = 10,
    fps: float = 1.0,
    subtitles_dir: str = "/mnt/ssd/data/longvideobench/subtitles"
) -> List[Dict]:
    """Search subtitles for a specific video

    This function can be called by the LLM when it needs to find specific
    information in video subtitles.

    Args:
        video_id: Video identifier (e.g., "Y0IaijKNGX8")
        query: Natural language search query
        topk: Number of top results to return (default: 10)
        fps: Frames per second for frame conversion (default: 1.0)
        subtitles_dir: Directory containing subtitle files

    Returns:
        List of dicts with:
            - rank: Result ranking (1-indexed)
            - score: Similarity score (0-1)
            - text: Subtitle text content
            - start: Start timestamp (HH:MM:SS.mmm)
            - end: End timestamp (HH:MM:SS.mmm)
            - start_sec: Start time in seconds (float)
            - end_sec: End time in seconds (float)
            - start_frame: Start frame number (int)
            - end_frame: End frame number (int)
            - time_formatted: Human-readable time (M:SS or H:MM:SS)

    Example:
        >>> results = search_video_subtitles("Y0IaijKNGX8", "master chief arrives")
        >>> print(results[0]["text"])
        >>> print(f"Found at {results[0]['time_formatted']}, frame {results[0]['start_frame']}")
    """
    # Construct path to subtitle embeddings
    embeddings_file = f"{video_id}_en_embeddings.jsonl"
    embeddings_path = os.path.join(subtitles_dir, embeddings_file)

    # Check if embeddings exist
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Subtitle embeddings not found: {embeddings_path}\n"
            f"Run embed_subtitles.py first to generate embeddings."
        )

    # Search subtitles
    results = search_subtitles(embeddings_path, query, topk, fps)

    return results


def format_results_for_llm(results: List[Dict]) -> str:
    """Format search results as readable text for LLM consumption

    Args:
        results: List of search results from search_video_subtitles

    Returns:
        Formatted string with search results
    """
    if not results:
        return "No matching subtitles found."

    lines = ["Subtitle Search Results:", ""]

    for res in results:
        lines.append(f"[{res['rank']}] Time: {res['time_formatted']} (Frame {res['start_frame']}-{res['end_frame']})")
        lines.append(f"    Score: {res['score']:.3f}")
        lines.append(f"    Text: {res['text']}")
        lines.append("")

    return "\n".join(lines)


# Async version for pipeline integration
async def search_subtitles_async(
    video_id: str,
    question_uid: str,
    query: str,
    topk: int = 10,
    fps: float = 1.0
) -> List[Dict]:
    """Async wrapper for subtitle search (compatible with existing pipeline)

    Args:
        video_id: Video identifier
        question_uid: Unique question identifier (for logging)
        query: Search query
        topk: Number of results
        fps: Frames per second

    Returns:
        List of search results
    """
    results = search_video_subtitles(video_id, query, topk, fps)

    # Log for tracking (similar to caption search)
    log_dir = f"logs/log_video_{video_id}_{question_uid}"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/subtitle_search.log", "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Top result: {results[0]['text'] if results else 'None'}\n")
        f.write(f"Timestamp: {results[0]['start']} ({results[0]['start_frame']} frames)\n\n")

    return results


def main():
    """Test the subtitle search tool"""
    import argparse

    parser = argparse.ArgumentParser(description="Test subtitle search tool")
    parser.add_argument("video_id", help="Video ID (e.g., Y0IaijKNGX8)")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--topk", type=int, default=5, help="Number of results")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")

    args = parser.parse_args()

    print(f"\nSearching subtitles for video: {args.video_id}")
    print(f"Query: {args.query}\n")

    try:
        results = search_video_subtitles(args.video_id, args.query, args.topk, args.fps)
        print(format_results_for_llm(results))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
