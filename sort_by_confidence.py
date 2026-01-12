#!/usr/bin/env python3
"""
Sort questions from critic assessment files by confidence score
Loads all *critic_assessment.json files and outputs sorted results
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict


def load_all_critic_assessments(root_dir: str) -> List[Dict]:
    """
    Recursively load all critic assessment JSON files

    Args:
        root_dir: Root directory to search for *critic_assessment.json files

    Returns:
        List of all assessment dictionaries from all files
    """
    all_assessments = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_critic_assessment.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                        # Handle both list and dict formats
                        if isinstance(data, list):
                            assessments = data
                        elif isinstance(data, dict):
                            assessments = [data]
                        else:
                            print(f"Warning: Unexpected format in {filepath}")
                            continue

                        # Add source file to each assessment for traceability
                        for assessment in assessments:
                            assessment['_source_file'] = filepath

                        all_assessments.extend(assessments)
                        print(f"✓ Loaded {len(assessments)} assessments from {filepath}")

                except Exception as e:
                    print(f"✗ Error reading {filepath}: {e}")

    return all_assessments


def sort_by_confidence(assessments: List[Dict], reverse: bool = False) -> List[Dict]:
    """
    Sort assessments by confidence score

    Args:
        assessments: List of assessment dictionaries
        reverse: If True, sort high to low. If False, sort low to high.

    Returns:
        Sorted list of assessments
    """
    # Handle missing confidence scores by treating them as -1
    return sorted(
        assessments,
        key=lambda x: x.get('confidence', -1),
        reverse=reverse
    )


def print_sorted_results(assessments: List[Dict], limit: int = None):
    """
    Print sorted assessments in a readable format

    Args:
        assessments: Sorted list of assessments
        limit: Optional limit on number of results to display
    """
    display_items = assessments[:limit] if limit else assessments

    print("\n" + "="*80)
    print(f"SORTED QUESTIONS BY CONFIDENCE SCORE")
    print("="*80)
    print(f"Total assessments: {len(assessments)}")
    if limit and len(assessments) > limit:
        print(f"Displaying: {limit} (use --limit to adjust)")
    print("="*80 + "\n")

    for i, assessment in enumerate(display_items, 1):
        confidence = assessment.get('confidence', -1)
        uid = assessment.get('uid', 'unknown')
        question = assessment.get('question', 'No question text')
        answer = assessment.get('answer', 'N/A')
        source_file = assessment.get('_source_file', 'unknown')

        # Truncate long questions for display
        if len(question) > 100:
            question_display = question[:97] + "..."
        else:
            question_display = question

        print(f"{i}. Confidence: {confidence}% | UID: {uid}")
        print(f"   Answer: {answer}")
        print(f"   Question: {question_display}")
        print(f"   Source: {source_file}")

        # Show suggestions if confidence is low
        if confidence < 70:
            possible_errors = assessment.get('possible_errors', [])
            suggestion = assessment.get('suggestion')

            if possible_errors:
                print(f"   Concerns: {', '.join(possible_errors)}")
            if suggestion:
                print(f"   Suggestion: {suggestion}")

        print()


def save_sorted_results(assessments: List[Dict], output_file: str):
    """
    Save sorted assessments to a JSON file

    Args:
        assessments: Sorted list of assessments
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(assessments, f, indent=2)

    print(f"\n✓ Saved {len(assessments)} sorted assessments to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Sort questions from critic assessment files by confidence score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sort all assessments in a directory (low to high confidence)
  python sort_by_confidence.py /path/to/videos_processed

  # Sort high to low confidence
  python sort_by_confidence.py /path/to/videos_processed --reverse

  # Save to output file
  python sort_by_confidence.py /path/to/videos_processed -o sorted_results.json

  # Display only top 10 results
  python sort_by_confidence.py /path/to/videos_processed --limit 10 --reverse
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Root directory containing *critic_assessment.json files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )

    parser.add_argument(
        '-r', '--reverse',
        action='store_true',
        help='Sort high to low (default: low to high)'
    )

    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='Limit number of results displayed (default: show all)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=int,
        default=None,
        help='Only show assessments below this confidence threshold'
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1

    print(f"Searching for critic assessment files in: {args.directory}")
    print("="*80)

    # Load all assessments
    assessments = load_all_critic_assessments(args.directory)

    if not assessments:
        print("\nNo critic assessment files found!")
        return 1

    # Filter by confidence threshold if specified
    if args.confidence_threshold is not None:
        original_count = len(assessments)
        assessments = [a for a in assessments if a.get('confidence', -1) < args.confidence_threshold]
        print(f"\nFiltered to {len(assessments)}/{original_count} assessments below {args.confidence_threshold}% confidence")

    # Sort assessments
    sorted_assessments = sort_by_confidence(assessments, reverse=args.reverse)

    # Print results
    print_sorted_results(sorted_assessments, limit=args.limit)

    # Save to file if requested
    if args.output:
        save_sorted_results(sorted_assessments, args.output)

    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    confidences = [a.get('confidence', -1) for a in assessments if a.get('confidence', -1) >= 0]

    if confidences:
        print(f"Total assessments: {len(assessments)}")
        print(f"Average confidence: {sum(confidences) / len(confidences):.1f}%")
        print(f"Lowest confidence: {min(confidences)}%")
        print(f"Highest confidence: {max(confidences)}%")

        # Count by confidence ranges
        low = sum(1 for c in confidences if c < 50)
        medium = sum(1 for c in confidences if 50 <= c < 80)
        high = sum(1 for c in confidences if c >= 80)

        print(f"\nBreakdown:")
        print(f"  Low (<50%): {low}")
        print(f"  Medium (50-79%): {medium}")
        print(f"  High (≥80%): {high}")

    return 0


if __name__ == "__main__":
    exit(main())
