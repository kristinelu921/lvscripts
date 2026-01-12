#!/usr/bin/env python3
"""
Extract subtitles from longvideobench, convert timestamps to frame numbers,
and map questions/GT answers to downloaded videos.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def timestamp_to_seconds(timestamp):
    """Convert HH:MM:SS.mmm to total seconds"""
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_frame(seconds, fps=1):
    """Convert seconds to frame number (assuming 1 fps for frame extraction)"""
    return int(seconds)

def extract_subtitle_frames(subtitle_path, fps=1):
    """
    Extract subtitles and map them to frame numbers.

    Args:
        subtitle_path: Path to subtitle JSON file
        fps: Frames per second (default 1, since frames are extracted at 1fps)

    Returns:
        Dictionary mapping frame numbers to subtitle lines
    """
    if not os.path.exists(subtitle_path):
        return {}

    with open(subtitle_path, 'r') as f:
        subtitles = json.load(f)

    frame_subtitles = defaultdict(list)

    for sub in subtitles:
        # Skip entries that don't have required fields
        if not isinstance(sub, dict) or 'start' not in sub or 'end' not in sub or 'line' not in sub:
            continue

        start_sec = timestamp_to_seconds(sub['start'])
        end_sec = timestamp_to_seconds(sub['end'])

        start_frame = seconds_to_frame(start_sec, fps)
        end_frame = seconds_to_frame(end_sec, fps)

        # Map subtitle to all frames in its duration
        for frame_num in range(start_frame, end_frame + 1):
            frame_subtitles[frame_num].append(sub['line'])

    # Combine multiple subtitle lines for each frame
    frame_subtitles_combined = {}
    for frame, lines in frame_subtitles.items():
        frame_subtitles_combined[frame] = ' '.join(lines)

    return frame_subtitles_combined

def get_video_id_from_filename(filename):
    """Extract video ID from .mp4 filename"""
    return filename.replace('.mp4', '')

def extract_questions_for_videos(json_path, video_ids):
    """
    Extract questions and GT answers for specific video IDs.

    Args:
        json_path: Path to lvb_val.json or lvb_test_wo_gt.json
        video_ids: Set of video IDs to extract questions for

    Returns:
        Dictionary mapping video_id to list of questions
    """
    with open(json_path, 'r') as f:
        all_questions = json.load(f)

    video_questions = defaultdict(list)

    for q in all_questions:
        video_id = q['video_id']
        if video_id in video_ids:
            question_data = {
                'uid': q['id'],
                'question': q['question'],
                'candidates': q['candidates'],
                'correct_choice': q.get('correct_choice', None),  # May not exist in test set
                'position': q.get('position', []),
                'topic_category': q.get('topic_category', ''),
                'question_category': q.get('question_category', ''),
                'level': q.get('level', '')
            }
            video_questions[video_id].append(question_data)

    return video_questions

def main():
    # Paths
    videos_dir = Path('/mnt/ssh/data/longvideobench/videos')
    subtitles_dir = Path('/mnt/ssh/data/longvideobench/subtitles')
    val_json = Path('/mnt/ssh/data/longvideobench/lvb_val.json')
    output_dir = Path('/mnt/ssh/data/longvideobench')

    # Get all downloaded video IDs
    video_files = list(videos_dir.glob('*.mp4'))
    video_ids = {get_video_id_from_filename(f.name) for f in video_files}

    print(f"Found {len(video_ids)} downloaded videos")

    # Extract questions for downloaded videos
    print("Extracting questions...")
    video_questions = extract_questions_for_videos(val_json, video_ids)

    # Save questions per video
    questions_output = output_dir / 'downloaded_videos_questions.json'
    with open(questions_output, 'w') as f:
        json.dump(video_questions, f, indent=2)
    print(f"Saved questions to {questions_output}")

    # Process subtitles for each video
    print("Processing subtitles...")
    all_subtitle_frames = {}

    for video_id in video_ids:
        subtitle_file = subtitles_dir / f'{video_id}_en.json'

        if subtitle_file.exists():
            frame_subs = extract_subtitle_frames(subtitle_file)
            if frame_subs:
                all_subtitle_frames[video_id] = {
                    'frames': frame_subs,
                    'subtitle_path': str(subtitle_file)
                }
                print(f"  {video_id}: {len(frame_subs)} frames with subtitles")
        else:
            print(f"  {video_id}: No subtitle file found")

    # Save subtitle frames mapping
    subtitles_output = output_dir / 'subtitles_frame_mapping.json'
    with open(subtitles_output, 'w') as f:
        json.dump(all_subtitle_frames, f, indent=2)
    print(f"Saved subtitle mappings to {subtitles_output}")

    # Create summary statistics
    stats = {
        'total_videos': len(video_ids),
        'videos_with_questions': len(video_questions),
        'total_questions': sum(len(qs) for qs in video_questions.values()),
        'videos_with_subtitles': len(all_subtitle_frames),
        'videos_without_subtitles': len(video_ids) - len(all_subtitle_frames)
    }

    stats_output = output_dir / 'extraction_stats.json'
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*60)

if __name__ == '__main__':
    main()
