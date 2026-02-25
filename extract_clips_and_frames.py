#!/usr/bin/env python3
"""
Extract frames (1fps) and clips (scene-based, 2fps) from videos for Kimi processing.
Processes videos in kimi/{dataset}/video_files/{video_id}/ structure.

Usage:
    python extract_clips_and_frames.py --dataset longvideobench
    python extract_clips_and_frames.py --dataset all --workers 8
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"    Warning: Could not get duration: {e}")
        return None


def detect_scenes(video_path, threshold=0.3):
    """
    Detect scene changes in video using ffmpeg scene detection.

    Returns: List of scene change timestamps in seconds.
    """
    try:
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-filter:v', f'select=\'gt(scene,{threshold})\',showinfo',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Parse timestamps from showinfo output
        timestamps = []
        output = result.stdout if result.stdout else ""
        for line in output.split('\n'):
            if 'pts_time:' in line:
                try:
                    pts_time = line.split('pts_time:')[1].split()[0]
                    timestamps.append(float(pts_time))
                except (IndexError, ValueError):
                    continue

        # Add start timestamp if not present
        if not timestamps or timestamps[0] > 0:
            timestamps.insert(0, 0.0)

        return timestamps
    except Exception as e:
        print(f"    Warning: Scene detection failed: {e}")
        return [0.0]


def extract_frames_for_video(video_path, output_dir, fps=1):
    """Extract frames at specified FPS with 224px smaller side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing_frames = list(output_dir.glob('frame_*.jpg'))
    if existing_frames:
        return {
            'status': 'skipped',
            'frames_extracted': len(existing_frames),
            'message': 'Frames already exist'
        }

    # Extract frames
    output_pattern = str(output_dir / 'frame_%04d.jpg')
    cmd = [
        'ffmpeg', '-threads', '0',
        '-i', str(video_path),
        '-vf', f"fps={fps},scale='if(gte(iw,ih),-2,224)':'if(gte(iw,ih),224,-2)'",
        '-q:v', '2',
        output_pattern, '-y'
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        extracted_frames = list(output_dir.glob('frame_*.jpg'))
        return {
            'status': 'success',
            'frames_extracted': len(extracted_frames)
        }
    except subprocess.CalledProcessError as e:
        return {
            'status': 'error',
            'frames_extracted': 0,
            'error': str(e)
        }


def extract_clips_for_video(video_path, output_dir, scene_timestamps, max_clip_duration=120, fps=2):
    """
    Extract video clips based on scene timestamps.
    Clips are named clip_{start_seconds}_{end_seconds}.mp4
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if clips already exist
    existing_clips = list(output_dir.glob('clip_*.mp4'))
    if existing_clips:
        # Load existing metadata if available
        metadata_file = output_dir / 'clips_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return {
                    'status': 'skipped',
                    'clips_extracted': len(metadata),
                    'message': 'Clips already exist'
                }
        else:
            return {
                'status': 'skipped',
                'clips_extracted': len(existing_clips),
                'message': 'Clips already exist'
            }

    # Get video duration
    duration = get_video_duration(video_path)
    if not duration:
        return {'status': 'error', 'clips_extracted': 0, 'error': 'Could not get duration'}

    # Create clip segments (split long scenes into max_clip_duration chunks)
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
                'start': current_start,
                'end': current_end,
                'duration': current_end - current_start
            })
            current_start = current_end

    # Extract each clip with timestamp-based naming
    extracted_clips = []
    for clip_info in clips_info:
        start_sec = int(clip_info['start'])
        end_sec = int(clip_info['end'])
        clip_filename = f"clip_{start_sec}_{end_sec}.mp4"
        clip_path = output_dir / clip_filename

        # FFmpeg command: extract clip at 2fps, 224px smaller side
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(clip_info['start']),
            '-i', str(video_path),
            '-t', str(clip_info['duration']),
            '-vf', f"fps={fps},scale='if(gte(iw,ih),-2,224)':'if(gte(iw,ih),224,-2)'",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-an',
            str(clip_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            extracted_clips.append({
                'filename': clip_filename,
                'start': clip_info['start'],
                'end': clip_info['end'],
                'duration': clip_info['duration']
            })
        except subprocess.CalledProcessError as e:
            print(f"      Warning: Failed to extract {clip_filename}: {e}")
            continue

    # Save clips metadata
    metadata_file = output_dir / 'clips_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(extracted_clips, f, indent=2)

    return {
        'status': 'success',
        'clips_extracted': len(extracted_clips),
        'clips_info': extracted_clips
    }


def process_single_video(video_id, dataset_dir):
    """Process a single video: extract frames and clips."""
    video_files_dir = dataset_dir / 'video_files' / video_id
    videos_dir = dataset_dir / 'videos'

    # Find video file
    video_file = videos_dir / f"{video_id}.mp4"
    if not video_file.exists():
        # Try other common extensions
        for ext in ['.mkv', '.avi', '.mov', '.webm']:
            alt_file = videos_dir / f"{video_id}{ext}"
            if alt_file.exists():
                video_file = alt_file
                break

    if not video_file.exists():
        print(f"  ✗ {video_id}: Video file not found")
        return {'video_id': video_id, 'status': 'error', 'error': 'Video file not found'}

    print(f"  Processing {video_id}...")

    # Create output directories
    frames_dir = video_files_dir / 'frames'
    clips_dir = video_files_dir / 'clips'

    # Extract frames at 1 FPS
    print(f"    Extracting frames...")
    frames_result = extract_frames_for_video(video_file, frames_dir, fps=1)

    # Detect scenes
    print(f"    Detecting scenes...")
    scene_timestamps = detect_scenes(video_file, threshold=0.3)
    print(f"    Found {len(scene_timestamps)} scenes")

    # Extract clips at 2 FPS
    print(f"    Extracting clips...")
    clips_result = extract_clips_for_video(video_file, clips_dir, scene_timestamps, max_clip_duration=120, fps=2)

    print(f"  ✓ {video_id}: {frames_result['frames_extracted']} frames, {clips_result['clips_extracted']} clips")

    return {
        'video_id': video_id,
        'frames': frames_result,
        'clips': clips_result,
        'status': 'success'
    }


def process_dataset(dataset_name, base_dir, workers=4):
    """Process all videos in a dataset."""
    dataset_dir = Path(base_dir) / 'kimi' / dataset_name
    video_files_dir = dataset_dir / 'video_files'

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    # Get list of video IDs
    video_ids = [d.name for d in video_files_dir.iterdir() if d.is_dir()]

    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    # Process videos in parallel
    if workers > 1:
        with Pool(processes=min(workers, cpu_count())) as pool:
            process_func = partial(process_single_video, dataset_dir=dataset_dir)
            results = pool.map(process_func, video_ids)
    else:
        results = [process_single_video(vid_id, dataset_dir) for vid_id in video_ids]

    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    print(f"\n{dataset_name} Summary: {successful}/{len(video_ids)} videos processed successfully")


def main():
    parser = argparse.ArgumentParser(description='Extract frames and clips for Kimi processing')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    print("="*60)
    print("Kimi Video Clips & Frames Extraction")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Workers: {args.workers}")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        process_dataset(dataset, args.base_dir, args.workers)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Run caption_clips_kimi.py to generate clip captions")
    print("  2. Run embed_clip_captions.py to generate embeddings")
    print("="*60)


if __name__ == '__main__':
    main()
