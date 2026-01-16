#!/usr/bin/env python3
"""
Setup LongVideoBench validation set
- Creates videos_processed directory structure
- Extracts frames at 1 FPS with short side = 224px
- Creates symlinks to validation videos
"""

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Paths
LVB_ROOT = "/mnt/ssd/data/longvideobench"
VAL_JSON = f"{LVB_ROOT}/lvb_val.json"
VIDEOS_SOURCE = f"{LVB_ROOT}/full_download/videos"
VIDEOS_PROCESSED = f"{LVB_ROOT}/videos_processed_val"

def extract_frames(video_path, output_dir, fps=1, short_side=224):
    """Extract frames from video at specified FPS with short side = 224px"""
    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg to extract frames with scaling
    # scale=-1:224 means: height=224, width=auto (maintains aspect ratio) if height is shorter
    # scale=224:-1 means: width=224, height=auto (maintains aspect ratio) if width is shorter
    # We use scale='min(224,iw)':'min(224,ih)':force_original_aspect_ratio=increase
    # which ensures short side = 224
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps},scale=\'min({short_side},iw)\':\'min({short_side},ih)\':force_original_aspect_ratio=increase,scale={short_side}:{short_side}:force_original_aspect_ratio=decrease',
        '-q:v', '2',  # High quality
        f'{output_dir}/frame_%04d.jpg',
        '-loglevel', 'error'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Rename frames to seconds format
        for frame_file in sorted(Path(output_dir).glob("frame_*.jpg")):
            frame_num = int(frame_file.stem.split('_')[1])
            seconds = frame_num - 1  # Frame 1 = second 0
            new_name = frame_file.parent / f"frame_{seconds:04d}.jpg"
            frame_file.rename(new_name)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return False

def setup_validation_set():
    """Setup validation set directory structure and extract frames"""

    # Load validation questions
    print("Loading validation set...")
    with open(VAL_JSON, 'r') as f:
        val_data = json.load(f)

    # Get unique video IDs
    video_ids = set()
    for item in val_data:
        video_ids.add(item.get('video_id', item.get('video', 'unknown')))

    video_ids = sorted(list(video_ids))
    print(f"Found {len(video_ids)} unique videos in validation set")
    print(f"Extracting frames with short side = 224px")

    # Create base directory
    os.makedirs(VIDEOS_PROCESSED, exist_ok=True)

    # Process each video
    print("\nProcessing validation videos...")
    successful = 0
    skipped = 0
    failed = 0

    failed_list = []

    for video_id in tqdm(video_ids):
        video_path = f"{VIDEOS_SOURCE}/{video_id}.mp4"
        video_dir = f"{VIDEOS_PROCESSED}/{video_id}"
        frames_dir = f"{video_dir}/frames"

        # Skip if already processed
        if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
            skipped += 1

            continue

        # Check if source video exists
        if not os.path.exists(video_path):
            print(f"\nWarning: Video not found: {video_id}")
            failed += 1
            failed_list.append(video_id)
            continue
        

        # Create directories
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        # Create symlink to original video
        symlink_path = f"{video_dir}/video.mp4"
        if not os.path.exists(symlink_path):
            try:
                os.symlink(video_path, symlink_path)
            except Exception as e:
                print(f"\nWarning: Could not create symlink for {video_id}: {e}")

        # Extract frames with short side = 224px
        if extract_frames(video_path, frames_dir, fps=1, short_side=224):
            successful += 1
        else:
            print(f"\nFailed to extract frames for {video_id}")
            failed += 1

    print("\n" + "="*60)
    print("Validation Set Setup Complete!")
    print("="*60)
    print(f"Successfully processed: {successful} videos")
    print(f"Already processed (skipped): {skipped} videos")
    print(f"Failed: {failed} videos")
    print(f"Total videos: {len(video_ids)}")
    print(f"\nOutput directory: {VIDEOS_PROCESSED}")
    print(f"Frame resolution: short side = 224px")
    print("="*60)

    with open(f"{LVB_ROOT}/failed_videos.txt", 'w') as f:
        for video_id in failed_list:
            f.write(f"{video_id}\n")

    print("length of failed list", len(failed_list))
        
    return failed_list

def clean_val_json(failed_list):
    val_data = json.load(open('/mnt/ssd/data/longvideobench/lvb_val.json', 'r'))
    total_removed = 0
    print("Length of failed list", len(failed_list))
    print("Removing... ")
    for item in val_data:
        if item['video_id'] in failed_list:
            val_data.remove(item)
            total_removed +=1
    
    json.dump(val_data, open('/mnt/ssd/data/longvideobench/lvb_val_cleaned.json', 'w'), indent=4)
    print(f"Total removed: {total_removed}")

if __name__ == "__main__":
    failed_list = setup_validation_set()
    clean_val_json(failed_list)
