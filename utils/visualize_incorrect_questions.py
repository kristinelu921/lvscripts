#!/usr/bin/env python3
"""
Script to visualize incorrect questions with their evidence frames in a grid layout.
Creates organized folders with frame grids and metadata for easy review.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

def create_frame_grid(frame_paths, video_base_path, grid_cols=4, max_size=300):
    """
    Create a grid of frames from the given paths.

    Args:
        frame_paths: List of relative frame paths
        video_base_path: Base path to the video's frames directory
        grid_cols: Number of columns in the grid
        max_size: Maximum width/height for each frame thumbnail

    Returns:
        PIL Image containing the grid
    """
    # Sort frame paths numerically by frame number
    def get_frame_number(frame_path):
        try:
            return int(frame_path.split('frame_')[1].split('.jpg')[0])
        except (IndexError, ValueError):
            return 0

    sorted_frame_paths = sorted(frame_paths, key=get_frame_number)

    # Load all frames
    frames = []
    for frame_path in sorted_frame_paths:
        full_path = video_base_path / frame_path
        if full_path.exists():
            try:
                img = Image.open(full_path)
                frames.append(img)
            except Exception as e:
                print(f"  Warning: Could not load {full_path}: {e}")
        else:
            print(f"  Warning: Frame not found: {full_path}")

    if not frames:
        # Create a placeholder image
        placeholder = Image.new('RGB', (max_size, max_size), color='gray')
        return placeholder

    # Calculate grid dimensions
    grid_rows = math.ceil(len(frames) / grid_cols)

    # Resize frames to thumbnails
    thumbnails = []
    for frame in frames:
        # Maintain aspect ratio
        frame.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        thumbnails.append(frame)

    # Find max dimensions for uniform grid
    max_width = max(thumb.width for thumb in thumbnails)
    max_height = max(thumb.height for thumb in thumbnails)

    # Create grid canvas
    grid_width = max_width * grid_cols
    grid_height = max_height * grid_rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Place thumbnails in grid
    for idx, thumb in enumerate(thumbnails):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(thumb, (x, y))

    return grid_image


def create_metadata_overlay(question_data, width=1600):
    """
    Create an image with metadata text overlaid.

    Args:
        question_data: Dictionary containing question metadata
        width: Width of the metadata image

    Returns:
        PIL Image with metadata text
    """
    # Prepare text
    text_parts = []
    text_parts.append(f"UID: {question_data['uid']}")
    text_parts.append(f"Video ID: {question_data['video_id']}")
    text_parts.append("")
    text_parts.append("QUESTION:")
    text_parts.append(textwrap.fill(question_data['question'], width=100))
    text_parts.append("")
    text_parts.append("CANDIDATES:")
    for idx, candidate in enumerate(question_data['candidates']):
        marker = "✓" if idx == question_data['correct_choice_idx'] else " "
        try:
            pred_idx = int(question_data['predicted_answer'])
            pred_marker = "✗" if idx == pred_idx else " "
        except (ValueError, KeyError):
            pred_marker = " "
        text_parts.append(f"  [{marker}][{pred_marker}] {idx}. {candidate}")
    text_parts.append("")
    # Handle predicted answer
    try:
        pred_idx = int(question_data['predicted_answer'])
        if 0 <= pred_idx < len(question_data['candidates']):
            pred_text = question_data['candidates'][pred_idx]
        else:
            pred_text = f"OUT OF RANGE (index {pred_idx})"
    except (ValueError, KeyError):
        pred_text = f"INVALID ({question_data['predicted_answer']})"

    text_parts.append(f"PREDICTED: {question_data['predicted_answer']} - {pred_text}")
    text_parts.append(f"CORRECT: {question_data['correct_choice_idx']} - {question_data['correct_answer']}")
    text_parts.append("")
    text_parts.append("REASONING:")
    text_parts.append(textwrap.fill(question_data.get('reasoning', 'N/A'), width=100))
    text_parts.append("")
    text_parts.append(f"Frames shown: {len(question_data['evidence_frames'])} / Total frames: {question_data.get('num_frames_in_video', 'N/A')}")

    text = "\n".join(text_parts)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Create temporary image to measure text
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Calculate text size
    lines = text.split('\n')
    line_heights = []
    max_line_width = 0

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        max_line_width = max(max_line_width, line_width)

    total_height = sum(line_heights) + len(lines) * 4  # 4px spacing

    # Create final image
    img_width = max(width, max_line_width + 40)
    img_height = total_height + 40
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw text
    y_offset = 20
    for line in lines:
        draw.text((20, y_offset), line, fill='black', font=font)
        bbox = draw.textbbox((20, y_offset), line, font=font)
        y_offset += (bbox[3] - bbox[1]) + 4

    return img


def process_incorrect_questions(json_path, output_dir, video_base_dir):
    """
    Process all incorrect questions and create visualizations.

    Args:
        json_path: Path to incorrect_questions.json
        output_dir: Directory to save visualizations
        video_base_dir: Base directory containing video frames
    """
    # Load questions
    with open(json_path, 'r') as f:
        questions = json.load(f)

    print(f"Processing {len(questions)} incorrect questions...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each question
    for idx, question in enumerate(questions):
        print(f"\nProcessing {idx + 1}/{len(questions)}: {question['uid']}")

        # Create folder for this question
        question_dir = output_path / f"{idx:04d}_{question['uid']}"
        question_dir.mkdir(exist_ok=True)

        # Get video frames directory
        video_id = question['video_id']
        video_frames_dir = Path(video_base_dir) / video_id

        # Create frame grid
        print(f"  Creating frame grid with {len(question['evidence_frames'])} frames...")
        grid_image = create_frame_grid(
            question['evidence_frames'],
            video_frames_dir,
            grid_cols=4
        )

        # Save frame grid
        grid_path = question_dir / "frames_grid.jpg"
        grid_image.save(grid_path, quality=95)
        print(f"  Saved: {grid_path}")

        # Create metadata overlay
        print("  Creating metadata overlay...")
        metadata_image = create_metadata_overlay(question, width=grid_image.width)

        # Save metadata
        metadata_path = question_dir / "metadata.jpg"
        metadata_image.save(metadata_path, quality=95)
        print(f"  Saved: {metadata_path}")

        # Create combined image (metadata on top, frames below)
        combined_height = metadata_image.height + grid_image.height
        combined_width = max(metadata_image.width, grid_image.width)
        combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
        combined_image.paste(metadata_image, (0, 0))
        combined_image.paste(grid_image, (0, metadata_image.height))

        # Save combined image
        combined_path = question_dir / "combined.jpg"
        combined_image.save(combined_path, quality=95)
        print(f"  Saved: {combined_path}")

        # Save raw metadata as JSON
        json_path_out = question_dir / "metadata.json"
        with open(json_path_out, 'w') as f:
            json.dump(question, f, indent=2)
        print(f"  Saved: {json_path_out}")

    print(f"\n✓ Done! Processed {len(questions)} questions.")
    print(f"Output directory: {output_path.absolute()}")


if __name__ == "__main__":
    # Paths
    incorrect_questions_json = "/mnt/ssd/data/lvscripts/lvb_val/analysis_20260122/incorrect_questions.json"
    output_directory = "/mnt/ssd/data/lvscripts/lvb_val/analysis_20260122/incorrect_questions_visualized"
    video_frames_base = "/mnt/ssd/data/longvideobench/videos_processed_val"

    process_incorrect_questions(
        incorrect_questions_json,
        output_directory,
        video_frames_base
    )
