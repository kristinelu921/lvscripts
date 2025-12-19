#!/usr/bin/env python3
"""
VideoMME Dataset Adapter for Long-Context Video Understanding Pipeline

This script adapts the existing pipeline to work with the VideoMME dataset from HuggingFace.

Usage:
    python run_videomme.py --output_dir ./videomme_results --subset validation --max_videos 10

Dataset structure:
    - video_id: Video identifier
    - videoID: YouTube video ID
    - question: Question text
    - options: List of 4 options [A, B, C, D]
    - answer: Correct answer letter
    - duration: Video duration category
    - domain: Content domain
    - sub_category: Content subcategory
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import existing pipeline components
from caption_frames_os import (
    caption_frame_with_os,
    create_logs,
    CES_log_prompt,
    global_summary_prompt,
    sort_captions
)
from os_model import Pipeline as OSPipeline, query_model_iterative_with_retry as os_query
from critic_model_os import CriticPipeline, critic_assess
from critic_response import (
    Pipeline as CriticRespPipeline,
    query_model_iterative_with_retry as critic_resp_query,
    create_enhanced_prompt,
)


def download_youtube_video(video_id: str, output_path: str) -> bool:
    """
    Download YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID
        output_path: Path to save the video file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if yt-dlp is installed
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return False

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",  # Prefer mp4 format
            "-o", output_path,
            url
        ]

        print(f"Downloading video {video_id}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Download failed: {result.stderr}")
            return False

        return os.path.exists(output_path)

    except subprocess.TimeoutExpired:
        print(f"Download timed out for video {video_id}")
        return False
    except Exception as e:
        print(f"Error downloading video {video_id}: {e}")
        return False


def extract_frames_ffmpeg(video_path: str, frames_dir: str, fps: int = 1) -> bool:
    """
    Extract frames from video at specified FPS using ffmpeg.

    Args:
        video_path: Path to video file
        frames_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Install with: sudo apt-get install ffmpeg")
        return False

    os.makedirs(frames_dir, exist_ok=True)

    try:
        # Extract frames at specified FPS
        # Output format: frame_0001.jpg, frame_0002.jpg, etc. (4-digit zero-padded frame numbers)
        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality
            output_pattern,
            "-y"  # Overwrite existing files
        ]

        print(f"Extracting frames from {video_path} at {fps} FPS...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Frame extraction failed: {result.stderr}")
            return False

        # Check if frames were created
        frame_count = len(list(Path(frames_dir).glob("frame_*.jpg")))
        print(f"Extracted {frame_count} frames")
        return frame_count > 0

    except subprocess.TimeoutExpired:
        print("Frame extraction timed out")
        return False
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return False


async def setup_video_directory(
    video_id: str,
    youtube_id: str,
    output_dir: str,
    vlm_model: str,
    skip_existing: bool = True
) -> Optional[str]:
    """
    Setup video directory with all necessary preprocessing:
    1. Download video from YouTube
    2. Extract frames at 1 FPS
    3. Generate frame captions
    4. Generate CES logs and global summary
    5. Embed captions for semantic search

    Args:
        video_id: VideoMME video identifier
        youtube_id: YouTube video ID
        output_dir: Base output directory
        vlm_model: VLM model name for captioning
        skip_existing: Skip if video already processed

    Returns:
        Path to video directory if successful, None otherwise
    """
    vid_path = os.path.join(output_dir, video_id)
    frames_dir = os.path.join(vid_path, "frames")
    captions_dir = os.path.join(vid_path, "captions")

    # Check if already processed
    if skip_existing:
        embeddings_file = os.path.join(captions_dir, "frame_captions_sorted_embeddings.jsonl")
        if os.path.exists(embeddings_file):
            print(f"Video {video_id} already processed, skipping...")
            return vid_path

    os.makedirs(vid_path, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)

    # Save video metadata
    metadata = {
        "video_id": video_id,
        "youtube_id": youtube_id,
        "youtube_url": f"https://www.youtube.com/watch?v={youtube_id}",
        "status": "processing"
    }
    with open(os.path.join(vid_path, f"{video_id}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 1: Download video
    video_path = os.path.join(vid_path, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        if not download_youtube_video(youtube_id, video_path):
            print(f"Failed to download video {video_id}")
            return None
    else:
        print(f"Video already downloaded: {video_path}")

    # Step 2: Extract frames
    if not os.path.exists(frames_dir) or len(list(Path(frames_dir).glob("frame_*.jpg"))) == 0:
        if not extract_frames_ffmpeg(video_path, frames_dir, fps=1):
            print(f"Failed to extract frames for video {video_id}")
            return None
    else:
        print(f"Frames already extracted: {frames_dir}")

    # Step 3: Generate captions
    captions_file = os.path.join(captions_dir, "frame_captions.json")
    captions_sorted_file = os.path.join(captions_dir, "frame_captions_sorted.json")

    if not os.path.exists(captions_file):
        print(f"Generating captions for {video_id}...")
        try:
            # Change to captions directory for caption_frame_with_os
            original_dir = os.getcwd()
            os.chdir(vid_path)
            await caption_frame_with_os(
                frames_dir="frames",
                output_file="captions/frame_captions.json",
                max_concurrent=20
            )
            os.chdir(original_dir)

            # Sort captions
            sort_captions(vid_path)
        except Exception as e:
            print(f"Failed to generate captions: {e}")
            os.chdir(original_dir)
            return None
    else:
        print(f"Captions already exist: {captions_file}")
        # Ensure sorted version exists
        if not os.path.exists(captions_sorted_file):
            sort_captions(vid_path)

    # Step 4: Generate CES logs
    ces_logs_file = os.path.join(captions_dir, "CES_logs.txt")
    if not os.path.exists(ces_logs_file):
        print(f"Generating CES logs for {video_id}...")
        try:
            original_dir = os.getcwd()
            os.chdir(vid_path)
            await create_logs(
                captions_dir="captions/frame_captions_sorted.json",
                output_file="captions/CES_logs.txt",
                prompt_fct=CES_log_prompt,
                frames_dir="frames",
                model=vlm_model
            )
            os.chdir(original_dir)
        except Exception as e:
            print(f"Failed to generate CES logs: {e}")
            os.chdir(original_dir)
            return None
    else:
        print(f"CES logs already exist: {ces_logs_file}")

    # Step 5: Generate global summary
    global_summary_file = os.path.join(captions_dir, "global_summary.txt")
    if not os.path.exists(global_summary_file):
        print(f"Generating global summary for {video_id}...")
        try:
            original_dir = os.getcwd()
            os.chdir(vid_path)
            await create_logs(
                captions_dir="captions/frame_captions_sorted.json",
                output_file="captions/global_summary.txt",
                prompt_fct=global_summary_prompt,
                frames_dir="frames",
                model=vlm_model
            )
            os.chdir(original_dir)
        except Exception as e:
            print(f"Failed to generate global summary: {e}")
            os.chdir(original_dir)
            return None
    else:
        print(f"Global summary already exists: {global_summary_file}")

    # Step 6: Embed captions
    embeddings_file = os.path.join(captions_dir, "frame_captions_sorted_embeddings.jsonl")
    if not os.path.exists(embeddings_file):
        print(f"Embedding captions for {video_id}...")
        try:
            # Import embed function
            from embed_frame_captions import main as embed_main
            import sys
            old_argv = sys.argv
            sys.argv = ['embed_frame_captions.py', captions_sorted_file, embeddings_file]
            embed_main()
            sys.argv = old_argv
        except Exception as e:
            print(f"Failed to embed captions: {e}")
            return None
    else:
        print(f"Embeddings already exist: {embeddings_file}")

    # Update metadata
    metadata["status"] = "ready"
    with open(os.path.join(vid_path, f"{video_id}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return vid_path


def format_videomme_question(question: str, options: List[str]) -> str:
    """
    Format VideoMME question with multiple choice options.

    Args:
        question: Question text
        options: List of 4 options

    Returns:
        Formatted question string
    """
    formatted = f"{question}\n\n"
    formatted += "Options:\n"
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D
        formatted += f"{letter}. {option}\n"
    formatted += "\nAnswer with a single letter (A, B, C, or D)."
    return formatted


async def process_videomme_question(
    example: Dict[str, Any],
    output_dir: str,
    llm_model: str,
    vlm_model: str,
    use_critic: bool = True
) -> Dict[str, Any]:
    """
    Process a single VideoMME question through the pipeline.

    Args:
        example: VideoMME dataset example
        output_dir: Base output directory
        llm_model: LLM model name
        vlm_model: VLM model name
        use_critic: Whether to use critic module

    Returns:
        Result dictionary with predictions and metadata
    """
    video_id = example["video_id"]
    question_id = example["question_id"]
    question = example["question"]
    options = example["options"]
    ground_truth = example["answer"]

    vid_path = os.path.join(output_dir, video_id)

    # Format question with options
    formatted_question = format_videomme_question(question, options)

    # Run OS model
    print(f"\nProcessing question {question_id} for video {video_id}...")
    os_model = OSPipeline(llm_model, vlm_model)
    answers_file = os.path.join(vid_path, f"{video_id}_answers.json")

    try:
        answer = await os_query(os_model, formatted_question, question_id, vid_path, answers_file)

        if isinstance(answer, str):
            # Load from file
            with open(answers_file, "r") as f:
                answers = json.load(f)
            answer = next((a for a in answers if a.get("uid") == question_id), None)

        if not answer or not isinstance(answer, dict):
            print(f"Failed to get answer for {question_id}")
            return None

        result = {
            "video_id": video_id,
            "question_id": question_id,
            "question": question,
            "options": options,
            "ground_truth": ground_truth,
            "os_prediction": answer.get("answer", ""),
            "os_reasoning": answer.get("reasoning", ""),
            "os_frames": answer.get("evidence_frame_numbers", []),
            "correct": answer.get("answer", "").strip().upper() == ground_truth.upper(),
        }

        # Run critic if enabled
        if use_critic:
            try:
                critic = CriticPipeline(llm_model, vlm_model)
                assessment = await critic_assess(
                    critic,
                    formatted_question,
                    question_id,
                    answer.get("answer"),
                    answer.get("reasoning", ""),
                    answer.get("evidence_frame_numbers", []),
                    output_dir,
                    video_id,
                )

                result["critic_confidence"] = assessment.get("confidence", 0)
                result["critic_feedback"] = assessment.get("feedback", "")

                # Re-evaluate if confidence < 70%
                if assessment.get("confidence", 100) < 70:
                    print(f"Low confidence ({assessment.get('confidence')}%), re-evaluating...")
                    enhanced_question = create_enhanced_prompt(assessment)
                    critic_resp_model = CriticRespPipeline(llm_model, vlm_model)
                    re_eval = await critic_resp_query(
                        critic_resp_model,
                        enhanced_question,
                        question_id,
                        vid_path,
                    )

                    result["re_eval_prediction"] = re_eval.get("answer", "")
                    result["re_eval_reasoning"] = re_eval.get("reasoning", "")
                    result["final_prediction"] = re_eval.get("answer", "")
                    result["final_correct"] = re_eval.get("answer", "").strip().upper() == ground_truth.upper()
                else:
                    result["final_prediction"] = result["os_prediction"]
                    result["final_correct"] = result["correct"]

            except Exception as e:
                print(f"Critic failed for {question_id}: {e}")
                result["final_prediction"] = result["os_prediction"]
                result["final_correct"] = result["correct"]
        else:
            result["final_prediction"] = result["os_prediction"]
            result["final_correct"] = result["correct"]

        return result

    except Exception as e:
        print(f"Error processing question {question_id}: {e}")
        return None


async def run_videomme_evaluation(
    dataset_subset: str = "validation",
    output_dir: str = "./videomme_results",
    llm_model: str = "deepseek-ai/DeepSeek-V3.1",
    vlm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    max_videos: Optional[int] = None,
    max_questions_per_video: Optional[int] = None,
    use_critic: bool = True,
    skip_existing: bool = True
):
    """
    Run evaluation on VideoMME dataset.

    Args:
        dataset_subset: Dataset subset to use (validation, test, etc.)
        output_dir: Directory to store results
        llm_model: LLM model name
        vlm_model: VLM model name
        max_videos: Maximum number of videos to process (None = all)
        max_questions_per_video: Maximum questions per video (None = all)
        use_critic: Whether to use critic module
        skip_existing: Skip already processed videos
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not found. Install with: pip install datasets")
        return

    print(f"Loading VideoMME dataset (subset: {dataset_subset})...")
    try:
        ds = load_dataset("lmms-lab/Video-MME", split=dataset_subset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset loaded: {len(ds)} examples")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Group questions by video
    video_questions = {}
    for example in ds:
        video_id = example["video_id"]
        if video_id not in video_questions:
            video_questions[video_id] = []
        video_questions[video_id].append(example)

    print(f"Found {len(video_questions)} unique videos")

    # Limit number of videos if specified
    video_ids = list(video_questions.keys())
    if max_videos:
        video_ids = video_ids[:max_videos]
        print(f"Processing first {max_videos} videos")

    # Process each video
    for video_idx, video_id in enumerate(video_ids, 1):
        print(f"\n{'='*80}")
        print(f"Video {video_idx}/{len(video_ids)}: {video_id}")
        print(f"{'='*80}")

        examples = video_questions[video_id]
        if max_questions_per_video:
            examples = examples[:max_questions_per_video]

        print(f"Processing {len(examples)} questions for this video")

        # Get YouTube ID from first example
        youtube_id = examples[0]["videoID"]

        # Setup video directory (download, extract, caption, embed)
        vid_path = await setup_video_directory(
            video_id, youtube_id, output_dir, vlm_model, skip_existing
        )

        if not vid_path:
            print(f"Failed to setup video {video_id}, skipping...")
            continue

        # Process each question for this video
        for example in examples:
            result = await process_videomme_question(
                example, output_dir, llm_model, vlm_model, use_critic
            )

            if result:
                results.append(result)

                # Save intermediate results
                results_file = os.path.join(output_dir, "videomme_results.json")
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)

        # Print progress
        if results:
            correct = sum(1 for r in results if r.get("final_correct", False))
            accuracy = correct / len(results) * 100
            print(f"\nProgress: {len(results)} questions processed, {correct} correct ({accuracy:.2f}%)")

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total questions processed: {len(results)}")

    if results:
        os_correct = sum(1 for r in results if r.get("correct", False))
        final_correct = sum(1 for r in results if r.get("final_correct", False))

        os_accuracy = os_correct / len(results) * 100
        final_accuracy = final_correct / len(results) * 100

        print(f"OS Model Accuracy: {os_correct}/{len(results)} ({os_accuracy:.2f}%)")
        if use_critic:
            print(f"Final Accuracy (with Critic): {final_correct}/{len(results)} ({final_accuracy:.2f}%)")
            improvement = final_accuracy - os_accuracy
            print(f"Improvement: {improvement:+.2f}%")

        # Save final results
        results_file = os.path.join(output_dir, "videomme_results.json")
        summary_file = os.path.join(output_dir, "videomme_summary.json")

        summary = {
            "dataset_subset": dataset_subset,
            "total_questions": len(results),
            "os_correct": os_correct,
            "os_accuracy": os_accuracy,
            "final_correct": final_correct,
            "final_accuracy": final_accuracy,
            "llm_model": llm_model,
            "vlm_model": vlm_model,
            "use_critic": use_critic,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run VideoMME evaluation using long-context video understanding pipeline"
    )
    parser.add_argument(
        "--subset",
        default="validation",
        help="Dataset subset to use (default: validation)"
    )
    parser.add_argument(
        "--output_dir",
        default="./videomme_results",
        help="Directory to store results (default: ./videomme_results)"
    )
    parser.add_argument(
        "--llm_model",
        default="deepseek-ai/DeepSeek-V3.1",
        help="LLM model name (default: deepseek-ai/DeepSeek-V3.1)"
    )
    parser.add_argument(
        "--vlm_model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="VLM model name (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all)"
    )
    parser.add_argument(
        "--max_questions_per_video",
        type=int,
        default=None,
        help="Maximum questions per video (default: all)"
    )
    parser.add_argument(
        "--no_critic",
        action="store_true",
        help="Disable critic module (only use OS model)"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess videos even if already processed"
    )

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            run_videomme_evaluation(
                dataset_subset=args.subset,
                output_dir=args.output_dir,
                llm_model=args.llm_model,
                vlm_model=args.vlm_model,
                max_videos=args.max_videos,
                max_questions_per_video=args.max_questions_per_video,
                use_critic=not args.no_critic,
                skip_existing=not args.reprocess,
            )
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
