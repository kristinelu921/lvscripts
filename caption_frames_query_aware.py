#!/usr/bin/env python3
"""
Unified caption generation for video frames - supports both query-aware and standard modes.

Query-aware mode: Captions frames with knowledge of all questions for the video,
                  allowing the model to focus on aspects relevant to the questions.
Standard mode: Captions frames without question context using a generic prompt.

Output files:
  - Query-aware: frame_captions_query_aware.json
  - Standard: frame_captions.json
  - Both modes produce: frame_captions_sorted.json (after sorting)

caption-frames_query_aware /path/to/videos --run-all --no-query-aware --use-subtitles
"""
import os
import json
import asyncio
import re
import time
from pathlib import Path
from together import AsyncTogether
from prompts import CES_log_prompt, global_summary_prompt
import base64

with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key = env_data["together_key"]
    os.environ['TOGETHER_API_KEY'] = together_key

async_client_together = AsyncTogether(api_key=together_key)

def create_query_aware_prompt(questions):
    """Create a prompt that includes all questions as context

    Args: 
        questions: List of question dicts with 'question' field

    Returns:
        Prompt string with questions embedded
    """
    if not questions:
        # Fallback to standard prompt if no questions
        return """Please write ONE DESCRIPTIVE sentence that includes [subjects], [actions], [location/scene] if possible/relevant. Make sure to include detailed descriptions of subjects, actions, OBJECTS, large text, and the location/scene."""

    questions_text = "\n".join([f"  - {q['question']}" for q in questions])

    prompt = f"""You will be captioning frames from a video that will be used to answer these questions:

{questions_text}

Please caption this frame with ONE DESCRIPTIVE sentence, focusing on details that might help answer these questions. Include [subjects], [actions], [location/scene], OBJECTS, visible text, and any other relevant details. Pay special attention to elements mentioned in the questions above."""

    return prompt

async def process_single_frame_query_aware(frame_path, prompt, semaphore, results, output_file,
                                           file_lock, frame_num, total_frames,
                                           model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                                           use_subtitles=False, subtitle_frames=None, model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Process a single frame with query-aware captioning"""

    async with semaphore:
        print(f"Processing {frame_path} ({frame_num}/{total_frames})")
        try:
            # Read image and convert to base64
            with open(frame_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Use Together's async API
            response = await async_client_together.chat.completions.create(
                model=model_vlm,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                max_tokens=512
            )

            # Extract text from response
            response_text = response.choices[0].message.content

            # Append subtitle if available and requested
            if use_subtitles and subtitle_frames:
                frame_match = re.search(r'frame_(\d+)\.jpg', frame_path)
                if frame_match:
                    frame_num_val = int(frame_match.group(1))
                    if frame_num_val in subtitle_frames:
                        response_text += f" | Subtitle: {subtitle_frames[frame_num_val]}"

            result_entry = frame_path.split(".jpg")[0][-17:] + " seconds: " + response_text

            async with file_lock:
                results.append(result_entry)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

            print(f"Completed {frame_path}")
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            return None

async def caption_video_query_aware(video_id, frames_dir, output_file, questions_path,
                                   max_concurrent=20, use_subtitles=False, subtitle_frames=None,
                                   query_aware=True, model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Caption all frames for a video with optional query-aware prompting

    Args:
        video_id: Video identifier
        frames_dir: Directory containing frame images
        output_file: Output file for captions
        questions_path: Path to questions JSON
        max_concurrent: Max concurrent API calls
        use_subtitles: Whether to append subtitles
        subtitle_frames: Dict mapping frame numbers to subtitle text
        query_aware: If True, use questions to guide captioning. If False, use standard captioning.
    """
    # Load questions for this video if query-aware mode
    questions = []
    print(f"Using standard (non-query-aware) captioning for video {video_id}")

    # Create prompt (query-aware or standard)
    prompt = create_query_aware_prompt(questions)

    # Get frame files
    frames_path = Path(frames_dir)
    frame_files = sorted([str(f) for f in frames_path.glob("*.jpg")])

    print(f"Processing {len(frame_files)} frames with query-aware captioning...")

    # Load existing results if file exists
    processed_frames = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                for entry in existing_results:
                    if entry.startswith("frames/"):
                        name = entry.split(" seconds:")[0]
                        processed_frames.add(name)
                results = existing_results
        except Exception as e:
            print(f"Warning: Could not load {output_file}: {e}")
            results = []
    else:
        with open(output_file, 'w') as f:
            json.dump([], f)

    frames_to_process = [frame for frame in frame_files
                        if frame.split(".")[0][-17:] not in processed_frames]

    if not frames_to_process:
        print(f"All frames already processed for {video_id}")
        return results

    print(f"Processing {len(frames_to_process)} new frames...")

    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()

    tasks = [
        process_single_frame_query_aware(
            frame_path, prompt, semaphore, results, output_file, file_lock,
            i+1, len(frames_to_process), use_subtitles=use_subtitles,
            subtitle_frames=subtitle_frames,
            model_vlm=model_vlm,
        )
        for i, frame_path in enumerate(frames_to_process)
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    return results

async def process_all_videos_query_aware(vid_folder, questions_path='/mnt/ssd/data/longvideobench/downloaded_videos_questions.json',
                                        use_subtitles=False, subtitle_mapping_path='/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json',
                                        query_aware=True, model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Process all videos with optional query-aware captioning

    Args:
        vid_folder: Folder containing video directories (named by video_id)
        questions_path: Path to questions JSON
        use_subtitles: Whether to use subtitles
        subtitle_mapping_path: Path to subtitle frame mapping
        query_aware: If True, use questions to guide captioning. If False, use standard captioning.
    """
    # Determine which videos to process
    if query_aware and os.path.exists(questions_path):
        # Load all questions to know which videos to process
        with open(questions_path, 'r') as f:
            all_questions = json.load(f)
        video_ids = list(all_questions.keys())
        print(f"Found {len(video_ids)} videos with questions")
    else:
        # Process all videos in the folder
        video_ids = [d for d in os.listdir(vid_folder)
                    if os.path.isdir(os.path.join(vid_folder, d))]
        print(f"Found {len(video_ids)} videos to process")

    # Load subtitle mappings if requested
    subtitle_data = {}
    if use_subtitles and os.path.exists(subtitle_mapping_path):
        with open(subtitle_mapping_path, 'r') as f:
            subtitle_data = json.load(f)
        print(f"Loaded subtitle mappings for {len(subtitle_data)} videos")

    # Process each video
    for video_id in video_ids:
        video_dir = os.path.join(vid_folder, video_id)
        frames_dir = os.path.join(video_dir, 'frames')

        if not os.path.exists(frames_dir):
            print(f"Skipping {video_id}: frames directory not found at {frames_dir}")
            continue

        os.makedirs(os.path.join(video_dir, 'captions'), exist_ok=True)

        # Choose output filename based on mode
        if query_aware:
            output_file = os.path.join(video_dir, 'captions', 'frame_captions_query_aware.json')
        else:
            output_file = os.path.join(video_dir, 'captions', 'frame_captions.json')

        # Get subtitle frames for this video if available
        subtitle_frames = None
        if use_subtitles:
            subtitle_frames = subtitle_data.get(video_id, {}).get('frames', {})
            if subtitle_frames:
                subtitle_frames = {int(k): v for k, v in subtitle_frames.items()}

        print(f"\n{'='*60}")
        print(f"Processing video: {video_id}")
        print(f"{'='*60}")

        await caption_video_query_aware(
            video_id=video_id,
            frames_dir=frames_dir,
            output_file=output_file,
            questions_path=questions_path,
            max_concurrent=10,
            use_subtitles=use_subtitles,
            subtitle_frames=subtitle_frames,
            query_aware=query_aware,
            model_vlm=model_vlm,
            model_llm=model_llm
        )

        print(f"Completed {video_id}")

def sort_captions_query_aware(vid_folder, query_aware=True):
    """Sort captions by frame number

    Args:
        vid_folder: Directory containing video folders
        query_aware: If True, look for frame_captions_query_aware.json. If False, look for frame_captions.json
    """
    video_dirs = [d for d in os.listdir(vid_folder)
                 if os.path.isdir(os.path.join(vid_folder, d))]

    for video_id in video_dirs:
        captions_dir = os.path.join(vid_folder, video_id, 'captions')

        # Choose input filename based on mode
        if query_aware:
            input_file = os.path.join(captions_dir, 'frame_captions_query_aware.json')
        else:
            input_file = os.path.join(captions_dir, 'frame_captions.json')

        output_file = os.path.join(captions_dir, 'frame_captions_sorted.json')

        if not os.path.exists(input_file):
            print(f"Skipping {video_id}: no captions found at {input_file}")
            continue

        os.makedirs(captions_dir, exist_ok=True)

        with open(input_file, "r") as f:
            print(f"Loading captions for {video_id}...")
            captions = json.load(f)
            sorted_captions = sorted(captions, key=lambda x: x.split(" seconds:")[0])
            print(f"Sorting captions for {video_id}...")
            with open(output_file, "w") as out_f:
                json.dump(sorted_captions, out_f, indent=2)
            print(f"Saved sorted captions for {video_id}")

async def create_logs_query_aware(captions_file, output_file, prompt_fct, frames_dir,
                                  model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                                  max_attempts=3):
    """Generate CES logs or global summary for query-aware captions

    Args:
        captions_file: Path to sorted captions JSON
        output_file: Path to output file
        prompt_fct: Prompt function (CES_log_prompt or global_summary_prompt)
        frames_dir: Path to frames directory
        model: Model to use for generation
        max_attempts: Maximum retry attempts
    """
    if os.path.exists(output_file):
        print(f"Already done: {output_file}")
        return

    if not os.path.exists(captions_file):
        print(f"Skipping: captions file not found at {captions_file}")
        return

    with open(captions_file, "r") as f:
        captions_data = json.load(f)

    print(f"Processing {captions_file}...")

    # Generate prompt
    if callable(prompt_fct):
        prompt = prompt_fct(captions_data)
    else:
        prompt = prompt_fct

    # Make API call
    for attempt in range(max_attempts):
        try:
            response = await async_client_together.chat.completions.create(
                model=model_vlm,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }],
                stream=False,
                max_tokens=1000000
            )

            # Extract content
            content = response.choices[0].message.content if response.choices else "No content found"

            # Remove <think> tags if present
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()

            # Format output based on output file type
            if "CES" in output_file or "logs" in output_file:
                formatted_output = "="*80 + "\n"
                formatted_output += "CHARACTER, EVENT, AND SCENE LOGS\n"
                formatted_output += "="*80 + "\n\n"
                formatted_output += content
                formatted_output += "\n\n" + "="*80 + "\n"
                formatted_output += f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                formatted_output += "="*80 + "\n"
            else:
                # Global summary
                formatted_output = content

            with open(output_file, "w") as f:
                f.write(formatted_output)

            print(f"Completed: {output_file}")
            return

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
            if attempt == max_attempts - 1:
                with open(output_file, "w") as f:
                    f.write(f"Failed to generate after {max_attempts} attempts: {e}")
            continue

async def generate_ces_logs_query_aware(vid_folder, model_vlm='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'):
    """Generate CES logs for all videos with query-aware captions

    Args:
        vid_folder: Directory containing video folders
    """
    video_dirs = [d for d in os.listdir(vid_folder)
                 if os.path.isdir(os.path.join(vid_folder, d))]

    print(f"Generating CES logs for {len(video_dirs)} videos...")

    tasks = []
    for video_id in video_dirs:
        captions_file = os.path.join(vid_folder, video_id, 'captions', 'frame_captions_sorted.json')
        output_file = os.path.join(vid_folder, video_id, 'captions', 'CES_logs.txt')
        frames_dir = os.path.join(vid_folder, video_id, 'frames')

        tasks.append(create_logs_query_aware(
            captions_file, output_file, CES_log_prompt, frames_dir, model_vlm=model_vlm
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report failures
    for video_id, result in zip(video_dirs, results):
        if isinstance(result, Exception):
            print(f"Failed to generate CES logs for {video_id}: {result}")

async def generate_global_summaries_query_aware(vid_folder, model_vlm='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'):
    """Generate global summaries for all videos with query-aware captions

    Args:
        vid_folder: Directory containing video folders
    """
    video_dirs = [d for d in os.listdir(vid_folder)
                 if os.path.isdir(os.path.join(vid_folder, d))]

    print(f"Generating global summaries for {len(video_dirs)} videos...")

    tasks = []
    for video_id in video_dirs:
        captions_file = os.path.join(vid_folder, video_id, 'captions', 'frame_captions_sorted.json')
        output_file = os.path.join(vid_folder, video_id, 'captions', 'global_summary.txt')
        frames_dir = os.path.join(vid_folder, video_id, 'frames')

        tasks.append(create_logs_query_aware(
            captions_file, output_file, global_summary_prompt, frames_dir, model_vlm=model_vlm
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report failures
    for video_id, result in zip(video_dirs, results):
        if isinstance(result, Exception):
            print(f"Failed to generate global summary for {video_id}: {result}")

async def run_all_query_aware_captions(vid_folder, questions_path='/mnt/ssd/data/longvideobench/downloaded_videos_questions.json',
                                      use_subtitles=False, subtitle_mapping_path='/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json',
                                      query_aware=True, model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Run complete caption pipeline including CES logs and global summaries

    Args:
        vid_folder: Directory containing video folders
        questions_path: Path to questions JSON
        use_subtitles: Whether to integrate subtitles
        subtitle_mapping_path: Path to subtitle mappings
        query_aware: If True, use questions to guide captioning. If False, use standard captioning.
    """
    mode_name = "QUERY-AWARE" if query_aware else "STANDARD"
    print("\n" + "="*80)
    print(f"{mode_name} CAPTION PIPELINE")
    print("="*80 + "\n")

    # Step 1: Generate captions
    print(f"Step 1/5: Generating {mode_name.lower()} captions...")
    await process_all_videos_query_aware(vid_folder, questions_path, use_subtitles, subtitle_mapping_path, query_aware, model_vlm=model_vlm, model_llm=model_llm)
    print(f"✓ {mode_name} captions generated\n")

    # Step 2: Sort captions
    print("Step 2/5: Sorting captions...")
    sort_captions_query_aware(vid_folder, query_aware)
    print("✓ Captions sorted\n")

    # Step 3: Embed captions
    print("Step 3/5: Embedding captions...")
    try:
        from embed_frame_captions import embed_many
        await embed_many(vid_folder)
        print("✓ Captions embedded\n")
    except Exception as e:
        print(f"⚠ Warning: Could not embed captions: {e}\n")

    # Step 4: Generate global summaries
    print("Step 4/5: Generating global summaries...")
    await generate_global_summaries_query_aware(vid_folder, model_vlm=model_vlm)
    print("✓ Global summaries generated\n")

    # Step 5: Generate CES logs
    print("Step 5/5: Generating CES logs...")
    await generate_ces_logs_query_aware(vid_folder, model_vlm=model_vlm)
    print("✓ CES logs generated\n")

    print("="*80)
    print(f"{mode_name} CAPTION PIPELINE COMPLETE")
    print("="*80)

async def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="caption_frames_query_aware.py",
        description="Caption frames with optional query-aware prompting. Supports both query-aware and standard captioning modes."
    )

    parser.add_argument('vid_folder', help='Directory containing video folders (named by video_id)')
    parser.add_argument('--questions', default='/mnt/ssd/data/longvideobench/downloaded_videos_questions.json',
                       help='Path to questions JSON file')
    parser.add_argument('--use-subtitles', action='store_true',
                       help='Append subtitles to frame captions')
    parser.add_argument('--model-vlm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='VLM model to use')
    parser.add_argument('--model-llm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='LLM model to use')
    parser.add_argument('--query-aware', dest='query_aware', action='store_true', default=True,
                       help='Use query-aware captioning (default)')
    parser.add_argument('--no-query-aware', dest='query_aware', action='store_false',
                       help='Use standard (non-query-aware) captioning')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete pipeline: captions + sort + embed + CES logs + global summary')
    parser.add_argument('--ces-only', action='store_true',
                       help='Only generate CES logs (requires sorted captions)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate global summaries (requires sorted captions)')
    parser.add_argument('--sort-only', action='store_true',
                       help='Only sort existing captions')

    args = parser.parse_args()

    if args.run_all:
        # Run complete pipeline
        await run_all_query_aware_captions(
            vid_folder=args.vid_folder,
            questions_path=args.questions,
            use_subtitles=args.use_subtitles,
            query_aware=args.query_aware,
            model_vlm=args.model_vlm,
            model_llm=args.model_llm
        )
    elif args.ces_only:
        # Only generate CES logs
        print("Generating CES logs only...")
        await generate_ces_logs_query_aware(args.vid_folder, model_vlm=args.model_vlm)
        print("CES logs complete")
    elif args.summary_only:
        # Only generate global summaries
        print("Generating global summaries only...")
        await generate_global_summaries_query_aware(args.vid_folder, model_vlm=args.model_vlm)
        print("Global summaries complete")
    elif args.sort_only:
        # Only sort captions
        print("Sorting captions only...")
        sort_captions_query_aware(args.vid_folder, query_aware=args.query_aware)
        print("Sorting complete")
    else:
        # Default: only generate captions
        await process_all_videos_query_aware(
            vid_folder=args.vid_folder,
            questions_path=args.questions,
            use_subtitles=args.use_subtitles,
            query_aware=args.query_aware,
            model_vlm=args.model_vlm,
            model_llm=args.model_llm
        )

        mode_name = "QUERY-AWARE" if args.query_aware else "STANDARD"
        print("\n" + "="*60)
        print(f"{mode_name} CAPTIONS PROCESSED")
        print("="*60)
        print("\nTo run the complete pipeline including CES logs and summaries, use --run-all")

if __name__ == "__main__":
    asyncio.run(main())
