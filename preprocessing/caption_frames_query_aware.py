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
import sys
import json
import asyncio
import re
import time
from pathlib import Path
from together import AsyncTogether
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import CES_log_prompt, global_summary_prompt

with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key = env_data["together_key"]
    kimi_api_key = env_data["kimi_api_key"]
    os.environ['TOGETHER_API_KEY'] = together_key

async_client_together = AsyncTogether(api_key=together_key)

# Kimi API client setup
import aiohttp

async def call_kimi_api(messages, model="kimi-k2.5", temperature=1.0):
    """
    Call Kimi API with messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use (default: kimi-k2.5)
        temperature: Temperature for generation (must be 1.0 for kimi-k2.5)

    Returns:
        Response text
    """
    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kimi_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                error_text = await response.text()
                raise Exception(f"Kimi API error {response.status}: {error_text}")

def create_prompt():
    """Create a standard frame captioning prompt

    Returns:
        Prompt string for frame captioning
    """
    return """Describe this frame in ONE detailed sentence. Include:
    - Count and identify ALL subjects (people, animals, objects) with
    specific attributes (colors, clothing, brands, types)
    - Describe ALL actions and what is happening
    - List ALL visible text (signs, labels, numbers, brand names, on-screen
    text, captions)
    - Specify spatial relationships and positioning (left/right,
    foreground/background, next to, holding)
    - Describe the setting/location/environment
    - Note key visual attributes (colors, sizes, states, emotions,
    expressions)

    Be specific and concrete. Examples:
    - Not "a person" but "a woman in a red Nike jersey"
    - Not "some animals" but "three golden retrievers"
    - Not "text visible" but "sign reads 'EXIT' in white letters"
    - Not "on a table" but "silver iPhone on a wooden table next to a blue
    coffee mug"
    """


def create_clip_prompt_characters():
    """Create prompt for character-focused clip captioning"""
    return """Analyze this video clip and describe the CHARACTERS and their ACTIONS in detail.

Focus on:
- ALL characters present (people, animals)
- Physical actions and movements of each character
- Talking, speaking, or vocalizations
- Facial expressions and emotions
- Body language and gestures
- Interactions between characters
- Any motion or movement patterns

Be specific and detailed. Describe WHO is doing WHAT throughout the clip."""


def create_clip_prompt_objects():
    """Create prompt for object-focused clip captioning"""
    return """Analyze this video clip and list ALL OBJECTS visible in detail.

Provide a comprehensive LIST of objects with their attributes:
- Items and their names
- Colors of each object
- Locations and positions
- Sizes (large, small, medium)
- Textures and materials
- Any visible TEXT (signs, labels, words)
- Brands or logos

Format as a structured list with descriptors for each object."""


def create_clip_prompt_scene():
    """Create prompt for scene/setting/mood-focused clip captioning"""
    return """Analyze this video clip and provide a STRUCTURED description of the location, scene, spatial layout, and mood.

Use this format:

**LOCATION:**
- Type: [indoor/outdoor, specific location like kitchen, park, office, etc.]
- Setting: [brief description of the environment]

**SPATIAL LAYOUT:**
- Foreground: [objects and elements in the front]
- Midground: [objects and elements in the middle]
- Background: [objects and elements in the back]
- Left side: [notable objects on the left]
- Right side: [notable objects on the right]
- Center: [what occupies the central area]

**VISUAL CONDITIONS:**
- Lighting: [type, quality, direction, color temperature]
- Time of day: [if determinable]
- Weather/Atmosphere: [if applicable]

**MOOD & ATMOSPHERE:**
- Overall tone: [emotional quality of the scene]
- Visual style: [composition, framing, aesthetic]

Provide specific, concrete details about the spatial arrangement of objects and the scene's visual organization."""                   
async def process_single_clip(clip_info, clips_dir, semaphore, results, output_file,
                                          file_lock, clip_num, total_clips):
    """Process a single clip with three types of captions using Kimi API"""

    async with semaphore:
        clip_filename = clip_info['filename']
        clip_path = os.path.join(clips_dir, clip_filename)

        print(f"Processing {clip_filename} ({clip_num}/{total_clips})")

        try:
            # Read video clip and convert to base64
            with open(clip_path, "rb") as clip_file:
                clip_bytes = clip_file.read()
                clip_base64 = base64.b64encode(clip_bytes).decode('utf-8')

            # Get three types of captions
            captions = {}

            # 1. Character/Action caption
            print(f"  Getting character/action caption...")
            char_messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{clip_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": create_clip_prompt_characters()
                    }
                ]
            }]
            captions['characters_actions'] = await call_kimi_api(char_messages)

            # 2. Objects caption
            print(f"  Getting objects caption...")
            obj_messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{clip_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": create_clip_prompt_objects()
                    }
                ]
            }]
            captions['objects'] = await call_kimi_api(obj_messages)

            # 3. Scene/Setting/Mood caption
            print(f"  Getting scene/setting/mood caption...")
            scene_messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{clip_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": create_clip_prompt_scene()
                    }
                ]
            }]
            captions['scene_setting_mood'] = await call_kimi_api(scene_messages)

            # Create result entry
            result_entry = {
                'clip_filename': clip_filename,
                'start': clip_info['start'],
                'end': clip_info['end'],
                'duration': clip_info['duration'],
                'captions': captions
            }

            async with file_lock:
                results.append(result_entry)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

            print(f"Completed {clip_filename}")
        except Exception as e:
            print(f"Error processing {clip_filename}: {e}")
            return None


async def process_single_frame(frame_path, prompt, semaphore, results, output_file,
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

async def caption_clips(video_id, clips_dir, output_file, max_concurrent=5):
    """Caption all clips for a video using Kimi API with three caption types

    Args:
        video_id: Video identifier
        clips_dir: Directory containing clip videos
        output_file: Output file for clip captions
        max_concurrent: Max concurrent API calls (default 5 for video processing)
    """
    print(f"Captioning clips for video {video_id}")

    # Load clips metadata
    metadata_file = os.path.join(clips_dir, 'clips_metadata.json')
    if not os.path.exists(metadata_file):
        print(f"  No clips metadata found at {metadata_file}")
        return []

    with open(metadata_file, 'r') as f:
        clips_info = json.load(f)

    print(f"Processing {len(clips_info)} clips...")

    # Load existing results if file exists
    results = []
    processed_clips = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                for entry in existing_results:
                    processed_clips.add(entry['clip_filename'])
                results = existing_results
        except Exception as e:
            print(f"Warning: Could not load {output_file}: {e}")
            results = []
    else:
        with open(output_file, 'w') as f:
            json.dump([], f)

    clips_to_process = [clip for clip in clips_info
                       if clip['filename'] not in processed_clips]

    if not clips_to_process:
        print(f"All clips already processed for {video_id}")
        return results

    print(f"Processing {len(clips_to_process)} new clips...")

    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()

    tasks = [
        process_single_clip(
            clip_info, clips_dir, semaphore, results, output_file, file_lock,
            i+1, len(clips_to_process)
        )
        for i, clip_info in enumerate(clips_to_process)
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    return results


async def caption_video(video_id, frames_dir, output_file,
                                   max_concurrent=20, use_subtitles=False, subtitle_frames=None,
                                   model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Caption all frames for a video

    Args:
        video_id: Video identifier
        frames_dir: Directory containing frame images
        output_file: Output file for captions
        max_concurrent: Max concurrent API calls
        use_subtitles: Whether to append subtitles
        subtitle_frames: Dict mapping frame numbers to subtitle text
        model_vlm: Vision-language model to use
        model_llm: Language model to use
    """
    print(f"Using standard captioning for video {video_id}")

    # Create prompt
    prompt = create_prompt()

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
        process_single_frame(
            frame_path, prompt, semaphore, results, output_file, file_lock,
            i+1, len(frames_to_process), use_subtitles=use_subtitles,
            subtitle_frames=subtitle_frames,
            model_vlm=model_vlm,
        )
        for i, frame_path in enumerate(frames_to_process)
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    return results

async def process_all_clips(vid_folder, reverse=False):
    """Process all videos' clips with Kimi API captioning

    Args:
        vid_folder: Folder containing video directories (with clips subdirs)
        reverse: If True, process videos in reverse alphabetical order
    """
    video_ids = [d for d in os.listdir(vid_folder)
                if os.path.isdir(os.path.join(vid_folder, d))]

    # Sort and optionally reverse
    video_ids = sorted(video_ids, reverse=reverse)

    mode_indicator = "[REVERSE]" if reverse else "[FORWARD]"
    print(f"{mode_indicator} Found {len(video_ids)} videos to process clips for")

    # Process each video's clips
    for video_id in video_ids:
        video_dir = os.path.join(vid_folder, video_id)
        clips_dir = os.path.join(video_dir, 'clips')

        if not os.path.exists(clips_dir):
            print(f"Skipping {video_id}: clips directory not found at {clips_dir}")
            continue

        os.makedirs(os.path.join(video_dir, 'captions'), exist_ok=True)
        output_file = os.path.join(video_dir, 'captions', 'clip_captions.json')

        print(f"\n{'='*60}")
        print(f"Processing clips for video: {video_id}")
        print(f"{'='*60}")

        await caption_clips(
            video_id=video_id,
            clips_dir=clips_dir,
            output_file=output_file,
            max_concurrent=3  # Lower concurrency for video API calls
        )

        print(f"Completed {video_id}")


async def process_all_videos(vid_folder,
                                        use_subtitles=False, subtitle_mapping_path='/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json',
                                        model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", reverse=False):
    """Process all videos with standard captioning

    Args:
        vid_folder: Folder containing video directories (named by video_id)
        use_subtitles: Whether to use subtitles
        subtitle_mapping_path: Path to subtitle frame mapping
        model_vlm: Vision-language model to use
        model_llm: Language model to use
        reverse: If True, process videos in reverse alphabetical order
    """

    video_ids = [d for d in os.listdir(vid_folder)
                if os.path.isdir(os.path.join(vid_folder, d))]

    # Sort and optionally reverse
    video_ids = sorted(video_ids, reverse=reverse)

    mode_indicator = "[REVERSE]" if reverse else "[FORWARD]"
    print(f"{mode_indicator} Found {len(video_ids)} videos to process")
    if reverse:
        print(f"{mode_indicator} Processing in REVERSE order: {video_ids[0]} -> {video_ids[-1]}")

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

        # Use standard caption output file
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

        await caption_video(
            video_id=video_id,
            frames_dir=frames_dir,
            output_file=output_file,
            max_concurrent=10,
            use_subtitles=use_subtitles,
            subtitle_frames=subtitle_frames,
            model_vlm=model_vlm,
            model_llm=model_llm
        )

        print(f"Completed {video_id}")

def sort_captions(vid_folder):
    """Sort captions by frame number

    Args:
        vid_folder: Directory containing video folders
    """
    video_dirs = [d for d in os.listdir(vid_folder)
                 if os.path.isdir(os.path.join(vid_folder, d))]

    for video_id in video_dirs:
        captions_dir = os.path.join(vid_folder, video_id, 'captions')

        # Use standard caption input file
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

async def create_logs(captions_file, output_file, prompt_fct, frames_dir,
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

async def generate_ces_logs(vid_folder, model_vlm='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'):
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

        tasks.append(create_logs(
            captions_file, output_file, CES_log_prompt, frames_dir, model_vlm=model_vlm
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report failures
    for video_id, result in zip(video_dirs, results):
        if isinstance(result, Exception):
            print(f"Failed to generate CES logs for {video_id}: {result}")

async def generate_global_summaries(vid_folder, model_vlm='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'):
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

        tasks.append(create_logs(
            captions_file, output_file, global_summary_prompt, frames_dir, model_vlm=model_vlm
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report failures
    for video_id, result in zip(video_dirs, results):
        if isinstance(result, Exception):
            print(f"Failed to generate global summary for {video_id}: {result}")

async def generate_clip_summary_for_video(vid_folder, video_id):
    """Generate CES logs and global summary from clip captions for a single video using Kimi API

    Args:
        vid_folder: Directory containing video folders
        video_id: Video ID
    """
    clip_captions_file = os.path.join(vid_folder, video_id, 'captions', 'clip_captions.json')

    if not os.path.exists(clip_captions_file):
        return

    # Load clip captions
    with open(clip_captions_file) as f:
        clip_data = json.load(f)

    # Format captions for prompts
    captions_text = []
    for clip in clip_data:
        clip_name = clip.get('clip_filename', 'unknown')
        start_time = clip.get('start', 0)
        end_time = clip.get('end', 0)

        captions_text.append(f"\n[Clip: {clip_name}, Time: {start_time:.1f}s - {end_time:.1f}s]")

        for caption_type, caption_content in clip['captions'].items():
            captions_text.append(f"\n{caption_type.upper()}:")
            captions_text.append(caption_content[:1000])  # Truncate if too long

    captions_data = "\n".join(captions_text)

    # Generate CES logs
    ces_output = os.path.join(vid_folder, video_id, 'captions', 'CES_logs.txt')
    if not os.path.exists(ces_output):
        print(f"  Generating CES logs for {video_id}...")
        ces_prompt = CES_log_prompt(captions_data)
        ces_messages = [{"role": "user", "content": ces_prompt}]
        ces_response = await call_kimi_api(ces_messages, model="kimi-k2.5")

        formatted_output = "="*80 + "\n"
        formatted_output += "CHARACTER, EVENT, AND SCENE LOGS\n"
        formatted_output += "="*80 + "\n\n"
        formatted_output += ces_response
        formatted_output += "\n\n" + "="*80 + "\n"
        formatted_output += f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        formatted_output += "="*80 + "\n"

        with open(ces_output, 'w') as f:
            f.write(formatted_output)
        print(f"    ✓ Saved: {ces_output}")

    # Generate global summary
    summary_output = os.path.join(vid_folder, video_id, 'captions', 'global_summary.txt')
    if not os.path.exists(summary_output):
        print(f"  Generating global summary for {video_id}...")
        summary_prompt = global_summary_prompt(captions_data)
        summary_messages = [{"role": "user", "content": summary_prompt}]
        summary_response = await call_kimi_api(summary_messages, model="kimi-k2.5")

        with open(summary_output, 'w') as f:
            f.write(summary_response)
        print(f"    ✓ Saved: {summary_output}")


async def generate_clip_summaries(vid_folder):
    """Generate CES logs and global summaries from clip captions using Kimi API

    Args:
        vid_folder: Directory containing video folders with clip captions
    """
    video_dirs = [d for d in os.listdir(vid_folder)
                 if os.path.isdir(os.path.join(vid_folder, d))]

    print(f"\n{'='*60}")
    print(f"Generating summaries from clip captions for {len(video_dirs)} videos...")
    print(f"{'='*60}\n")

    for video_id in video_dirs:
        try:
            await generate_clip_summary_for_video(vid_folder, video_id)
        except Exception as e:
            print(f"❌ Failed to generate summaries for {video_id}: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Clip summaries generated")
    print(f"{'='*60}\n")

async def run_all_captions(vid_folder,
                                      use_subtitles=False, subtitle_mapping_path='/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json',
                                      model_vlm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", reverse=False):
    """Run complete caption pipeline including CES logs and global summaries

    Args:
        vid_folder: Directory containing video folders
        use_subtitles: Whether to integrate subtitles
        subtitle_mapping_path: Path to subtitle mappings
        model_vlm: Vision-language model to use
        model_llm: Language model to use
        reverse: If True, process videos in reverse alphabetical order
    """
    print("\n" + "="*80)
    print("CAPTION PIPELINE")
    print("="*80 + "\n")

    # Step 1: Generate captions
    print("Step 1/5: Generating captions...")
    await process_all_videos(vid_folder, use_subtitles, subtitle_mapping_path, model_vlm=model_vlm, model_llm=model_llm, reverse=reverse)
    print("✓ Captions generated\n")

    # Step 2: Sort captions
    print("Step 2/5: Sorting captions...")
    sort_captions(vid_folder)
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
    await generate_global_summaries(vid_folder, model_vlm=model_vlm)
    print("✓ Global summaries generated\n")

    # Step 5: Generate CES logs
    print("Step 5/5: Generating CES logs...")
    await generate_ces_logs(vid_folder, model_vlm=model_vlm)
    print("✓ CES logs generated\n")

    print("="*80)
    print("CAPTION PIPELINE COMPLETE")
    print("="*80)

async def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="caption_frames_query_aware.py",
        description="Caption frames or clips for videos. Generates standard frame captions or clip captions with Kimi API."
    )

    parser.add_argument('vid_folder', help='Directory containing video folders (named by video_id)')
    parser.add_argument('--use-subtitles', action='store_true',
                       help='Append subtitles to frame captions')
    parser.add_argument('--model-vlm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='VLM model to use for frame captions')
    parser.add_argument('--model-llm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='LLM model to use for frame captions')
    parser.add_argument('--clip', action='store_true',
                       help='Caption video clips instead of frames using Kimi API (generates 3 caption types per clip)')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete pipeline: captions + sort + embed + CES logs + global summary')
    parser.add_argument('--ces-only', action='store_true',
                       help='Only generate CES logs (requires sorted captions)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate global summaries (requires sorted captions)')
    parser.add_argument('--sort-only', action='store_true',
                       help='Only sort existing captions')
    parser.add_argument('--reverse', action='store_true',
                       help='Process videos in reverse alphabetical order (useful for parallel processing)')

    args = parser.parse_args()

    if args.clip:
        # Process clips instead of frames
        print("Processing video clips with Kimi API...")
        await process_all_clips(args.vid_folder, reverse=args.reverse)
        print("\n" + "="*60)
        print("CLIP CAPTIONS PROCESSED")
        print("="*60)

        # Auto-generate CES logs and global summaries from clip captions
        print("\nGenerating CES logs and global summaries from clip captions...")
        await generate_clip_summaries(args.vid_folder)
        print("✅ Complete! Clips captioned and summaries generated.")
    elif args.run_all:
        # Run complete pipeline
        await run_all_captions(
            vid_folder=args.vid_folder,
            use_subtitles=args.use_subtitles,
            model_vlm=args.model_vlm,
            model_llm=args.model_llm,
            reverse=args.reverse
        )
    elif args.ces_only:
        # Only generate CES logs
        print("Generating CES logs only...")
        await generate_ces_logs(args.vid_folder, model_vlm=args.model_vlm)
        print("CES logs complete")
    elif args.summary_only:
        # Only generate global summaries
        print("Generating global summaries only...")
        await generate_global_summaries(args.vid_folder, model_vlm=args.model_vlm)
        print("Global summaries complete")
    elif args.sort_only:
        # Only sort captions
        print("Sorting captions only...")
        sort_captions(args.vid_folder)
        print("Sorting complete")
    else:
        # Default: only generate captions
        await process_all_videos(
            vid_folder=args.vid_folder,
            use_subtitles=args.use_subtitles,
            model_vlm=args.model_vlm,
            model_llm=args.model_llm,
            reverse=args.reverse
        )

        print("\n" + "="*60)
        print("CAPTIONS PROCESSED")
        print("="*60)
        print("\nTo run the complete pipeline including CES logs and summaries, use --run-all")

if __name__ == "__main__":
    asyncio.run(main())
