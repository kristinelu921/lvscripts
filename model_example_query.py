#!/usr/bin/env python3

import base64
import time
import os
import requests
from together import Together, AsyncTogether
#from google.genai import Client
# PIL is optional for this environment. Keep import guarded to avoid hard
# dependency failures when image manipulation utilities are not used.
try:
    from PIL import Image  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    Image = None
#from google.genai import types
import json
import subprocess
import asyncio
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
#from token_tracker import record, num_tokens


def _normalize_model_name(model_name: str) -> str:
    """Return fallback-friendly model name for current Together availability."""
    if not isinstance(model_name, str):
        return model_name

    normalized = model_name.strip()
    if normalized == "moonshotai/Kimi-K2.5":
        return normalized

    # Older/alternate Kimi K2.5 endpoints often include provider prefixes and hashes.
    # Together frequently exposes the same model under `moonshotai/Kimi-K2.5`.
    segments = normalized.split("/")
    model_root = segments[-1] if segments else normalized
    if model_root.startswith("Kimi-K2.5-") and any("moonshotai" in seg for seg in segments):
        return "moonshotai/Kimi-K2.5"
    if "Kimi-K2.5" in normalized and normalized.endswith("-9b8c5484"):
        return "moonshotai/Kimi-K2.5"

    return normalized


LLM_QUERY_TIMEOUT_SECONDS = int(os.environ.get("TOGETHER_REQUEST_TIMEOUT_SECONDS", "180"))
VLM_QUERY_TIMEOUT_SECONDS = int(os.environ.get("TOGETHER_REQUEST_TIMEOUT_SECONDS", "180"))

# Initialize client with API key
with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key_PRIV = env_data["together_key"]
    kimi_api_key = env_data.get("kimi_api_key")


os.environ['TOGETHER_API_KEY'] = together_key_PRIV
#os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
client_together = Together()
# Don't create global async client - create per request instead
#genai.configure()

# Kimi API support (optional: only needed for non-Together Kimi endpoints)
try:
    import aiohttp
except Exception:  # pragma: no cover - environment dependent
    aiohttp = None

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

    if aiohttp is None:
        raise RuntimeError("aiohttp is required for KIMI API calls")

    timeout = aiohttp.ClientTimeout(total=LLM_QUERY_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                message = result['choices'][0].get('message', {})
                return _extract_message_text(message)
            else:
                error_text = await response.text()
                raise Exception(f"Kimi API error {response.status}: {error_text}")


def _video_clip_output_path(input_file, start_second, end_second):
    base = os.path.basename(input_file)
    name = os.path.splitext(base)[0]
    output_dir = os.path.dirname(input_file)
    return os.path.join(output_dir, f"{name}.{int(start_second)}_{int(end_second)}.query_clip.mp4")


def _format_base64_video_file(video_path):
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode('utf-8')
    return f"data:video/mp4;base64,{video_b64}"


def trim_video_for_kimi(input_file, start_second, end_second, output_file=None):
    """Create a short clip file with fallback to re-encode on keyframe-only trim failures."""
    if output_file is None:
        output_file = _video_clip_output_path(input_file, start_second, end_second)

    if start_second < 0 or end_second < 0:
        raise ValueError("start_second and end_second must be non-negative")
    if end_second <= start_second:
        raise ValueError("end_second must be greater than start_second")

    duration = end_second - start_second

    # First try stream copy (fast)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_second),
        '-i', input_file,
        '-t', str(duration),
        '-c', 'copy',
        '-movflags', '+faststart',
        output_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        # Fallback to re-encode for robustness when stream copy fails.
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_second),
            '-i', input_file,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'superfast',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg clip trim failed: {result.stderr[:500]}")

    return output_file


def _extract_message_text(message):
    """Return the first usable text field from an API message payload."""
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning", "")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    if isinstance(reasoning, list):
        pieces = []
        for item in reasoning:
            if isinstance(item, str):
                pieces.append(item)
        if pieces:
            return "".join(pieces)

    return ""


def _query_together_api_sync_messages(model, messages, max_tokens=4096, temperature=0.7):
    """Query Together API via HTTP with explicit timeout and robust message payload support."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    headers = {
        "Authorization": f"Bearer {together_key_PRIV}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": _normalize_model_name(model),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=LLM_QUERY_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise RuntimeError(f"Together API error {response.status_code}: {response.text}")

    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("Together response did not include choices")

    message = choices[0].get("message", {})
    return _extract_message_text(message)


def _query_together_api_sync(model, prompt, max_tokens=4096, temperature=0.7):
    """Backwards-compatible wrapper for string prompts."""
    return _query_together_api_sync_messages(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature)


async def query_vlm_kimi_video(model, video_path, query, temperature=1.0):
    """Query Kimi with a raw video clip payload."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "video_url", "video_url": {"url": _format_base64_video_file(video_path)}}
            ]
        }],
        "temperature": temperature
    }

    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kimi_api_key}",
        "Content-Type": "application/json"
    }

    if aiohttp is None:
        raise RuntimeError("aiohttp is required for KIMI API video calls")

    timeout = aiohttp.ClientTimeout(total=VLM_QUERY_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            error_text = await response.text()
            raise Exception(f"Kimi API error {response.status}: {error_text}")

def log(message, file_title):
    if not os.path.exists(file_title):
        os.makedirs(file_title)
    else:
        with open(f"{file_title}/log.log", "a") as f:
            f.write(message + "\n")

async def query_vlm_kimi(model, image_paths, query, max_retries=20, batch_size=16):
    """Query Kimi VLM about frames with batched images

    Args:
        model: Kimi model name (e.g., kimi-k2.5)
        image_paths: List of image file paths
        query: Text prompt for the VLM
        max_retries: Maximum retry attempts
        batch_size: Initial number of images per VLM call (default 30)
    """
    grouped_response = []
    failed_images = []
    warned_missing_files = set()
    print("="*10 + " Querying Kimi VLM " + "="*10 + f"for {len(image_paths)} images in batches of up to {batch_size}...")

    # Process in batches
    current_batch_size = batch_size
    batch_start = 0

    while batch_start < len(image_paths):
        batch_end = min(batch_start + current_batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]

        print(f"Processing batch {batch_start}-{batch_end} ({len(batch_paths)} images) with batch_size={current_batch_size}")

        for attempt in range(max_retries):
            if attempt > 0:
                wait_time = min(2 ** attempt, 60)
                print(f"Retrying batch after {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            try:
                # Build content array with labeled frames
                content = []

                # Add initial query text
                intro_text = f"{query}\n\nYou are viewing {len(batch_paths)} frames from the video. Each frame is labeled with its position:\n"
                content.append({"type": "text", "text": intro_text})

                # Add each image with label
                valid_images = []
                for idx, image_path in enumerate(batch_paths, 1):
                    try:
                        if not os.path.exists(image_path):
                            if image_path not in warned_missing_files:
                                print(f"Warning: Image file not found: {image_path}")
                                failed_images.append((image_path, "File not found"))
                                warned_missing_files.add(image_path)
                            continue

                        # Read and encode image
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')

                        # Add frame label
                        content.append({"type": "text", "text": f"\n--- Frame {idx} ({image_path}) ---"})
                        # Add image
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                        valid_images.append(image_path)

                    except Exception as e:
                        print(f"Error reading image {image_path}: {e}")
                        failed_images.append((image_path, f"Read error: {e}"))
                        continue

                if not valid_images:
                    print("No valid images in batch, skipping...")
                    batch_start = batch_end
                    break

                print(f"Sending {len(valid_images)} images to Kimi API...")

                # Call Kimi API
                try:
                    messages = [{
                        "role": "user",
                        "content": content
                    }]

                    content_response = await call_kimi_api(messages, model=model, temperature=1.0)

                    print(f"✓ Successfully processed batch with {len(valid_images)} images")
                    print(f"Response preview: {content_response[:100]}...")

                    # Store response with batch info
                    grouped_response.append({
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "image_paths": valid_images,
                        "response": content_response
                    })

                    # Success - move to next batch
                    batch_start = batch_end
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    if "max" in error_msg or "token" in error_msg or "limit" in error_msg:
                        print(f"⚠ Token limit error with batch_size={current_batch_size}. Reducing batch size...")
                        current_batch_size = max(current_batch_size // 2, 1)
                        if current_batch_size < len(batch_paths):
                            print(f"Retrying with smaller batch_size={current_batch_size}")
                            batch_end = min(batch_start + current_batch_size, len(image_paths))
                            batch_paths = image_paths[batch_start:batch_end]
                            continue
                    raise

            except Exception as e:
                print(f"Batch processing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for image_path in batch_paths:
                        if not any(img == image_path for img, _ in failed_images):
                            failed_images.append((image_path, f"Batch error: {e}"))
                    batch_start = batch_end
                    break

    # Report failures
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for img_path, error in failed_images[:5]:
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")

    # Format response
    if grouped_response:
        full_response = "\n\n".join([batch["response"] for batch in grouped_response])
        return full_response
    else:
        return f"Error: Could not process any images successfully. {len(failed_images)} images failed."

async def query_vlm(model, image_paths, query, max_retries=20, batch_size=16):
    """Query VLM about frames with batched images in single API call

    Sends up to 30 images in ONE VLM conversation with labeled frames.
    Implements exponential backoff if max tokens timeout occurs (30 -> 15 -> 7, etc).

    Args:
        model: VLM model name
        image_paths: List of image file paths
        query: Text prompt for the VLM
        max_retries: Maximum retry attempts
        batch_size: Initial number of images per VLM call (default 30)
    """
    normalized_model = _normalize_model_name(model)

    # Check if using direct Moonshot Kimi API (not a Together endpoint)
    # Together endpoints have format: "username/provider/model-id" or contain "/"
    if "kimi" in normalized_model.lower() and "/" not in normalized_model:
        if isinstance(image_paths, (str, os.PathLike)):
            if str(image_paths).lower().endswith(".mp4"):
                return await query_vlm_kimi_video(normalized_model, str(image_paths), query, temperature=1.0)
            if os.path.isdir(image_paths):
                raise ValueError(f"query_vlm received a directory for KIMI query: {image_paths}")
        if isinstance(image_paths, (list, tuple)) and len(image_paths) == 1:
            single_path = image_paths[0]
            if isinstance(single_path, (str, os.PathLike)) and str(single_path).lower().endswith(".mp4"):
                return await query_vlm_kimi_video(normalized_model, str(single_path), query, temperature=1.0)
        return await query_vlm_kimi(normalized_model, image_paths, query, max_retries, batch_size)

    grouped_response = []
    failed_images = []
    warned_missing_files = set()  # Track files we've already warned about
    print("="*10 + " Querying VLM " + "="*10 + f"for {len(image_paths)} images in batches of up to {batch_size}...")

    # Create a new async client for this request to avoid session issues
    async_client = AsyncTogether(api_key=together_key_PRIV)

    # Process in batches to avoid overwhelming the API
    current_batch_size = batch_size
    batch_start = 0

    while batch_start < len(image_paths):
        batch_end = min(batch_start + current_batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]

        print(f"Processing batch {batch_start}-{batch_end} ({len(batch_paths)} images) with batch_size={current_batch_size}")

        for attempt in range(max_retries):
            if attempt > 0:
                # Wait before retry with exponential backoff
                wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                print(f"Retrying batch after {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            try:
                # Build content array with labeled frames
                content = []

                # Add initial query text with frame count
                intro_text = f"{query}\n\nYou are viewing {len(batch_paths)} frames from the video. Each frame is labeled with its position:\n"
                content.append({"type": "text", "text": intro_text})

                # Add each image with label
                valid_images = []
                for idx, image_path in enumerate(batch_paths, 1):
                    try:
                        # Check if file exists
                        if not os.path.exists(image_path):
                            # Only warn once per missing file
                            if image_path not in warned_missing_files:
                                print(f"Warning: Image file not found: {image_path}")
                                failed_images.append((image_path, "File not found"))
                                warned_missing_files.add(image_path)
                            continue

                        # Read and encode image
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')

                        # Add frame label
                        content.append({"type": "text", "text": f"\n--- Frame {idx} ({image_path}) ---"})
                        # Add image
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                        valid_images.append(image_path)

                    except Exception as e:
                        print(f"Error reading image {image_path}: {e}")
                        failed_images.append((image_path, f"Read error: {e}"))
                        continue

                if not valid_images:
                    print("No valid images in batch, skipping...")
                    # Move to next batch even if no valid images
                    batch_start = batch_end
                    break

                print(f"Sending {len(valid_images)} images in ONE API call...")

                # Make single API call with all images
                try:
                    messages = [{
                        "role": "user",
                        "content": content
                    }]
                    response = await asyncio.to_thread(
                        _query_together_api_sync_messages,
                        normalized_model,
                        messages,
                        4096,
                        0.7
                    )

                    content_response = response
                    print(f"✓ Successfully processed batch with {len(valid_images)} images")
                    print(f"Response preview: {content_response[:100]}...")

                    # Store response with batch info
                    grouped_response.append({
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "image_paths": valid_images,
                        "response": content_response
                    })

                    # Success - move to next batch
                    batch_start = batch_end
                    break

                except asyncio.TimeoutError:
                    print(f"⚠ Timeout with batch_size={current_batch_size}. Reducing batch size...")
                    # Exponential backoff: reduce batch size
                    current_batch_size = max(current_batch_size // 2, 1)
                    if current_batch_size < len(batch_paths):
                        print(f"Retrying with smaller batch_size={current_batch_size}")
                        # Re-adjust batch_end with new batch size
                        batch_end = min(batch_start + current_batch_size, len(image_paths))
                        batch_paths = image_paths[batch_start:batch_end]
                        continue
                    else:
                        raise  # Re-raise if we can't reduce further

                except Exception as e:
                    error_msg = str(e).lower()
                    if "max" in error_msg and "token" in error_msg:
                        print(f"⚠ Max tokens error with batch_size={current_batch_size}. Reducing batch size...")
                        # Exponential backoff: reduce batch size
                        current_batch_size = max(current_batch_size // 2, 1)
                        if current_batch_size < len(batch_paths):
                            print(f"Retrying with smaller batch_size={current_batch_size}")
                            batch_end = min(batch_start + current_batch_size, len(image_paths))
                            batch_paths = image_paths[batch_start:batch_end]
                            continue
                    raise  # Re-raise other errors

            except Exception as e:
                print(f"Batch processing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for image_path in batch_paths:
                        if not any(img == image_path for img, _ in failed_images):
                            failed_images.append((image_path, f"Batch error: {e}"))
                    # Move to next batch even on failure
                    batch_start = batch_end
                    break

    # Report failures
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for img_path, error in failed_images[:5]:  # Show first 5
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    else:
        pass

    # Format and condense response if we have any successful responses
    if grouped_response:
        try:
            # Format batched responses with frame info
            formatted_responses = []
            for batch in grouped_response:
                batch_text = f"\n=== Batch {batch['batch_start']}-{batch['batch_end']} ({len(batch['image_paths'])} frames) ===\n"
                batch_text += f"Frames: {', '.join(batch['image_paths'])}\n"
                batch_text += f"VLM Response: {batch['response']}\n"
                formatted_responses.append(batch_text)

            all_responses_text = '\n'.join(formatted_responses)

            # Condense the batched responses
            condensed_response = await condense_vlm_response(all_responses_text)
            if condensed_response:
                print("CONDENSED RESPONSE: ", condensed_response[:50] + "...")
            else:
                print("CONDENSED RESPONSE: None")

            return {
                "batched_responses": grouped_response,
                "formatted_responses": formatted_responses,
                "condensed_response": condensed_response,
                "failed_images": failed_images
            }
        except Exception as e:
            print(f"Error condensing response: {e}")
            return {
                "batched_responses": grouped_response,
                "formatted_responses": None,
                "condensed_response": None,
                "failed_images": failed_images
            }
    else:
        print("No successful VLM responses")
        return None

async def condense_vlm_response(response):
    """Use an LLM to condense the response into a more cohesive summary of a scene"""
    # Create a new async client for this request
    async_client = AsyncTogether(api_key=together_key_PRIV)
    try:
        result = await async_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "Please condense the following VLM response with the timestamps into a more cohesive summary of a scene with character tracking, according to the question. The response is: " + str(response)}]
            }],
            stream=False
        )
        return result.choices[0].message.content
    except Exception as e:
        print(f"Error condensing VLM response: {e}")
        return None

#THEN WE CAN QUERY GEMINI ABOUT THE CLIP.
def query_gemini_about_clip(start_time, end_time, query):
    """Query Gemini about a clip"""
    try:
        clip_path = trim_with_subprocess("video.mp4", start_time, end_time)
        if os.path.getsize(clip_path) < 20000000:
            video_bytes = open(clip_path, "rb").read()
        else:
            assert False, "Video is too large"
        
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(
                parts=[
                    types.Part(
                        inline_data = types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                    types.Part(
                        text = query
                    )
                ]
            )])
        
        return response.text
        

    except Exception as e:
        print(f"Error querying gemini about video clip: {e}")
        return None


def trim_with_subprocess(input_file, start_time, end_time):
    """ USE GOOD NAMING PRACTICES. """
    output_file = f"{input_file.split('/')[-1]}.{start_time}-{end_time}.mp4"
    cmd = [
        'ffmpeg', '-i', input_file, 
        '-ss', str(start_time), 
        '-t', str(end_time - start_time),
        '-c', 'copy', output_file, '-y'
    ]
    subprocess.run(cmd)
    return output_file

def query_llm(model, prompt, max_tokens=4096, temperature=0.7):
    """
    Query any open-source model w/ Together AI
    """
    try:
        text = _query_together_api_sync(model, prompt, max_tokens=max_tokens, temperature=temperature)
        if text:
            return text

        # fallback for SDK message shape differences
        response = client_together.chat.completions.create(
            model=_normalize_model_name(model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        raw_message = response.choices[0].message
        return _extract_message_text(raw_message.__dict__ if hasattr(raw_message, "__dict__") else raw_message)
    except Exception as e:
        print(f"Error querying {model}: {e}")
        raise RuntimeError(f"Error querying {model}: {e}") from e

async def query_llm_async(model_name, prompt, temperature=0.7):
    normalized_model = _normalize_model_name(model_name)
    # Check if using direct Moonshot Kimi API (not a Together endpoint)
    # Together endpoints have format: "username/provider/model-id" or contain "/"
    if "kimi" in normalized_model.lower() and "/" not in normalized_model:
        messages = [{"role": "user", "content": prompt}]
        return await call_kimi_api(messages, model=normalized_model, temperature=1.0)

    # Offload blocking sync call to a thread.
    response = await asyncio.wait_for(
        asyncio.to_thread(query_llm, normalized_model, prompt),
        timeout=LLM_QUERY_TIMEOUT_SECONDS
    )
    return response

async def query_vlm_async(model_name, image_paths, query):
    normalized_model = _normalize_model_name(model_name)
    task = asyncio.create_task(query_vlm(normalized_model, image_paths, query))
    response = await task
    return response
