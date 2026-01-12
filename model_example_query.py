#!/usr/bin/env python3

import base64
import time
import os
from together import Together, AsyncTogether
#from google.genai import Client
from together.types.chat_completions import PromptPart
from PIL import Image
#from google.genai import types
import json
import subprocess
import asyncio
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
#from token_tracker import record, num_tokens

# Initialize client with API key
with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key_PRIV = env_data["together_key"]


os.environ['TOGETHER_API_KEY'] = together_key_PRIV
#os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
client_together = Together()
# Don't create global async client - create per request instead
#genai.configure()

def log(message, file_title):
    if not os.path.exists(file_title):
        os.makedirs(file_title)
    else:
        with open(f"{file_title}/log.log", "a") as f:
            f.write(message + "\n")

async def query_vlm(model, image_paths, query, max_retries=20, batch_size=30):
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
                    response = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model=model,
                            messages=[{
                                "role": "user",
                                "content": content
                            }],
                            stream=False,
                            max_tokens=4096
                        ),
                        timeout=120  # 2 minute timeout for batch
                    )

                    # Extract response
                    content_response = response.choices[0].message.content
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
        response = client_together.chat.completions.create(
            model = model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        text = response.choices[0].message.content
        return text
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return None

async def query_llm_async(model_name, prompt):
    # Offload blocking sync call to a thread; do NOT wrap a non-coroutine in create_task
    response = await asyncio.to_thread(query_llm, model_name, prompt)
    return response

async def query_vlm_async(model_name, image_paths, query):
    task = asyncio.create_task(query_vlm(model_name, image_paths, query))
    response = await task
    return response