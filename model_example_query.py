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
    gemini_key_PRIV = env_data["gemini_key"]


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
    """Query VLM about frames with retry logic and error handling"""
    grouped_response = []
    failed_images = []
    print("="*10 + " Querying VLM " + "="*10 + f"for image_paths {image_paths[:1]}...")
    
    # Create a new async client for this request to avoid session issues
    async_client = AsyncTogether(api_key=together_key_PRIV)
    
    # Process in batches to avoid overwhelming the API
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        for attempt in range(max_retries):
            if attempt > 0:
                # Wait before retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Retrying batch after {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            
            try:
                # Create tasks for images in this batch
                tasks = []
                for image_path in batch_paths:
                    try:
                        # Check if file exists
                        if not os.path.exists(image_path):
                            print(f"Warning: Image file not found: {image_path}")
                            failed_images.append((image_path, "File not found"))
                            continue
                        
                        # Read and encode image with error handling
                        try:
                            with open(image_path, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        except Exception as e:
                            print(f"Error reading image {image_path}: {e}")
                            failed_images.append((image_path, f"Read error: {e}"))
                            continue
                        
                        #print(f"Querying image: {image_path}")
                        
                        # Create async task with timeout
                        task = asyncio.wait_for(
                            async_client.chat.completions.create(
                                model=model,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": query},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                                    ]
                                }],
                                stream=False
                            ),
                            timeout=60  # 30 second timeout per image
                        )
                        tasks.append((image_path, task))
                    except Exception as e:
                        print(f"Error preparing task for {image_path}: {e}")
                        failed_images.append((image_path, str(e)))
                
                if not tasks:
                    print("No valid tasks in batch, skipping...")
                    break
                
                # Execute tasks with return_exceptions to handle individual failures
                responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                # Process responses
                batch_failed = []
                for (image_path, _), response in zip(tasks, responses):
                    if isinstance(response, Exception):
                        print(f"Failed to process {image_path}: {response}")
                        batch_failed.append(image_path)
                        if attempt == max_retries - 1:
                            failed_images.append((image_path, str(response)))
                    else:
                        try:
                            content = response.choices[0].message.content
                            grouped_response.append(f"timestamp: {image_path} {content}\n")
                        except Exception as e:
                            print(f"Error extracting response for {image_path}: {e}")
                            batch_failed.append(image_path)
                            if attempt == max_retries - 1:
                                failed_images.append((image_path, f"Response error: {e}"))
            
                
                # If all succeeded, move to next batch
                if not batch_failed:
                    break
                    
                # Otherwise, retry only failed images
                batch_paths = batch_failed
                print(f"Retrying {len(batch_failed)} failed images in batch...")
                
            except Exception as e:
                print(f"Batch processing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for image_path in batch_paths:
                        if not any(img == image_path for img, _ in failed_images):
                            failed_images.append((image_path, f"Batch error: {e}"))

    # Report failures
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for img_path, error in failed_images[:5]:  # Show first 5
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    else:
        pass
    
    # Condense response if we have any successful responses
    if grouped_response:
        try:
            condensed_response = await condense_vlm_response(' '.join(grouped_response))
            if condensed_response:
                print("CONDENSED RESPONSE: ", condensed_response[:50] + "...")
            else:
                print("CONDENSED RESPONSE: None")
            return {
                "individual responses": grouped_response, 
                "condensed response": condensed_response,
                "failed_images": failed_images
            }
        except Exception as e:
            print(f"Error condensing response: {e}")
            return {
                "individual responses": grouped_response,
                "condensed response": None,
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