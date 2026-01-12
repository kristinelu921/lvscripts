########## CREATES BASELINE CAPTIONS FOR ALL FRAMES ##########
#!/usr/bin/env python3
import os
import json
import time
import asyncio
import re
from pathlib import Path
from together import AsyncTogether
from prompts import CES_log_prompt, global_summary_prompt
import base64


with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key = env_data["together_key"]
    os.environ['TOGETHER_API_KEY'] = together_key

async_client_together = AsyncTogether(api_key=together_key)
async def process_single_frame(frame_path, prompt, semaphore, results, output_file, file_lock, frame_num, total_frames, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """process a single frame with async API call and semaphore control"""

    async with semaphore:
        print(f"Processing {frame_path}")
        try:
            # Read image and convert to base64
            with open(frame_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Use Together's async API directly with proper format
            response = await async_client_together.chat.completions.create(
                model=model,
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

            print(f"API response received for {frame_path}")

            # Extract text from Together API response
            response_text = response.choices[0].message.content

            # Append subtitle if available and requested
            if use_subtitles and subtitle_frames:
                # Extract frame number from path (e.g., frames/frame_0123.jpg -> 123)
                frame_match = re.search(r'frame_(\d+)\.jpg', frame_path)
                if frame_match:
                    frame_num = int(frame_match.group(1))
                    if frame_num in subtitle_frames:
                        response_text += f" | Subtitle: {subtitle_frames[frame_num]}"

            result_entry = frame_path.split(".jpg")[0][-17:] + " seconds: " + response_text

            async with file_lock:
                results.append(result_entry)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent = 2)
            
            print(f"Completed {frame_path}")
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            return None

async def caption_frame_with_os(frames_dir = 'frames', output_file = 'captions/frame_captions.json', max_concurrent = 20, use_subtitles = False, subtitle_frames = None, model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """caption frames with tgt, optionally appending subtitles

    Args:
        frames_dir: Directory containing frame images
        output_file: Output file for captions
        max_concurrent: Max concurrent API calls
        use_subtitles: Whether to append subtitles to captions
        subtitle_frames: Dict mapping frame numbers to subtitle text {frame_num: subtitle_text}
    """

    prompt = """Please write ONE DESCRIPTIVE sentence that includes [subjects], [actions], [location/scene] if possible/relevant. Make sure to include detailed descriptions of subjects, actions, OBJECTS, large text, and the location/scene.""" #TODO: CAN OPTIMIZE HERE

    frames_path = Path(frames_dir)
    print(type(frames_path))
    frame_files = sorted([str(f) for f in frames_path.glob("*.jpg")])

    print(f"Processing {len(frame_files)} frames with Together API...")

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
    
    frames_to_process = [frame for frame in frame_files if frame.split(".")[0][-17:] not in processed_frames]

    semaphore = asyncio.Semaphore(max_concurrent)
    
    file_lock = asyncio.Lock()

    tasks = [process_single_frame(frame_path, prompt, semaphore, results, output_file, file_lock, i+1, len(frames_to_process), model) for i, frame_path in enumerate(frames_to_process)]

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

    return results

async def create_logs(captions_dir = 'frame_captions_sorted.json', output_file = "CES_file", prompt_fct = "", frames_dir = "/frames", model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", max_attempts = 3, num_chunks = 1):#picked for its long context length

    with open(captions_dir, "r") as captions_file:
        captions_data = json.load(captions_file)

    #frames_dir = "../frames/" #TODO: edit if needed
    num_frames = len(os.listdir(frames_dir))
    print("num_frames: ", num_frames)
    print("captions file accessed and read") #logging 

    # Read existing logs if they exist
    if os.path.exists("CES_logs.txt"):
        with open("CES_logs.txt", "r") as log_file:
            curr_logs = log_file.read()
    else:
        curr_logs = ""

    if os.path.exists(output_file):
        print("already done")
        return

    # Handle prompt_fct - if it's a function, call it; if it's a string, use it directly
    if callable(prompt_fct):
        if prompt_fct.__name__ == 'CES_log_prompt':
            prompt = prompt_fct(captions_data)
        else:
            prompt = prompt_fct(captions_data)
    else:
        prompt = prompt_fct
    for i in range(num_chunks):
        tasks = []
        for i in range(max_attempts):
            try:
                task = async_client_together.chat.completions.create(
                    model= model,
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    stream = False,
                    max_tokens = 1000000
                )
                tasks.append(task)
                print("task created")
                responses = await asyncio.gather(*[task])
                break
            except Exception as e:
                print(e)
                continue

        
        # Extract the actual content from the response
        if responses and responses[0]:
            # Extract the message content from the response object
            content = str(responses[0].choices[0].message.content) if responses[0].choices else "No content found"
            
            # Remove <think> tags if present
            if "<think>" in content and "</think>" in content:
                # Extract content after </think> tag
                content = content.split("</think>")[-1].strip()
            
            # Format the content nicely
            formatted_logs = "="*80 + "\n"
            formatted_logs += "CHARACTER, EVENT, AND SCENE LOGS\n"
            formatted_logs += "="*80 + "\n\n"
            formatted_logs += content
            formatted_logs += "\n\n" + "="*80 + "\n"
            formatted_logs += f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            formatted_logs += "="*80 + "\n"
            
            with open(output_file, "w") as f:
                f.write(formatted_logs)
            print("task completed")

        else:
            with open(output_file, "w") as f:
                f.write("No response received")
    return

async def process_many_captions(vid_folder, use_subtitles=False, subtitle_mapping_path='/mnt/ssh/data/longvideobench/subtitles_frame_mapping.json', model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Process captions for many videos, optionally with subtitles

    Args:
        vid_folder: Folder containing video directories
        use_subtitles: Whether to use subtitles
        subtitle_mapping_path: Path to subtitle frame mapping JSON
    """
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)
    print(curr_paths)

    # Load subtitle mappings if requested
    subtitle_data = {}
    if use_subtitles and os.path.exists(subtitle_mapping_path):
        with open(subtitle_mapping_path, 'r') as f:
            subtitle_data = json.load(f)
        print(f"Loaded subtitle mappings for {len(subtitle_data)} videos")

    for num in curr_paths:
        print("num", num)
        os.makedirs(f'{curr_folder}/{num}/captions', exist_ok = True)
        #with open(f'{curr_folder}/{num}/captions/frame_captions.json', 'w') as f:
        #    json.dump([], f)

    tasks = []
    for num in curr_paths:
        # Get subtitle frames for this video if available
        subtitle_frames = subtitle_data.get(num, {}).get('frames', {}) if use_subtitles else None
        # Convert string keys to int keys
        if subtitle_frames:
            subtitle_frames = {int(k): v for k, v in subtitle_frames.items()}

        tasks.append(caption_frame_with_os(
            frames_dir = f'{curr_folder}/{num}/frames',
            output_file = f'{curr_folder}/{num}/captions/frame_captions.json',
            max_concurrent = 10,
            use_subtitles = use_subtitles,
            subtitle_frames = subtitle_frames,
            model = model
        ))

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)


async def create_captions(dirname, use_subtitles=False, model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    await process_many_captions(dirname, use_subtitles=use_subtitles, model = model)

async def log_many_captions(vid_folder, model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)
    #curr_paths = ['00000032']
    print("logging")
    print(curr_paths)


    prompt_function = CES_log_prompt  # Rename to avoid conflict
    print(prompt_function)
    tasks = [create_logs(captions_dir = f'{curr_folder}/{num}/captions/frame_captions_sorted.json', output_file = f'{curr_folder}/{num}/captions/CES_logs.txt', prompt_fct = prompt_function, frames_dir = f'{curr_folder}/{num}/frames', model = model) for num in curr_paths]
    print(tasks)

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    print(completed_tasks)
    print("awaited")

async def summary_many_captions(vid_folder, model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)
    print("curr paths: ", curr_paths)
    prompt_function = global_summary_prompt  # Rename to avoid conflict
    tasks = [create_logs(captions_dir = f'{curr_folder}/{num}/captions/frame_captions_sorted.json', output_file = f'{curr_folder}/{num}/captions/global_summary.txt', prompt_fct = prompt_function, frames_dir = f'{curr_folder}/{num}/frames', model = model) for num in curr_paths]
    print(tasks)

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    failed_tasks = []
    for j, result in enumerate(completed_tasks):
        if isinstance(result, Exception):
            print(f"there was an error")
            failed_tasks.append(result)

    print(failed_tasks)

async def log_main(funct_name, model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", vid_dir = './videos_two'):
    if funct_name == "CES_logs":
        await log_many_captions(vid_dir, model = model)
    elif funct_name == "global_summary":
        await summary_many_captions(vid_dir, model = model)

def sort_captions(dirname):
    curr_paths = os.listdir(dirname)
    for num in curr_paths:
        os.makedirs(f'{dirname}/{num}/captions', exist_ok = True)
        #with open(f'{curr_folder}/{num}/captions/frame_captions.json', 'w') as f:
        #    json.dump([], f)

        with open(f'{dirname}/{num}/captions/frame_captions.json', "r") as f:
            print("Loading captions...")
            captions = json.load(f)
            sorted_captions = sorted(captions, key=lambda x: x.split(" seconds:")[0])
            print("Sorting captions...")
            with open(f'{dirname}/{num}/captions/frame_captions_sorted.json', "w") as f:
                json.dump(sorted_captions, f, indent=2)
            print("Saved sorted captions.")

async def run_all_captions(dirname, use_subtitles=False, model_vlm = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", model_llm = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Run complete caption pipeline with optional subtitles

    Args:
        dirname: Directory containing video folders
        use_subtitles: Whether to integrate subtitles into captions
    """
    # Since this is already an async function, just await the async functions directly
    await create_captions(dirname, use_subtitles=use_subtitles, model = model_vlm)
    print("CAPTIONS CREATED" + (" (with subtitles)" if use_subtitles else ""))

    sort_captions(dirname)
    print("SORTED CAPTIONS")

    from embed_frame_captions import embed_many
    await embed_many(dirname)
    print("EMBEDDED CAPTIONS")

    await summary_many_captions(dirname, model = model_llm)
    print("SUMMARY PROCESSED")

    await log_many_captions(dirname, model = model_llm)
    print("LOGS PROCESSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog = "caption_frames_os.py",
        description = "captions frames with optional subtitle integration"
    )

    parser.add_argument('dirname', help='Directory containing video folders')
    parser.add_argument('--use-subtitles', action='store_true',
                        help='Append subtitles to frame captions')
    parser.add_argument('--model-vlm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='VLM model to use')
    parser.add_argument('--model-llm', default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        help='LLM model to use')
    args = parser.parse_args()

    dirname = args.dirname

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_all_captions(dirname, use_subtitles=args.use_subtitles))
    print("ALL CAPTIONS PROCESSED")
