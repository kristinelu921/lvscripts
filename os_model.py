from model_example_query import query_vlm, query_llm, query_llm_async
from search_frame_captions import batch_embed_query_async, search_captions, search_clip_captions
from search_subtitles import search_subtitles
from extract_fine_grained_frames import extract_fine_grained_for_pipeline
from prompts import initial_prompt, followup_prompt, response_parsing_prompt, finish_prompt, _expand_frames_with_surrounding
from subtitle_utils import SubtitleLoader
import math
import json
import os
from together import AsyncTogether, Together
#from google import genai
import asyncio
json_file_lock = asyncio.Lock()

# Import prompts
import prompts as default_prompts

def get_prompts_module(use_no_vlm=False):
    """Get the appropriate prompts module based on configuration"""
    # Always use default prompts (no_vlm mode not needed)
    return default_prompts

def log(message, file_title):
    if not os.path.exists(file_title):
        os.makedirs(file_title)
    else:
        with open(f"{file_title}/log.log", "a") as f:
            f.write(message + "\n")


async def append_to_json_file(filepath, data):
    """append to json file with async lock"""
    async with json_file_lock:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        results = json.loads(content)
                    else:
                        results = []
            else:
                results = []
        except Exception as e:
            print("bleb")
            results = []
        
        results.append(data)
        temp_file = filepath + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        os.replace(temp_file, filepath)

        print(f"saved answer {data.get('uid', 'unknown')}!")
        return
    
with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key_PRIV = env_data["together_key"]

os.environ['TOGETHER_API_KEY'] = together_key_PRIV
#client = genai.Client(api_key=gemini_key_PRIV)

# Debug: Check if API keys are loaded
if not together_key_PRIV:
    print("WARNING: Together API key is not set!")
else:
    print(f"Together API key loaded (length: {len(together_key_PRIV)})")

class Pipeline:
    def __init__(self, llm_model_name, vlm_model_name, max_num_iterations=15):
        # Store model names
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        
        # Create client objects with model names embedded
        self.llm = llm_model_name
        self.vlm = vlm_model_name
        
        self.max_num_iterations = max_num_iterations
        self.scratchpad = []
        self.messages = []
        self.records = []  # Store recorded events for organizational tracking
    
    
    def llm_query(self, prompt):
        return query_llm(self.llm, prompt)
    
    async def llm_query_async(self, prompt):
        return await query_llm_async(self.llm, prompt)
    
    async def vlm_query(self, image_paths, prompt):
        result = await query_vlm(self.vlm, image_paths, prompt)
        return result
        

my_model = Pipeline("deepseek-ai/DeepSeek-V3.1", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

async def query_model_iterative_with_retry(model, question, uid, vid_path, output_file, max_retries=15, candidates=None, use_no_vlm=False, videos_dir="/mnt/ssd/data/longvideobench/videos", pass_all_subtitles_to_llm=False, subtitles_dir=None, embeddings_path=None):
    """Wrapper to retry query_model_iterative if it hangs"""
    # Default to frame captions embeddings if not specified
    if embeddings_path is None:
        embeddings_path = f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl"
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
                else:
                    results = []
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {output_file}: {e}")
            results = []
        try:
            for item in results:
                if item["uid"] == uid:
                    print (f"already completed question {uid}")
                    return item
        except Exception as e:
            print("Checking if q already completed error")
            pass

    for attempt in range(max_retries):
        try:
            message = f"Attempt {attempt + 1}/{max_retries} for question: {question[:50]}..."
            log(message, f"logs/log_video_{vid_path}_{uid}")
            # Set 60 second timeout for the entire iterative process
            result = await asyncio.wait_for(
                query_model_iterative(model, question, uid, vid_path, candidates, use_no_vlm=use_no_vlm, videos_dir=videos_dir, pass_all_subtitles_to_llm=pass_all_subtitles_to_llm, subtitles_dir=subtitles_dir, embeddings_path=embeddings_path),
                timeout=240  # 3 minute timeout
            )
            print(f"Successfully completed on attempt {attempt + 1}")
            if output_file:
                await append_to_json_file(output_file, result)
                await append_to_json_file('completed_uid.json', {"uid": uid})
            return result
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}, retrying...")
            # Reset model state for retry
            model.messages = []
            model.scratchpad = []
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts due to timeout")
                result = {
                    "uid": uid,
                    "question": question,
                    "answer": "TIMEOUT",
                    "reasoning": f"Failed to complete after {max_retries} attempts due to timeout",
                    "evidence_frame_numbers": []
                }
                if output_file:
                    await append_to_json_file(output_file, result)
                    await append_to_json_file('completed_uid.json', {"uid": uid})

                return 
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            # Reset model state for retry
            model.messages = []
            model.scratchpad = []
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts due to error: {e}")

                result = {
                    "uid": uid,
                    "question": question,
                    "answer": "ERROR",
                    "reasoning": f"Error: {str(e)}",
                    "evidence_frame_numbers": []
                }
                # Remove the redundant file write - it's handled by append_to_json_file below
                if output_file:
                    await append_to_json_file(output_file, result)
                    await append_to_json_file('completed_uid.json', {"uid": uid})

                return result
    
    # Shouldn't reach here, but just in case
    result =    {
                    "uid": uid,
                    "question": question,
                    "answer": "TIMEOUT",
                    "reasoning": f"Failed to complete after {max_retries} attempts due to timeout",
                    "evidence_frame_numbers": []
                }
    return result
    
async def query_model_iterative(model, question, question_uid, vid_path, candidates=None, use_no_vlm=False, pre_existing_messages=None, videos_dir="/mnt/ssd/data/longvideobench/videos", pass_all_subtitles_to_llm=False, subtitles_dir=None, embeddings_path=None):
    """Iteratively query any open-source model to answer questions about video

    Args:
        question: The question to answer
        question_uid: Unique identifier for the question
        vid_path: Path to the video directory
        candidates: List of answer choices (optional)
        use_no_vlm: Whether to use no-VLM mode
        pre_existing_messages: Optional list of previous messages to continue conversation from
        videos_dir: Directory containing source video files (.mp4) for fine-grained frame extraction
        pass_all_subtitles_to_llm: Whether to pass all video subtitles to LLM in initial prompt
        subtitles_dir: Directory containing subtitle embeddings (auto-detects if None)
        embeddings_path: Path to caption embeddings file (auto-detects if None)
    """
    # Default to frame captions embeddings if not specified
    if embeddings_path is None:
        embeddings_path = f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl"

    global_sum_path = vid_path + "/captions/global_summary.txt"
    CES_logs_path = vid_path + "/captions/CES_logs.txt"
    with open(global_sum_path, "r") as f:
        global_summary = f.read()

    with open(CES_logs_path, "r") as f:
        CES_log = f.read()

    question = question.strip()

    # Variable to store criteria extracted from initial response
    question_criteria = None

    # Check if subtitle embeddings are available for this video
    video_id = os.path.basename(vid_path.rstrip('/'))

    # Auto-detect subtitles directory if not provided
    if subtitles_dir is None:
        # Try common subtitle locations based on video path
        if 'videomme' in vid_path or 'video_mme_long' in vid_path:
            subtitles_dir = "/mnt/ssd/data/videomme/video_mme_long/subtitles_json"
        elif 'longvideobench' in vid_path:
            subtitles_dir = "/mnt/ssd/data/longvideobench/subtitles_val"
        elif 'lvbench' in vid_path:
            subtitles_dir = "/mnt/ssd/data/lvbench/subtitles"
        else:
            # Default fallback
            subtitles_dir = "/mnt/ssd/data/videomme/video_mme_long/subtitles_json"

    subtitle_embeddings_path = f"{subtitles_dir}/{video_id}_en_embeddings_alibaba.jsonl"
    subtitles_available = os.path.exists(subtitle_embeddings_path)
    use_subtitles = True  # Enable subtitle search feature

    # Load all subtitles if requested (for passing to LLM in initial prompt)
    all_subtitles_text = None
    if pass_all_subtitles_to_llm:
        subtitle_json_path = f"{subtitles_dir}/{video_id}_en.json"
        if os.path.exists(subtitle_json_path):
            try:
                loader = SubtitleLoader(subtitle_json_path)
                # Format all subtitles into a readable text format
                all_subtitles = []
                for sub in loader.subtitles:
                    all_subtitles.append(f"[{sub['start']} - {sub['end']}] {sub['line']}")
                all_subtitles_text = "\n".join(all_subtitles)
                print(f"✓ Loaded {len(loader.subtitles)} subtitles for LLM initial context")
            except Exception as e:
                print(f"⚠️ Failed to load subtitles for LLM: {e}")
                all_subtitles_text = None
        else:
            print(f"⚠️ Subtitle JSON not found for LLM context: {subtitle_json_path}")

    if subtitles_available:
        print(f"✓ Subtitles available for video {video_id}")
    else:
        print(f"⚠️ No subtitle embeddings found for video {video_id}")

    # Get the appropriate prompts module based on configuration
    prompts = get_prompts_module(use_no_vlm)

    vlm_note = " You should be able to extract detailed frame-information from videos, do caption searches, and use your findings to answer the question." if not use_no_vlm else " You can do caption searches to find relevant frames, and use your findings to answer the question based on the semantic similarity of captions."

    # If pre-existing messages provided, use them; otherwise start fresh
    if pre_existing_messages:
        model.messages = list(pre_existing_messages)  # Copy the messages
        print(f"✓ Loaded {len(pre_existing_messages)} pre-existing messages for judge context")
    else:
        model.messages.append({"role": "system", "content": f"You are an expert at reasoning and tool-using, with the goal of answering this question about a long video.{vlm_note} You should be SUPER PICKY about your findings, NOT make assumptions, and always bias towards gathering more evidence before executing a final answer. Use EXACT evidence only. ALSO, when dealing with TEMPORAL questions, you cannot find VISUAL TIMES. If a question asks for a 'duration' of an event, you want to do many caption searches on consecutive ranges, and find scene-changes at the beginning and end of the event. You MUST CHOOSE AN ANSWER. NONE OF THE ABOVE IS NOT ACCEPTABLE."})

        # Build user content with optional subtitles
        user_content = "\n Here is a global summary of the video for general context: " + global_summary + "\n\n Here is also an INCOMPLETE character/event/scene log across the video. These will all be encountered, and there MAY BE MORE " + CES_log

        # Add all subtitles if requested
        if all_subtitles_text:
            user_content += "\n\n=== COMPLETE VIDEO TRANSCRIPT (All Subtitles) ===\n"
            user_content += "Below are ALL the subtitles from the entire video with timestamps. You can reference these to understand what is being said throughout the video:\n\n"
            user_content += all_subtitles_text
            user_content += "\n\n=== END OF TRANSCRIPT ===\n"

        user_content += "\nYour question is this: " + question

        model.messages.append({"role": "user", "content": user_content})
    prompt = str(model.messages) + prompts.initial_prompt(question, candidates, use_subtitles=use_subtitles, subtitles_available=subtitles_available)
    message = prompt
    log(message, f"logs/log_video_{vid_path}_{question_uid}")

    
    for i in range(model.max_num_iterations):
        # Query the specified model
        print("="*20 + f"Querying model with prompt: {i}, {question_uid} "+ "="*20)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"model.messages count: {len(model.messages)}")
        try:
            #print("reached this thing")
            response = await model.llm_query_async(prompt)
            #print("response reached", response)
        except Exception as e:
            print(f"Failed to get model response: {e}")
            print(f"Error type: {type(e).__name__}")
            continue

        model.messages.append({"role": "assistant", "content": response})
        print(f"✓ Added assistant response to messages (total: {len(model.messages)})")
        
        message = response
        log(message, f"logs/log_video_{vid_path}_{question_uid}")
        if not response:
            print(f"Failed to get response at iteration {i+1}")
            continue
            
        # Parse the response
        if response and "</think>" in response:
            response = response.split("</think>")[1].strip()

        parsing_prompt = prompts.response_parsing_prompt(response)
        parsed_response = await model.llm_query_async(parsing_prompt)

        if not parsed_response:
            print(f"Failed to get parsing response at iteration {i+1}")
            continue
        
        print("reached here")
        
        def extract_json(text):
            if not text:
                return None

            original_text = text  # Keep for debugging

            # Strategy 1: Try direct parsing
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                pass

            # Strategy 2: Extract from markdown code blocks
            if "```json" in text:
                try:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError) as e:
                    pass

            # Strategy 3: Extract from any code blocks
            if "```" in text:
                try:
                    json_str = text.split("```")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError) as e:
                    pass

            # Strategy 4: Remove "json_output = " prefix
            try:
                cleaned = text.strip()
                if cleaned.startswith("json_output = "):
                    cleaned = cleaned[14:]
                elif cleaned.startswith("json_output="):
                    cleaned = cleaned[12:]
                cleaned = cleaned.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                pass

            # Strategy 5: Find JSON object boundaries with proper brace matching
            import re
            try:
                # Look for both objects {} and arrays []
                start_obj = text.find("{")
                start_arr = text.find("[")

                # Determine which comes first
                if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                    start = start_obj
                    open_char, close_char = '{', '}'
                elif start_arr != -1:
                    start = start_arr
                    open_char, close_char = '[', ']'
                else:
                    return None

                # Find matching closing bracket/brace
                count = 0
                in_string = False
                escape = False

                for i in range(start, len(text)):
                    char = text[i]

                    # Handle string boundaries
                    if char == '"' and not escape:
                        in_string = not in_string
                    elif char == '\\' and not escape:
                        escape = True
                        continue

                    # Count braces/brackets only outside strings
                    if not in_string:
                        if char == open_char:
                            count += 1
                        elif char == close_char:
                            count -= 1
                            if count == 0:
                                json_str = text[start:i+1]
                                # Try to fix common issues
                                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                                return json.loads(json_str)

                    escape = False

            except (json.JSONDecodeError, ValueError) as e:
                pass

            # Strategy 6: Clean whitespace and try again
            try:
                cleaned = text.strip().rstrip('\\').strip()
                # Remove common trailing characters
                cleaned = re.sub(r'[,;]+$', '', cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                pass

            return None

        original_parsed_text = parsed_response  # Save for error message
        parsed_response = extract_json(parsed_response)
        if parsed_response is None:
            print(f"Failed to extract JSON from response")
            print(f"Raw response: {original_parsed_text[:500]}...")  # Print first 500 chars
            continue

        # Debug: Print detected tool and normalize it
        detected_tool = parsed_response.get("tool", "UNKNOWN")
        # Normalize: strip whitespace and convert to uppercase for matching
        if isinstance(detected_tool, str):
            detected_tool = detected_tool.strip().upper()
            parsed_response["tool"] = detected_tool  # Update the parsed response
        print(f"🔧 Detected tool: {detected_tool}")

        try:
            if parsed_response.get("tool") == "FINAL_ANSWER":
                # The parsed response has "frames" field, not "evidence_frame_numbers"
                message = f"FINAL_ANSWER: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # Convert answer to int format (handle both letter and number responses)
                parsed_answer = parsed_response.get("answer")
                if isinstance(parsed_answer, str):
                    parsed_answer = parsed_answer.strip().upper()
                    # If it's a letter (A, B, C, D, E), convert to number
                    if parsed_answer in ['A', 'B', 'C', 'D', 'E']:
                        parsed_answer = ord(parsed_answer) - ord('A')
                    # If it's already a number string, convert to int
                    elif parsed_answer.isdigit():
                        parsed_answer = int(parsed_answer)
                    else:
                        # Keep as is if unrecognized format
                        pass
                elif isinstance(parsed_answer, int):
                    # Already an int, keep as is
                    pass

                new_response = {
                    "uid": question_uid,
                    "question": question,
                    "answer": parsed_answer,
                    "reasoning": parsed_response.get("reasoning"),
                    "evidence_frame_numbers": parsed_response.get("frames")  # Map "frames" to "evidence_frame_numbers"
                }

                # Include criteria if they were extracted
                if question_criteria:
                    new_response["criteria"] = question_criteria

                # Include answer-specific criteria if present in response
                if "answer_criteria" in parsed_response:
                    new_response["answer_criteria"] = parsed_response.get("answer_criteria", [])

                with open(f"{vid_path}/{question_uid}_os_model.json", "w") as f:
                    json.dump(model.messages, f, indent=2)
                    with open(f"answers_logs.json", "a") as f:
                        f.write(f"saved model messages for question {question_uid}, video {vid_path}\n")

                return new_response
            elif parsed_response.get("tool") == "VLM_QUERY":
                message = f"VLM_QUERY: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print("parsed response: ", parsed_response)
                print("="*60 + "Querying VLM" + "="*60)

                # Get requested frames and expand them to include ±5 seconds of context
                frames = parsed_response.get("frames")
                expanded_frames = _expand_frames_with_surrounding(frames, seconds_before=5, seconds_after=5)
                print(f"Original frames ({len(frames)}): {frames}")
                print(f"Expanded frames ({len(expanded_frames)}): {expanded_frames[:10]}..." if len(expanded_frames) > 10 else f"Expanded frames ({len(expanded_frames)}): {expanded_frames}")

                prompt = "Here is a global summary of the video for general context: " + global_summary + "\n"
                prompt += f"Note: You are viewing {len(expanded_frames)} frames including ~5 seconds before/after the key frames for context.\n"
                prompt += parsed_response.get("prompt")
                print("PROMPT: ", prompt)

                new_frames = [(f"{vid_path}/" + frame) for frame in expanded_frames]
                retrieved_info = await model.vlm_query(new_frames, prompt)
                model.messages.append({"role": "vlm response", "content": retrieved_info})

            elif parsed_response.get("tool") == "CAPTION_SEARCH":
                # Handle multiple search queries or single query
                print("reaching parsed response get tool caption search")

                # Extract criteria if this is the first iteration (i == 0)
                if i == 0 and question_criteria is None and "criteria" in parsed_response:
                    question_criteria = parsed_response.get("criteria", [])
                    print(f"Extracted {len(question_criteria)} verification criteria")
                    message = f"Criteria: {question_criteria}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # Check for multiple search_queries (new format)
                search_queries = parsed_response.get("search_queries")

                # Fallback to single query (legacy format)
                if not search_queries:
                    search_query = parsed_response.get("input") or parsed_response.get("prompt")
                    if isinstance(search_query, list):
                        search_queries = search_query
                    elif search_query:
                        search_queries = [search_query]
                    else:
                        print("Warning: No search query found in CAPTION_SEARCH")
                        continue

                print(f"Performing {len(search_queries)} search queries: {search_queries}")
                message = f"Search queries ({len(search_queries)}): {search_queries}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # Perform all searches and collect results
                all_results = []
                # Detect if using clip captions based on embeddings path
                use_clip_search = 'clip_embeddings' in str(embeddings_path)

                for idx, query in enumerate(search_queries):
                    print(f"  Query {idx+1}/{len(search_queries)}: {query}")
                    # Use appropriate search function based on caption type
                    if use_clip_search:
                        results = await search_clip_captions(vid_path, question_uid, query, embeddings_path, 30)
                    else:
                        results = await search_captions(vid_path, question_uid, query, embeddings_path, 30)

                    all_results.append({
                        "query": query,
                        "results": results if isinstance(results, list) else [results]
                    })

                # Format results for LLM to process
                retrieved_info_parts = [f"Retrieved frames from {len(search_queries)} different search queries.\n"]
                retrieved_info_parts.append("Review all results below and choose the most relevant frames for your question.\n")

                for idx, query_result in enumerate(all_results):
                    retrieved_info_parts.append(f"\n--- Query {idx+1}: \"{query_result['query']}\" ---")
                    results = query_result['results']
                    if isinstance(results, list) and len(results) > 0:
                        retrieved_info_parts.append(json.dumps(results, indent=2))
                    else:
                        retrieved_info_parts.append("No results")

                retrieved_info = "\n".join(retrieved_info_parts)

                message = f"Caption search completed: {len(search_queries)} queries executed"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                model.messages.append({"role": "caption search results", "content": retrieved_info})

            elif parsed_response.get("tool") == "RECORD":
                # Record relevant observations for organizational tracking
                entries = parsed_response.get("entries", [])
                if not isinstance(entries, list):
                    entries = [entries]  # Convert single entry to list

                print(f"Recording {len(entries)} event(s)")
                for entry in entries:
                    # Parse time from entry (format: "Time: XX seconds, Event: ...")
                    import re
                    time_match = re.search(r'Time:\s*(\d+)\s*seconds?', entry, re.IGNORECASE)
                    if time_match:
                        time_sec = int(time_match.group(1))
                        model.records.append({"time": time_sec, "entry": entry})
                    else:
                        # If time not parseable, add anyway with time=-1
                        model.records.append({"time": -1, "entry": entry})

                # Sort records by time
                model.records.sort(key=lambda x: x["time"])

                message = f"RECORD: Added {len(entries)} entries. Total records: {len(model.records)}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # AUTOMATICALLY CALL VIEW_RECORDS after RECORD
                print(f"Automatically viewing {len(model.records)} recorded event(s)")

                retrieved_info = f"Successfully recorded {len(entries)} event(s).\n\n"
                retrieved_info += f"=== ALL RECORDED EVENTS ({len(model.records)} total) ===\n\n"
                for idx, record in enumerate(model.records, 1):
                    retrieved_info += f"{idx}. {record['entry']}\n"
                retrieved_info += "\n=== END OF RECORDS ===\n"
                retrieved_info += "\nUse this organized timeline to reason about sequences, relationships, and answer the question."

                log(f"VIEW_RECORDS: Auto-displayed {len(model.records)} records after RECORD", f"logs/log_video_{vid_path}_{question_uid}")
                model.messages.append({"role": "system", "content": retrieved_info})

            elif parsed_response.get("tool") == "VIEW_RECORDS":
                # Return all recorded observations sorted by time
                print(f"Viewing {len(model.records)} recorded event(s)")

                if len(model.records) == 0:
                    retrieved_info = "No events have been recorded yet. Use the RECORD tool after VLM_QUERY calls to track relevant observations."
                else:
                    retrieved_info = f"=== ALL RECORDED EVENTS ({len(model.records)} total) ===\n\n"
                    for idx, record in enumerate(model.records, 1):
                        retrieved_info += f"{idx}. {record['entry']}\n"
                    retrieved_info += "\n=== END OF RECORDS ===\n"
                    retrieved_info += "\nUse this organized timeline to reason about sequences, relationships, and answer the question."

                message = f"VIEW_RECORDS: Displayed {len(model.records)} records"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                model.messages.append({"role": "system", "content": retrieved_info})

            elif parsed_response.get("tool") == "SUBTITLE_SEARCH":
                # Embeddings-based subtitle semantic search
                query = parsed_response.get("query", "")
                topk = parsed_response.get("topk", 10)  # Optional parameter
                print(f"Searching subtitles (embeddings) for: '{query}' (top-{topk})")

                # Extract video ID from vid_path (e.g., "/path/to/videos_processed/Y0IaijKNGX8")
                video_id = os.path.basename(vid_path.rstrip('/'))

                # Path to subtitle embeddings (use the same subtitles_dir determined earlier)
                embeddings_path = f"{subtitles_dir}/{video_id}_en_embeddings_alibaba.jsonl"

                try:
                    # Search subtitles using embeddings
                    subtitle_results = search_subtitles(
                        embeddings_path=embeddings_path,
                        query=query,
                        topk=topk,
                        fps=1.0
                    )

                    print(f"✓ Subtitle search returned {len(subtitle_results) if subtitle_results else 0} results")

                    if subtitle_results:
                        # Format results with frame information and similarity scores
                        retrieved_info = f"Found {len(subtitle_results)} matching subtitle(s) for query: '{query}'\n"
                        retrieved_info += f"(Ranked by semantic similarity using embeddings)\n\n"

                        for result in subtitle_results:
                            retrieved_info += f"[#{result['rank']}] Score: {result['score']:.3f} | Frame {result['start_frame']} ({result['time_formatted']})\n"
                            retrieved_info += f"    Text: \"{result['text']}\"\n"
                            retrieved_info += f"    Time: {result['start_sec']:.2f}s - {result['end_sec']:.2f}s\n"
                            retrieved_info += f"    Frame path: frames/frame_{result['start_frame']:04d}.jpg\n\n"

                        retrieved_info += "\nYou can now query these specific frames with VLM_QUERY to verify the visual content."

                        message = f"SUBTITLE_SEARCH: Found {len(subtitle_results)} matches for '{query}'"
                        log(message, f"logs/log_video_{vid_path}_{question_uid}")
                        print(f"✓ SUBTITLE_SEARCH success: {len(subtitle_results)} results")
                    else:
                        retrieved_info = f"No subtitles found matching: '{query}'\nTry a different search term or use CAPTION_SEARCH instead."
                        message = f"SUBTITLE_SEARCH: No matches for '{query}'"
                        log(message, f"logs/log_video_{vid_path}_{question_uid}")
                        print(f"⚠️ SUBTITLE_SEARCH: No results")

                    model.messages.append({"role": "subtitle search results", "content": retrieved_info})
                    print(f"✓ Added subtitle results to model.messages")

                except FileNotFoundError as e:
                    retrieved_info = f"Subtitle embeddings not found for video {video_id}. Subtitles may not be available or not yet embedded. Use CAPTION_SEARCH instead."
                    message = f"SUBTITLE_SEARCH: Embeddings not found for {video_id}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})
                    print(f"✗ SUBTITLE_SEARCH: File not found - {embeddings_path}")
                except Exception as e:
                    retrieved_info = f"Error searching subtitles: {str(e)}. Try using CAPTION_SEARCH instead."
                    message = f"SUBTITLE_SEARCH: Error - {str(e)}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})
                    print(f"✗ SUBTITLE_SEARCH: Exception - {str(e)}")

            elif parsed_response.get("tool") == "EXTRACT_FINE_GRAINED_FRAMES":
                # Extract fine-grained frames at higher FPS
                start_sec = parsed_response.get("start_second", 0.0)
                end_sec = parsed_response.get("end_second", 0.0)
                fps = parsed_response.get("fps", 5)

                # Validate parameters
                if not (1 <= fps <= 10):
                    fps = min(10, max(1, fps))  # Clamp to valid range

                if start_sec >= end_sec:
                    retrieved_info = f"Error: start_second ({start_sec}) must be less than end_second ({end_sec})"
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                if start_sec < 0:
                    retrieved_info = f"Error: start_second cannot be negative. Got: {start_sec}"
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                # Limit extraction to reasonable duration (max 10 seconds at a time)
                max_duration = 10.0
                if (end_sec - start_sec) > max_duration:
                    retrieved_info = f"Error: Time range too large ({end_sec - start_sec:.1f}s). Maximum is {max_duration}s. Please use a smaller range or multiple calls."
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                print(f"Extracting fine-grained frames: {start_sec}s to {end_sec}s at {fps} FPS")

                # Extract video ID from vid_path
                video_id = os.path.basename(vid_path.rstrip('/'))

                try:
                    # Extract frames
                    frame_paths = extract_fine_grained_for_pipeline(
                        video_id=video_id,
                        start_second=start_sec,
                        end_second=end_sec,
                        fps=fps,
                        videos_dir=videos_dir,  # Use the passed videos_dir parameter
                        output_base=os.path.dirname(os.path.dirname(vid_path)),  # Get parent of video folder
                        vid_path=vid_path  # Pass the actual video directory path
                    )

                    if frame_paths:
                        duration = end_sec - start_sec
                        retrieved_info = f"Successfully extracted {len(frame_paths)} fine-grained frames from {start_sec}s to {end_sec}s at {fps} FPS.\n\n"
                        retrieved_info += f"Duration: {duration:.1f} seconds\n"
                        retrieved_info += f"Frame interval: {1.0/fps:.2f}s\n\n"
                        retrieved_info += "Extracted frames:\n"

                        for frame_path in frame_paths[:10]:  # Show first 10
                            retrieved_info += f"  - {frame_path}\n"

                        if len(frame_paths) > 10:
                            retrieved_info += f"  ... and {len(frame_paths) - 10} more frames\n"

                        retrieved_info += f"\nYou can now use these frames with VLM_QUERY to analyze fine details like hand motions, quick actions, or subtle movements."

                        message = f"EXTRACT_FINE_GRAINED_FRAMES: Extracted {len(frame_paths)} frames at {fps} FPS"
                        log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    else:
                        retrieved_info = f"Failed to extract frames from {start_sec}s to {end_sec}s"
                        message = f"EXTRACT_FINE_GRAINED_FRAMES: Failed"
                        log(message, f"logs/log_video_{vid_path}_{question_uid}")

                    model.messages.append({"role": "fine frames extraction", "content": retrieved_info})

                except FileNotFoundError as e:
                    retrieved_info = f"Video file not found for ID {video_id}. Cannot extract fine-grained frames. Error: {str(e)}"
                    message = f"EXTRACT_FINE_GRAINED_FRAMES: Video not found"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})
                except RuntimeError as e:
                    # RuntimeError contains detailed info from extract_fine_grained_frames.py
                    error_msg = str(e)
                    if "No frames were extracted" in error_msg:
                        # Provide helpful guidance to the model
                        retrieved_info = f"⚠️ Frame extraction failed: {error_msg}\n\n"
                        retrieved_info += "SUGGESTIONS:\n"
                        retrieved_info += "1. Check if the time range is valid (use CAPTION_SEARCH or SUBTITLE_SEARCH to find correct timestamps)\n"
                        retrieved_info += "2. Try a different time range\n"
                        retrieved_info += "3. Verify the timestamps are in seconds (not frame numbers)\n"
                        retrieved_info += "4. Make sure start_second and end_second are within the video duration"
                    else:
                        retrieved_info = f"Error extracting fine-grained frames: {error_msg}"
                    message = f"EXTRACT_FINE_GRAINED_FRAMES: Error - {error_msg[:100]}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})
                except Exception as e:
                    retrieved_info = f"Unexpected error extracting fine-grained frames: {str(e)}"
                    message = f"EXTRACT_FINE_GRAINED_FRAMES: Error - {str(e)}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})

            elif parsed_response.get("tool") == "CROP_OBJECT":
                # Crop specific objects from a frame for detailed analysis
                from crop_objects_gemini import crop_objects_from_frame

                frame_path_rel = parsed_response.get("frame", "")
                object_query = parsed_response.get("object_query", "")

                if not frame_path_rel:
                    retrieved_info = "Error: No frame specified for CROP_OBJECT. Please provide a frame path."
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                if not object_query:
                    retrieved_info = "Error: No object_query specified for CROP_OBJECT. Please describe what object to detect (e.g., 'bird on branch')."
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                # Convert relative frame path to absolute
                frame_path_abs = os.path.join(vid_path, frame_path_rel)

                if not os.path.exists(frame_path_abs):
                    retrieved_info = f"Error: Frame not found at {frame_path_rel}. Please check the frame path."
                    model.messages.append({"role": "system", "content": retrieved_info})
                    continue

                print(f"Cropping objects from {frame_path_rel}")
                print(f"Object query: '{object_query}'")

                message = f"CROP_OBJECT: Detecting '{object_query}' in {frame_path_rel}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                try:
                    # Perform object detection and cropping
                    result = crop_objects_from_frame(
                        frame_path=frame_path_abs,
                        object_query=object_query,
                        output_dir=None  # Will use default: vid_path/cropped_objects
                    )

                    if result['success']:
                        cropped_paths = result['cropped_paths']
                        detections = result['detections']

                        if cropped_paths:
                            # Convert absolute paths back to relative paths for LLM
                            cropped_paths_rel = []
                            for crop_path in cropped_paths:
                                # Get path relative to vid_path
                                rel_path = os.path.relpath(crop_path, vid_path)
                                cropped_paths_rel.append(rel_path)

                            retrieved_info = f"✓ Successfully detected and cropped {len(cropped_paths)} object(s) matching '{object_query}'\n\n"
                            retrieved_info += f"Source frame: {frame_path_rel}\n\n"
                            retrieved_info += "Cropped objects saved to:\n"
                            for idx, (crop_path, detection) in enumerate(zip(cropped_paths_rel, detections), 1):
                                label = detection.get('label', 'object')
                                retrieved_info += f"  {idx}. {crop_path} - {label}\n"

                            retrieved_info += f"\n📌 NEXT STEP: Use VLM_QUERY with these cropped images to analyze fine details.\n"
                            retrieved_info += f"Example: VLM_QUERY with frames: {cropped_paths_rel[:3]}\n\n"
                            retrieved_info += "These cropped images show ONLY the detected objects, making it easier to see small details."

                            message = f"CROP_OBJECT: Successfully cropped {len(cropped_paths)} objects"
                            log(message, f"logs/log_video_{vid_path}_{question_uid}")
                        else:
                            retrieved_info = f"⚠️ No objects matching '{object_query}' were detected in {frame_path_rel}.\n\n"
                            retrieved_info += "SUGGESTIONS:\n"
                            retrieved_info += "1. Try a different frame - use CAPTION_SEARCH to find frames containing the object\n"
                            retrieved_info += "2. Modify your object_query to be more general (e.g., 'bird' instead of 'blue bird')\n"
                            retrieved_info += "3. Use VLM_QUERY on the full frame first to confirm the object is present"

                            message = f"CROP_OBJECT: No objects detected for '{object_query}'"
                            log(message, f"logs/log_video_{vid_path}_{question_uid}")

                    else:
                        # Error occurred
                        error = result.get('error', 'Unknown error')
                        retrieved_info = f"Error cropping objects: {error}"
                        message = f"CROP_OBJECT: Error - {error}"
                        log(message, f"logs/log_video_{vid_path}_{question_uid}")

                    model.messages.append({"role": "crop object results", "content": retrieved_info})

                except Exception as e:
                    retrieved_info = f"Unexpected error in CROP_OBJECT: {str(e)}"
                    message = f"CROP_OBJECT: Exception - {str(e)}"
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    model.messages.append({"role": "system", "content": retrieved_info})

            else:
                tool_name = parsed_response.get('tool')
                message = f"Invalid or unrecognized tool: '{tool_name}' (type: {type(tool_name)}, repr: {repr(tool_name)})"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print(f"❌ {message}")
                print(f"   Valid tools: FINAL_ANSWER, VLM_QUERY, CAPTION_SEARCH, RECORD, VIEW_RECORDS, SUBTITLE_SEARCH, EXTRACT_FINE_GRAINED_FRAMES, CROP_OBJECT")
                print(f"   Parsed response keys: {list(parsed_response.keys())}")
                continue

            # Update prompt for next iteration
            if parsed_response.get("tool") == "CAPTION_SEARCH":
                num_queries = len(search_queries) if 'search_queries' in locals() else 1
                if num_queries > 1:
                    retrieved_info = f"""
The following results are from {num_queries} different search queries targeting different aspects of the question.

FRAME SELECTION STRATEGY:
When analyzing these multi-query results, use your best judgment to identify the BEST scene by:

1. **TIME CLUSTERING** (Most Important): Look for frames from similar time ranges across different queries
   - Identify which time period (frame numbers) appears most frequently across queries
   - Choose a WINDOW of frames from the same time range (e.g., if Query 1 returns frame_0050 and Query 2 returns frame_0052, these cluster together)
   - Prefer consecutive or nearby frames that tell a complete story

2. **OVERLAPPING FRAMES**: Frames that appear in multiple query results are high-confidence matches
   - If the same frame appears in 2+ queries, it's likely relevant to multiple criteria

3. **HIGH-CONFIDENCE SCORES**: Frames with high similarity scores (close to 1.0) from any query
   - Top results from each query are strong candidates

4. **CLUSTER ANALYSIS**: Look for groups of frames that cluster together temporally
   - Example: If multiple queries return frames in the 45-55 second range, focus there

Your goal: Select a WINDOW of frames from the SAME TIME PERIOD that best satisfies all the different search criteria.
Do not pick scattered frames from different parts of the video - choose a coherent scene.

Now review the results and choose the most relevant keyframes for VLM querying.
""" + "\n" + str(model.messages[-1].get("content", ""))
                else:
                    retrieved_info = "The following is the retrieved information from the caption search: Please read through and choose the most relevant few keyframes.\n" + str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "VLM_QUERY":
                retrieved_info = "The following is the retrieved information from the VLM query (frames expanded to include ±5 seconds of context for better understanding): Please read through and see if these are the scenes you're looking for. If not, please look for different scenes. If yes, extract detailed and important evidence from them.\n" + str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "RECORD":
                retrieved_info = str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "VIEW_RECORDS":
                retrieved_info = str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "SUBTITLE_SEARCH":
                retrieved_info = "The following is the retrieved information from subtitle search (semantic similarity using embeddings). Review the frames where relevant subtitle content appears and use VLM_QUERY to verify the visual content if needed.\n" + str(model.messages[-1].get("content", ""))
                print(f"✓ Preparing prompt with SUBTITLE_SEARCH results (length: {len(retrieved_info)} chars)")
            elif parsed_response.get("tool") == "EXTRACT_FINE_GRAINED_FRAMES":
                retrieved_info = "The following fine-grained frames have been extracted at higher FPS. You can now use VLM_QUERY with these frames to analyze detailed movements and actions.\n" + str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "CROP_OBJECT":
                retrieved_info = "Objects have been detected and cropped from the frame. These cropped images show ONLY the detected objects with fine details. Use VLM_QUERY with the cropped image paths to analyze specific object details.\n" + str(model.messages[-1].get("content", ""))
            else:
                retrieved_info = str(model.messages[-1].get("content", ""))

            # CRITICAL: Include full conversation history in prompt
            prompt = str(model.messages) + prompts.followup_prompt(retrieved_info, question, candidates, use_subtitles=use_subtitles, subtitles_available=subtitles_available)
            print(f"✓ Updated prompt for next iteration (length: {len(prompt)} chars, messages: {len(model.messages)})")
        except Exception as e:
            message = f"Error updating prompt: {e}"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")
            print(f"Error updating prompt: {e}")
            continue
    
    # Return final formatted scratchpad
    final_prompt = finish_prompt(model.messages, candidates)
    final_answer = await model.llm_query_async(final_prompt)
    
    if not final_answer:
        print(f"Failed to get final answer for question {question_uid}")
        result = {
            "uid": question_uid,
            "question": question,
            "answer": "ERROR",
            "reasoning": "Failed to get final answer from LLM",
            "evidence_frame_numbers": []
        }
        if question_criteria:
            result["criteria"] = question_criteria
        return result

    # Parse the final answer if it's in JSON format
    try:
        if isinstance(final_answer, str):
            # Try to extract JSON from the final answer
            if "```json" in final_answer:
                json_str = final_answer.split("```json")[1].split("```")[0].strip()
                parsed_final = json.loads(json_str)
            elif "{" in final_answer and "}" in final_answer:
                start = final_answer.find("{")
                end = final_answer.rfind("}") + 1
                json_str = final_answer[start:end]
                parsed_final = json.loads(json_str)
            else:
                # Return as is if not JSON
                answer = final_answer

                # Try to match answer against candidates if provided
                if candidates:
                    # Try exact match first
                    for idx, candidate in enumerate(candidates):
                        if str(final_answer).strip().lower() == candidate.strip().lower():
                            answer = idx
                            break
                        # Try matching without punctuation
                        if str(final_answer).strip().lower() == candidate.strip().rstrip('.').lower():
                            answer = idx
                            break
                        # Try matching if answer is just a number and candidate starts with that number
                        if str(final_answer).strip() in candidate.split('.')[0].strip():
                            answer = idx
                            break

                # Fallback to letter/number parsing if no candidates match
                if answer == final_answer:
                    if final_answer in "012345":
                        answer = int(final_answer)
                    elif final_answer in "ABCDE":
                        answer = ord(final_answer) - ord('A')

                result = {
                    "uid": question_uid,
                    "question": question,
                    "answer": answer,
                    "reasoning": "Final iteration response",
                    "evidence_frame_numbers": []
                }
                if question_criteria:
                    result["criteria"] = question_criteria
                return result
            with open(f"{vid_path}/{question_uid}_os_model.json", "w") as f:
                    json.dump(model.messages, f, indent=2)
                    with open(f"answers_logs.json", "a") as f:
                        f.write(f"saved model messages for question {question_uid}, video {vid_path}\n")

            answer = parsed_final.get("answer", "")

            # Try to match answer against candidates if provided
            if candidates:
                original_answer = answer
                for idx, candidate in enumerate(candidates):
                    if str(answer).strip().lower() == candidate.strip().lower():
                        answer = idx
                        break
                    # Try matching without punctuation
                    if str(answer).strip().lower() == candidate.strip().rstrip('.').lower():
                        answer = idx
                        break
                    # Try matching if answer is just a number and candidate starts with that number
                    if str(answer).strip() in candidate.split('.')[0].strip():
                        answer = idx
                        break

                # Fallback to letter/number parsing if no match
                if answer == original_answer:
                    if answer in "012345":
                        answer = int(answer)
                    elif answer in "ABCDE":
                        answer = ord(answer) - ord('A')
            else:
                # No candidates, use letter/number parsing
                if answer in "012345":
                    answer = int(answer)
                elif answer in "ABCDE":
                    answer = ord(answer) - ord('A')

            result = {
                "uid": question_uid,
                "question": question,
                "answer": answer,
                "reasoning": parsed_final.get("reasoning", ""),
                "evidence_frame_numbers": parsed_final.get("frames", [])
            }
            if question_criteria:
                result["criteria"] = question_criteria
            return result
    except:
        pass

    message = model.messages
    log(message, f"logs/log_video_{vid_path}_{question_uid}")

    answer = final_answer

    # Try to match answer against candidates if provided
    if candidates:
        for idx, candidate in enumerate(candidates):
            if str(final_answer).strip().lower() == candidate.strip().lower():
                answer = idx
                break
            # Try matching without punctuation
            if str(final_answer).strip().lower() == candidate.strip().rstrip('.').lower():
                answer = idx
                break
            # Try matching if answer is just a number and candidate starts with that number
            if str(final_answer).strip() in candidate.split('.')[0].strip():
                answer = idx
                break

    # Fallback to letter/number parsing if no match
    if answer == final_answer:
        if final_answer in "012345":
            answer = int(final_answer)
        elif final_answer in "ABCDE":
            answer = ord(final_answer) - ord('A')

    result = {
        "uid": question_uid,
        "question": question,
        "answer": answer,
        "reasoning": "Could not parse final response",
        "evidence_frame_numbers": []
    }
    if question_criteria:
        result["criteria"] = question_criteria
    return result

async def answer_question(question_uid, question, vid_folder, vid_num, candidates=None, vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", llm_model="deepseek-ai/DeepSeek-V3.1", use_no_vlm=False, videos_dir="/mnt/ssd/data/longvideobench/videos", pass_all_subtitles_to_llm=False, subtitles_dir=None, embeddings_path=None):
    try:
        # Create a separate Pipeline instance for each question to avoid shared state
        #qwen model : Qwen/Qwen3-235B-A22B-Instruct-2507-tput
        model = Pipeline(llm_model, vlm_model)
        curr_folder = str(vid_folder)
        num = vid_num
        vid_path = curr_folder + "/" + num
        print("vid_path", vid_path) #TODO: remove this
        answers_path = f'{curr_folder}/{num}/{num}_answers.json'
        answer = await query_model_iterative_with_retry(model, question, question_uid, vid_path, answers_path, candidates=candidates, use_no_vlm=use_no_vlm, videos_dir=videos_dir, pass_all_subtitles_to_llm=pass_all_subtitles_to_llm, subtitles_dir=subtitles_dir, embeddings_path=embeddings_path)
        print("answer", answer)
        return answer
    except Exception as e:
        print(f"Error processing question {question_uid}: {e}")
        return {
            "uid": question_uid,
            "question": question,
            "answer": "ERROR",
            "reasoning": f"Failed to process question: {str(e)}",
            "evidence_frame_numbers": []
        }

async def one_vid(vid_folder, vid_num):
    curr_folder = vid_folder
    num = vid_num
    vid_path = curr_folder + "/" + num
    questions_path = f'{curr_folder}/{num}/{num}_questions.json'
    print("Q PATH", questions_path)
    answers_path = f'{curr_folder}/{num}/{num}_answers.json'
    batch_size = 20
    with open(questions_path, "r") as f:
        questions = json.load(f)


    # Process questions in batches
    total_questions = len(questions)
    try:
        for i in range(math.ceil(total_questions/batch_size)):
            q_batch = [questions[j] for j in range(i * batch_size, (i+1) * batch_size) if j < len(questions)]
            print(f"Processing batch {i+1}/{math.ceil(total_questions/batch_size)} with {len(q_batch)} questions")
            
            tasks = [answer_question(q["uid"], q["question"], vid_folder, vid_num) for q in q_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for q, result in zip(q_batch, results):
                print(q["uid"])
                if isinstance(result, Exception):
                    print(f"Failed question {q['uid']}: {result}")
                else:
                    print("result: ", result)
                    print(f"Completed question {q['uid']}")
    except Exception as e:
        print(f"Error processing video {vid_num}: {e}")



    print(f"ans vid {num} have been generated")

    # Reformat answers
    reformatted_answers = await reformat_answers(f'{curr_folder}/{num}/{num}_answers.json')
    with open(f'{curr_folder}/{num}/{num}_answers_reformatted.json', "w") as f:
        json.dump(reformatted_answers, f, indent=2)

async def all_vids(vid_folder, batch_size = 1):
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)[:10]
    print(curr_paths)
    print(curr_paths)
    all_tasks = []
    task_info = []

    #curr_paths = ["00000031"]

    for num in curr_paths:
        all_tasks.append(one_vid(vid_folder, num))
        task_info.append(num)
    
    total_tasks = len(all_tasks)
    failed_tasks = []
    for i in range(0, total_tasks, batch_size):
        batch_tasks = all_tasks[i:i+batch_size]
        batch_info = task_info[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_tasks + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} (videos: {', '.join(batch_info)})")

        try:
            completed = await asyncio.gather(*batch_tasks, return_exceptions=True)
            print("completed", completed)
            for j, result in enumerate(completed):
                if isinstance(result, Exception):
                    print(f"Error processing video {batch_info[j]}: {result}")
                    failed_tasks.append(batch_info[j])
                else:
                    print(f"Successfully processed video {batch_info[j]}")
        except Exception as e:
            print(f"Critical error in batch processing: {e}")
            # Mark all videos in this batch as failed
            for video in batch_info:
                failed_tasks.append(video)
                print(f"Marking video {video} as failed due to batch error")

    print("all vids failed tasks:", failed_tasks)
    return failed_tasks

async def all_vids_main(vid_dir):
    await all_vids(vid_dir)

async def total_main(vid_dir):
    # Start the background embedding task
    embed_task = asyncio.create_task(batch_embed_query_async('embed_queries.json', 'ret_embeddings.json', 'openai'))

    
    # Run the main video processing
    try:
        await all_vids_main(vid_dir)
    finally:
        from token_tracker import flush_csv
        flush_csv("token_usage.csv")
        # Cancel the embedding task when done
        embed_task.cancel()
        try:
            await embed_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_dir", type=str, default="videos")
    args = parser.parse_args()
    vid_dir = args.vid_dir
    
    open('embed_queries.json', 'w').close()
    open('ret_embeddings.json', 'w').close()
    with open('embed_queries.json', 'w') as f:
        json.dump({}, f, indent=2)
    with open('ret_embeddings.json', 'w') as f:
        json.dump({}, f, indent=2)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(total_main(vid_dir))
    finally:
        from token_tracker import flush_csv
        flush_csv("token_usage.csv")
        loop.close()