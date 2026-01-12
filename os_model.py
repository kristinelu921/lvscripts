from model_example_query import query_vlm, query_llm, query_llm_async
from search_frame_captions import batch_embed_query_async, search_captions
import math
import json
import os
from together import AsyncTogether, Together
#from google import genai
import asyncio
json_file_lock = asyncio.Lock()

# Import prompts - will be conditionally switched based on use_no_vlm parameter
import prompts as default_prompts
import prompts_no_vlm

def get_prompts_module(use_no_vlm=False):
    """Get the appropriate prompts module based on configuration"""
    return prompts_no_vlm if use_no_vlm else default_prompts

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

async def query_model_iterative_with_retry(model, question, uid, vid_path, output_file, max_retries=15, candidates=None, use_no_vlm=False):
    """Wrapper to retry query_model_iterative if it hangs"""
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
                query_model_iterative(model, question, uid, vid_path, candidates, use_no_vlm=use_no_vlm),
                timeout=180  # 3 minute timeout
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
                    "reasoning": f"Failed to complete after {max_retries, output_file: str = 'collected_answers.json} attempts due to timeout",
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
    
async def query_model_iterative(model, question, question_uid, vid_path, candidates=None, use_no_vlm=False, pre_existing_messages=None):
    """Iteratively query any open-source model to answer questions about video

    Args:
        question: The question to answer
        question_uid: Unique identifier for the question
        vid_path: Path to the video directory
        candidates: List of answer choices (optional)
        use_no_vlm: Whether to use no-VLM mode
        pre_existing_messages: Optional list of previous messages to continue conversation from
    """
    global_sum_path = vid_path + "/captions/global_summary.txt"
    CES_logs_path = vid_path + "/captions/CES_logs.txt"
    with open(global_sum_path, "r") as f:
        global_summary = f.read()

    with open(CES_logs_path, "r") as f:
        CES_log = f.read()

    question = question.strip()

    # Variable to store criteria extracted from initial response
    question_criteria = None

    # Get the appropriate prompts module based on configuration
    prompts = get_prompts_module(use_no_vlm)

    vlm_note = " You should be able to extract detailed frame-information from videos, do caption searches, and use your findings to answer the question." if not use_no_vlm else " You can do caption searches to find relevant frames, and use your findings to answer the question based on the semantic similarity of captions."

    # If pre-existing messages provided, use them; otherwise start fresh
    if pre_existing_messages:
        model.messages = list(pre_existing_messages)  # Copy the messages
        print(f"✓ Loaded {len(pre_existing_messages)} pre-existing messages for judge context")
    else:
        model.messages.append({"role": "system", "content": f"You are an expert at reasoning and tool-using, with the goal of answering this question about a long video.{vlm_note} You should be SUPER PICKY about your findings, NOT make assumptions, and always bias towards gathering more evidence before executing a final answer. Use EXACT evidence only. ALSO, when dealing with TEMPORAL questions, you cannot find VISUAL TIMES. If a question asks for a 'duration' of an event, you want to do many caption searches on consecutive ranges, and find scene-changes at the beginning and end of the event. You MUST CHOOSE AN ANSWER. NONE OF THE ABOVE IS NOT ACCEPTABLE."})
        model.messages.append({"role": "user", "content": "\n Here is a global summary of the video for general context: " + global_summary + "\n\n Here is also an INCOMPLETE character/event/scene log across the video. These will all be encountered, and there MAY BE MORE " + CES_log +
"\nYour question is this: " + question})
    prompt = str(model.messages) + prompts.initial_prompt(question, candidates)
    message = prompt
    log(message, f"logs/log_video_{vid_path}_{question_uid}")

    
    for i in range(model.max_num_iterations):
        # Query the specified model
        print("="*20 + f"Querying model with prompt: {i}, {question_uid} "+ "="*20)
        print(f"Prompt length: {len(prompt)} characters")
        try:   
            #print("reached this thing")
            response = await model.llm_query_async(prompt)
            #print("response reached", response)
        except Exception as e:
            print(f"Failed to get model response: {e}")
            print(f"Error type: {type(e).__name__}")
            continue
        
        model.messages.append({"role": "assistant", "content": response})
        
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
        try:
            if parsed_response.get("tool") == "FINAL_ANSWER":
                # The parsed response has "frames" field, not "evidence_frame_numbers"
                message = f"FINAL_ANSWER: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                parsed_answer = parsed_response.get("answer")
                if parsed_answer in "012345":
                    parsed_answer = int(parsed_answer)
                elif parsed_answer in "ABCDE":  
                    parsed_answer = ord(parsed_answer) - ord('A')
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
                prompt = "Here is a global summary of the video for general context: " + global_summary + "\n" + parsed_response.get("prompt")
                print("PROMPT: ", prompt)
                frames = parsed_response.get("frames")
                new_frames = [(f"{vid_path}/" + frame) for frame in frames]
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
                for idx, query in enumerate(search_queries):
                    print(f"  Query {idx+1}/{len(search_queries)}: {query}")
                    results = await search_captions(vid_path, question_uid, query, f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl", 30)

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

                # Confirm to LLM
                retrieved_info = f"Successfully recorded {len(entries)} event(s). You now have {len(model.records)} total recorded events. Use VIEW_RECORDS to see all of them sorted by time."
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

            else:
                message = f"Invalid or unrecognized tool: {parsed_response.get('tool')}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print(f"Invalid or unrecognized tool: {parsed_response.get('tool')}")
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
                retrieved_info = "The following is the retrieved information from the VLM query: Please read through and see if these are the scenes you're looking for. If not, please look for different scenes. If yes, extract detailed and important evidence from them.\n" + str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "RECORD":
                retrieved_info = str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "VIEW_RECORDS":
                retrieved_info = str(model.messages[-1].get("content", ""))
            else:
                retrieved_info = str(model.messages[-1].get("content", ""))

            prompt = prompts.followup_prompt(retrieved_info, question, candidates)
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

async def answer_question(question_uid, question, vid_folder, vid_num, candidates=None, vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", llm_model="deepseek-ai/DeepSeek-V3.1", use_no_vlm=False):
    try:
        # Create a separate Pipeline instance for each question to avoid shared state
        #qwen model : Qwen/Qwen3-235B-A22B-Instruct-2507-tput
        model = Pipeline(llm_model, vlm_model)
        curr_folder = str(vid_folder)
        num = vid_num
        vid_path = curr_folder + "/" + num
        print("vid_path", vid_path) #TODO: remove this
        answers_path = f'{curr_folder}/{num}/{num}_answers.json'
        answer = await query_model_iterative_with_retry(model, question, question_uid, vid_path, answers_path, candidates=candidates, use_no_vlm=use_no_vlm)
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