from model_example_query import query_vlm, query_llm, query_llm_async
from search_frame_captions import batch_embed_query_async, search_captions
from prompts import initial_prompt, followup_prompt, response_parsing_prompt, finish_prompt, reformat_answers
import math
import json
import os
from together import AsyncTogether, Together
#from google import genai
import asyncio
json_file_lock = asyncio.Lock()

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
    gemini_key_PRIV = env_data["gemini_key"]

os.environ['TOGETHER_API_KEY'] = together_key_PRIV
os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
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
    
    
    def llm_query(self, prompt):
        return query_llm(self.llm, prompt)
    
    async def llm_query_async(self, prompt):
        return await query_llm_async(self.llm, prompt)
    
    async def vlm_query(self, image_paths, prompt):
        result = await query_vlm(self.vlm, image_paths, prompt)
        return result
        

my_model = Pipeline("deepseek-ai/DeepSeek-V3.1", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

async def query_model_iterative_with_retry(model, question, uid, vid_path, output_file, max_retries=15):
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
                    return f"Already Completed Question {uid}"
        except Exception as e:
            print("Checking if q already completed error")
            pass
    
    for attempt in range(max_retries):
        try:
            message = f"Attempt {attempt + 1}/{max_retries} for question: {question[:50]}..."
            log(message, f"logs/log_video_{vid_path}_{uid}")
            # Set 60 second timeout for the entire iterative process
            result = await asyncio.wait_for(
                query_model_iterative(model, question, uid, vid_path),
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
    
async def query_model_iterative(model, question, question_uid, vid_path):
    """Iteratively query any open-source model to answer questions about video
    
    Args:
        question: The question to answer
        model_name: The model to use (default: DeepSeek R1)
        max_num_iterations: Maximum iterations for reasoning
    """
    global_sum_path = vid_path + "/captions/global_summary.txt"
    CES_logs_path = vid_path + "/captions/CES_logs.txt"
    with open(global_sum_path, "r") as f:
        global_summary = f.read()
    
    with open(CES_logs_path, "r") as f:
        CES_log = f.read()
    
    question = question.strip()
    model.messages.append({"role": "system", "content": "You are an expert at reasoning and tool-using, with the goal of answering this question about a long video. You should be able to extract detailed frame-information from videos, do caption searches, and use your findings to answer the question. You should be SUPER PICKY about your findings, NOT make assumptions, and always bias towards gathering more evidence before executing a final answer. Use EXACT evidence only. ALSO, when dealing with TEMPORAL questions, you cannot find VISUAL TIMES. If a question asks for a 'duration' of an event, you want to do many VLM queries on consecutive ranges, and find scene-changes at the beginning and end of the event. You MUST CHOOSE AN ANSWER. NONE OF THE ABOVE IS NOT ACCEPTABLE."})
    model.messages.append({"role": "user", "content": "\n Here is a global summary of the video for general context: " + global_summary + "\n\n Here is also an INCOMPLETE character/event/scene log across the video. These will all be encountered, and there MAY BE MORE " + CES_log +
"\nYour question is this: " + question})
    prompt = str(model.messages) + initial_prompt(question)
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
        
        parsing_prompt = response_parsing_prompt(response)
        parsed_response = await model.llm_query_async(parsing_prompt)

        if not parsed_response:
            print(f"Failed to get parsing response at iteration {i+1}")
            continue
        
        print("reached here")
        
        def extract_json(text):
            if not text:
                return None
                
            # Strategy 1: Try direct parsing
            try:
                return json.loads(text)
            except:
                pass
            
            # Strategy 2: Extract from markdown code blocks
            if "```json" in text:
                try:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except:
                    pass
            
            # Strategy 3: Extract from any code blocks
            if "```" in text:
                try:
                    json_str = text.split("```")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except:
                    pass
            
            # Strategy 4: Find JSON object boundaries
            try:
                start = text.find("{")
                if start != -1:
                    # Find matching closing brace
                    count = 0
                    for i in range(start, len(text)):
                        if text[i] == "{":
                            count += 1
                        elif text[i] == "}":
                            count -= 1
                            if count == 0:
                                json_str = text[start:i+1]
                                return json.loads(json_str)
            except:
                pass
            
            # Strategy 5: Clean and retry
            try:
                # Remove common problematic patterns
                cleaned = text.strip()
                if cleaned.startswith("json_output = "):
                    cleaned = cleaned[14:]
                cleaned = cleaned.strip().rstrip('\\').strip()
                return json.loads(cleaned)
            except:
                pass
            
            return None
        
        parsed_response = extract_json(parsed_response)
        if parsed_response is None:
            print(f"Failed to extract JSON from response")
            print(f"Raw response: {parsed_response}")
            continue
        try:
            if parsed_response.get("tool") == "FINAL_ANSWER":
                # The parsed response has "frames" field, not "evidence_frame_numbers"
                message = f"FINAL_ANSWER: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                new_response = {
                    "uid": question_uid,
                    "question": question, 
                    "answer": parsed_response.get("answer"), 
                    "reasoning": parsed_response.get("reasoning"), 
                    "evidence_frame_numbers": parsed_response.get("frames")  # Map "frames" to "evidence_frame_numbers"
                }
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
                # Handle both 'input' and 'prompt' fields
                print("reaching parsed response get tool caption search")
                search_query = parsed_response.get("input") or parsed_response.get("prompt")
                if isinstance(search_query, list):
                    search_query = search_query[0]
                if not search_query:
                    print("Warning: No search query found in CAPTION_SEARCH")
                    continue
                print("search query from json: ", search_query)
                message = f"Search query: {search_query}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                retrieved_info = await search_captions(vid_path, question_uid, search_query, f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl", 30)
                # Convert list results to string
                if isinstance(retrieved_info, list):
                    retrieved_info = json.dumps(retrieved_info, indent=2)
                message = f"Caption search results: {retrieved_info}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                model.messages.append({"role": "caption search results", "content": retrieved_info})

            else:
                message = f"Invalid or unrecognized tool: {parsed_response.get('tool')}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print(f"Invalid or unrecognized tool: {parsed_response.get('tool')}")
                continue

            # Update prompt for next iteration
            if parsed_response.get("tool") == "CAPTION_SEARCH":
                prompt = "The following is the retrieved information from the caption search: Please read through and choose the most relevant few keyframes." + "\n"
            elif parsed_response.get("tool") == "VLM_QUERY":
                prompt = "The following is the retrieved information from the VLM query: Please read through and see if these are the scenes you're looking for. If not, please look for different scenes. If yes, extract detailed and important evidence from them." + "\n"
            prompt = followup_prompt(model.messages, question)
        except Exception as e:
            message = f"Error updating prompt: {e}"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")
            print(f"Error updating prompt: {e}")
            continue
    
    # Return final formatted scratchpad
    final_prompt = finish_prompt(model.messages)
    final_answer = await model.llm_query_async(final_prompt)
    
    if not final_answer:
        print(f"Failed to get final answer for question {question_uid}")
        return {
            "uid": question_uid,
            "question": question,
            "answer": "ERROR",
            "reasoning": "Failed to get final answer from LLM",
            "evidence_frame_numbers": []
        }

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
                return {
                    "uid": question_uid,
                    "question": question,
                    "answer": final_answer,
                    "reasoning": "Final iteration response",
                    "evidence_frame_numbers": []
                }
            with open(f"{vid_path}/{question_uid}_os_model.json", "w") as f:
                    json.dump(model.messages, f, indent=2)
                    with open(f"answers_logs.json", "a") as f:
                        f.write(f"saved model messages for question {question_uid}, video {vid_path}\n")
            return {
                "uid": question_uid,
                "question": question,
                "answer": parsed_final.get("answer", ""),
                "reasoning": parsed_final.get("reasoning", ""),
                "evidence_frame_numbers": parsed_final.get("frames", [])
            }
    except:
        pass
    
    message = model.messages
    log(message, f"logs/log_video_{vid_path}_{question_uid}")
    
    return {
        "uid": question_uid,
        "question": question,
        "answer": final_answer,
        "reasoning": "Could not parse final response",
        "evidence_frame_numbers": []
    }

async def answer_question(question_uid, question, vid_folder, vid_num):
    try:
        # Create a separate Pipeline instance for each question to avoid shared state
        #qwen model : Qwen/Qwen3-235B-A22B-Instruct-2507-tput
        model = Pipeline("openai/gpt-oss-120b", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
        curr_folder = vid_folder
        num = vid_num
        vid_path = curr_folder + "/" + num
        answers_path = f'{curr_folder}/{num}/{num}_answers.json'
        print("error is in query model iterative with retry")
        answer = await query_model_iterative_with_retry(model, question, question_uid, vid_path, answers_path)
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