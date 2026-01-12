
#!/usr/bin/env python3
"""
Critic Response Pipeline
Re-evaluates answers with confidence < 70 using the critic's feedback
"""

import json
import asyncio
import os
from os_model import Pipeline
from model_example_query import query_vlm, query_llm, query_llm_async
from search_frame_captions import search_captions, batch_embed_query_async
from prompts import initial_prompt, followup_prompt, response_parsing_prompt, finish_prompt, reformat_answers
json_file_lock = asyncio.Lock()
import traceback

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

        message = f"saved answer {data.get('uid', 'unknown')}!"
        print(f"saved answer {data.get('uid', 'unknown')}!")
        return

# Pipeline class is imported from os_model.py (line 11)
# No need to redefine it here

# Load environment variables
with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key = env_data["together_key"]

os.environ['TOGETHER_API_KEY'] = together_key


async def query_model_iterative_with_retry(model, question, uid, vid_path, candidates=None, max_retries=15):
    """Wrapper to retry query_model_iterative if it hangs"""
    print(f"QUESTION uid {uid} being called")
    # Extract video ID from path (e.g., /path/to/yU9fGAEcxJY -> yU9fGAEcxJY)
    num = os.path.basename(vid_path)
    output_file = f"{vid_path}/{num}_re_evaluated.json"

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
            print(f"Attempt {attempt + 1}/{max_retries} for question: {str(question)[:50]}...")
            # Set 60 second timeout for the entire iterative process
            result = await asyncio.wait_for(
                query_model_iterative(model, question, uid, vid_path, candidates=candidates),
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
def create_enhanced_prompt(assessment, candidates=None):
    """
    Create an enhanced prompt incorporating critic feedback

    Args:
        assessment: Critic assessment dictionary
        candidates: Optional list of answer choices

    Returns:
        Enhanced question string with critic context
    """
    question = assessment["question"]

    # Add candidates to question if provided
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"
        question_with_candidates = question + candidates_text
    else:
        question_with_candidates = question

    # Build enhanced prompt with critic insights
    enhanced = f"""{question_with_candidates}
IMPORTANT CONTEXT FROM PREVIOUS ANALYSIS:
- Previous answer: {assessment.get('answer', 'Unknown')}
- Confidence level: {assessment.get('confidence', -1)}%
"""

    if assessment.get("possible_errors"):
        enhanced += f"\nPotential issues identified:\n"
        for error in assessment["possible_errors"]:
            enhanced += f"  - {error}\n"

    if assessment.get("suggestion"):
        enhanced += f"\nSuggested approach: {assessment['suggestion']}\n"

    if assessment.get("evidence_frame_numbers"):
        enhanced += f"\nKey frames to examine: {', '.join(assessment['evidence_frame_numbers'])}\n"

    enhanced += """
INSTRUCTIONS FOR RE-EVALUATION:
1. Carefully examine the evidence frames mentioned above
2. Address any concerns or potential errors identified
3. Determine if you want to use the suggested approach if provided.
4. Be especially thorough in verifying details
5. Look for different scenes IF NEEDED.
6. Provide clear reasoning for your final answer

⚠️ CRITICAL: There is ALWAYS a correct answer among the choices provided. If all answers seem slightly off or imperfect, you MUST choose the BEST possible answer that most closely matches the evidence. Do not refuse to answer.

IMPORTANT: It is ALSO TOTALLY POSSIBLE that your original answer was CORRECT. Do your best to keep the old reasoning in mind, any new reasoning you have, and compare the two and use your best judgment to determine a final answer.
"""

    return enhanced

async def query_model_iterative(model, question, question_uid, vid_path, candidates=None):
    """Iteratively query any open-source model to answer questions about video

    FOR RE-EVALUATION: model.messages should already contain ALL previous LLM history + critic messages

    Args:
        question: The question to answer (enhanced with critic feedback for re-evaluation)
        question_uid: Unique identifier for the question
        vid_path: Path to the video directory
        candidates: Optional list of answer choices
    """
    message = f"Querying model iterative with question: {question}"
    log(message, f"logs/log_video_{vid_path}_{question_uid}")

    # Check if this is a re-evaluation (model.messages already has history)
    is_reevaluation = len(model.messages) > 0
    global_sum_path = vid_path + "/captions/global_summary.txt"
    CES_logs_path = vid_path + "/captions/CES_logs.txt"
    with open(global_sum_path, "r") as f:
        global_summary = f.read()

    with open(CES_logs_path, "r") as f:
        CES_log = f.read()

    if is_reevaluation:
        print(f"🔄 RE-EVALUATION MODE: Starting with {len(model.messages)} messages from previous conversation + critic")
        message = f"RE-EVALUATION with {len(model.messages)} messages in history"
        log(message, f"logs/log_video_{vid_path}_{question_uid}")

        # Add re-evaluation context to the existing history
        model.messages.append({"role": "system", "content": "🔄 RE-EVALUATION MODE: You now have access to VLM tooling to verify your answer. Review the critic's feedback above, then use VLM_QUERY to visually verify the frames and determine if your original answer was correct or needs revision."})
        model.messages.append({"role": "user", "content": f"\nRe-evaluation request: {question}"})
    else:
        # Original evaluation mode (shouldn't happen in critic_response.py, but keeping for safety)
        print("⚠️ Warning: query_model_iterative called without message history in critic_response.py")

        question = question.strip()

        model.messages.append({"role": "system", "content": "You are an expert at reasoning and tool-using, with the goal of answering this question about a long video. You should be able to extract detailed frame-information from videos, do caption searches, and use your findings to answer the question. You should be SUPER PICKY about your findings, NOT make assumptions, and always bias towards gathering more evidence before executing a final answer. Use EXACT evidence only. ALSO, when dealing with TEMPORAL questions, you cannot find VISUAL TIMES. If a question asks for a 'duration' of an event, you want to do many VLM queries on consecutive ranges, and find scene-changes at the beginning and end of the event. You MUST CHOOSE AN ANSWER. NONE OF THE ABOVE IS NOT ACCEPTABLE."})
        model.messages.append({"role": "user", "content": "\n Here is a global summary of the video for general context: " + global_summary + "\n\n Here is also an INCOMPLETE character/event/scene log across the video. These will all be encountered, and there MAY BE MORE " + CES_log +
    "\nYour question is this: " + question})
    prompt = str(model.messages) + initial_prompt(question, candidates=candidates)
    message = f"Prompt: {prompt}"
    log(message, f"logs/log_video_{vid_path}_{question_uid}")
    for i in range(model.max_num_iterations):
        # Query the specified model
        print("="*20 + f"Querying model with prompt: {i}, {question_uid} "+ "="*20)
        print(f"Prompt length: {len(prompt)} characters")
        try:   
            #print("reached this thing")
            response = await model.llm_query_async(prompt)
            message = f"Response: {response}"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")
            #print("response reached", response)
        except Exception as e:
            print(f"Failed to get model response: {e}")
            print(f"Error type: {type(e).__name__}")
            continue
        
        model.messages.append({"role": "assistant", "content": response})
        
        if not response:
            print(f"Failed to get response at iteration {i+1}")
            continue
            
        # Parse the response
        if response and "</think>" in response:
            response = response.split("</think>")[1].strip()
        
        parsing_prompt = response_parsing_prompt(response)
        parsed_response = await model.llm_query_async(parsing_prompt)
        message = f"Parsed response: {parsed_response}"
        log(message, f"logs/log_video_{vid_path}_{question_uid}")
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
            print(f"Raw response: {parsed_response[:50]}...")
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

                # Include answer-specific criteria if present in response
                if "answer_criteria" in parsed_response:
                    new_response["answer_criteria"] = parsed_response.get("answer_criteria", [])

                return new_response
            elif parsed_response.get("tool") == "VLM_QUERY":
                print("parsed response: ", str(parsed_response)[:50] + "...")
                print("="*10 + "Querying VLM" + "="*10)
                prompt = "Here is a global summary of the video for general context: " + global_summary + "\n" + parsed_response.get("prompt")
                message = f"PROMPT for VLM query: {prompt}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print("PROMPT: ", prompt[:50] + "...")
                frames = parsed_response.get("frames")
                new_frames = [(f"{vid_path}/" + frame) for frame in frames]
                retrieved_info = await model.vlm_query(new_frames, prompt)
                model.messages.append({"role": "vlm response", "content": retrieved_info})
                message = f"VLM response: {retrieved_info}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
            elif parsed_response.get("tool") == "CAPTION_SEARCH":
                # Handle multiple search queries or single query
                message = f"CAPTION_SEARCH: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print("reaching parsed response get tool caption search")

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
                model.messages.append({"role": "caption search results", "content": retrieved_info[:500]})
            else:
                print(f"Invalid or unrecognized tool: {parsed_response.get('tool')}")
                continue
            # Update prompt for next iteration
            if parsed_response.get("tool") == "CAPTION_SEARCH":
                num_queries = len(search_queries) if 'search_queries' in locals() else 1
                if num_queries > 1:
                    prompt = f"""
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
""" + "\n"
                else:
                    prompt = "The following is the retrieved information from the caption search: Please read through and choose the most relevant few keyframes." + "\n"
            elif parsed_response.get("tool") == "VLM_QUERY":
                prompt = "The following is the retrieved information from the VLM query: Please read through and see if these are the scenes you're looking for. If not, please look for different scenes. If yes, extract detailed and important evidence from them." + "\n"
            prompt = followup_prompt(model.messages, question, candidates=candidates)
        except Exception as e:
            message = f"Error updating prompt: {e}"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")
            print(f"Error updating prompt: {e}")
            traceback.print_exc()
            continue
    
    # Return final formatted scratchpad
    final_prompt = finish_prompt(model.messages, candidates=candidates)
    final_answer = await model.llm_query_async(final_prompt)
    
    if not final_answer:
        print(f"Failed to get final answer for question {question_uid}")
        message = f"Failed to get final answer for question {question_uid}"
        log(message, f"logs/log_video_{vid_path}_{question_uid}")
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
                message = f"Final answer: {final_answer}"
                answer = final_answer
                if final_answer in "012345":
                    answer = int(final_answer)
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                return {
                    "uid": question_uid,
                    "question": question,
                    "answer": answer,
                    "reasoning": "Final iteration response",
                    "evidence_frame_numbers": []
                }
            answer = parsed_final.get("answer")
            if answer in "012345":
                answer = int(answer)
            elif answer in "ABCDE":
                answer = ord(answer) - ord('A')
            return {
                "uid": question_uid,
                "question": question,
                "answer": answer,
                "reasoning": parsed_final.get("reasoning", ""),
                "evidence_frame_numbers": parsed_final.get("frames", [])
            }
    except:
        pass

    answer = final_answer
    if final_answer in "012345":
        answer = int(final_answer)
    elif final_answer in "ABCDE":
        answer = ord(final_answer) - ord('A')

    return {
        "uid": question_uid,
        "question": question,
        "answer": answer,
        "reasoning": "Could not parse final response",
        "evidence_frame_numbers": []
    }

async def re_evaluate_low_confidence_answers(
    vid_dir,
    num,
    confidence_threshold=70,
    llm_model="deepseek-ai/DeepSeek-V3.1",
    vlm_model="Qwen/Qwen2.5-VL-72B-Instruct"
):
    """
    Re-evaluate answers with low confidence scores
    
    Args:
        critic_file: Path to critic assessment JSON
        output_file: Path to save re-evaluated answers
        confidence_threshold: Re-evaluate if confidence is below this (default: 50)
        llm_model: LLM model to use for re-evaluation
        vlm_model: VLM model to use for frame analysis
    """
    
    # Load critic assessments

    critic_file = f"{vid_dir}/{num}/{num}_critic_assessment.json"
    print("critic file", critic_file)
    output_file = f"{vid_dir}/{num}/{num}_re_evaluated.json"
    if not os.path.exists(critic_file):
        with open('failure_log', 'a') as f:
            f.write('no critic file for {video_dir}/{num}')
        return

    with open(critic_file, 'r') as f:
        critic_assessments = json.load(f)

    # Load original questions with candidates
    questions_file = f"/mnt/ssh/data/longvideobench/downloaded_videos_questions.json"
    candidates_by_uid = {}
    if os.path.exists(questions_file):
        with open(questions_file, 'r') as f:
            all_questions = json.load(f)
            # The file structure is {video_id: [questions]}
            for video_id, questions in all_questions.items():
                for q in questions:
                    candidates_by_uid[q['uid']] = q.get('candidates', [])

    # Identify questions needing re-evaluation
    questions_to_reevaluate = []
    for assessment in critic_assessments:
        if assessment.get("confidence", 100) < confidence_threshold:
            # Add candidates to assessment for easy access
            assessment['candidates'] = candidates_by_uid.get(assessment.get('uid'), [])
            questions_to_reevaluate.append(assessment)
    
    print(f"Found {len(questions_to_reevaluate)} questions with confidence < {confidence_threshold}")
    
    # Re-evaluate each low-confidence question
    re_evaluated_results = []
    vid_path = f"{vid_dir}/{num}"
        
    # Create enhanced prompt with critic feedback
    max_concurrent = 10
    # Initialize pipeline for this question
    semaphore = asyncio.Semaphore(max_concurrent)
    async def solve_question(assessment):
        message = f"Solving question {assessment.get('uid')}"
        log(message, f"logs/log_video_{vid_path}_{assessment.get('uid')}")
        async with semaphore:
            for i in range(3):
                # Get candidates from the assessment
                candidates = assessment.get('candidates', [])
                enhanced_question = create_enhanced_prompt(assessment, candidates=candidates)
                message = f"Enhanced question: {enhanced_question}"
                log(message, f"logs/log_video_{vid_path}_{assessment.get('uid')}")

                # Create model and load ALL conversation history
                model = Pipeline(llm_model, vlm_model, max_num_iterations=10)

                # Load ALL LLM history from original conversation
                llm_history_path = f"{vid_path}/{assessment.get('uid')}_os_model.json"
                if os.path.exists(llm_history_path):
                    try:
                        with open(llm_history_path, 'r') as f:
                            llm_history = json.load(f)
                        print(f"✅ Loaded {len(llm_history)} messages from LLM history")
                        model.messages.extend(llm_history)
                        message = f"Loaded {len(llm_history)} LLM messages"
                        log(message, f"logs/log_video_{vid_path}_{assessment.get('uid')}")
                    except Exception as e:
                        print(f"⚠️ Could not load LLM history: {e}")
                else:
                    print(f"⚠️ LLM history not found at {llm_history_path}")

                # Load ALL critic messages
                critic_history_path = f"{vid_path}/{assessment.get('uid')}_critic_model.json"
                if os.path.exists(critic_history_path):
                    try:
                        with open(critic_history_path, 'r') as f:
                            critic_history = json.load(f)
                        print(f"✅ Loaded {len(critic_history)} messages from critic history")
                        # Add a separator message to clearly mark critic feedback
                        model.messages.append({"role": "system", "content": f"📋 CRITIC FEEDBACK (Confidence: {assessment.get('confidence')}%)"})
                        model.messages.extend(critic_history)
                        message = f"Loaded {len(critic_history)} critic messages"
                        log(message, f"logs/log_video_{vid_path}_{assessment.get('uid')}")
                    except Exception as e:
                        print(f"⚠️ Could not load critic history: {e}")
                else:
                    print(f"⚠️ Critic history not found at {critic_history_path}")

                print(f"🔄 Re-evaluation starting with {len(model.messages)} total messages (LLM + Critic)")
                #print("ENHANCED question: ", enhanced_question)
                try:
                    # Re-evaluate with enhanced context
                    result = await query_model_iterative_with_retry(
                        model,
                        enhanced_question,
                        assessment.get("uid", "unknown"),
                        vid_path,
                        candidates=candidates
                    )
                    message = f"Result: {result}"
                    log(message, f"logs/log_video_{vid_path}_{assessment.get('uid')}")
                    print(f"reached RESULT done awaiting {str(result)[:50]}...")
                    if isinstance(result, str):
                        print(f"ALREADY SOLVED question {assessment.get('uid')}")
                        # Return the original assessment since it was already done
                        return {
                            "uid": assessment.get("uid"),
                            "question": assessment.get("question"),
                            "answer": assessment.get("answer"),
                            "confidence": assessment.get("confidence"),
                            "reasoning": "Previously evaluated - keeping original answer",
                            "re_evaluated": False,
                            "original_confidence": assessment.get("confidence"),
                            "critic_concerns": assessment.get("possible_errors", []),
                            "critic_suggestion": assessment.get("suggestion"),
                            "critic_evidence": assessment.get("evidence_frame_numbers", [])
                        }
                
                    # Add metadata about re-evaluation
                    result["uid"] = assessment.get("uid")
                    result["original_answer"] = assessment.get("answer")
                    result["original_confidence"] = assessment.get("confidence")
                    # Log per-question
                    try:
                        with open("answers_logs.json", "a") as log_f:
                            log_f.write(f"critic response re-evaluated uid {assessment.get('uid')} video {num}\n")
                    except Exception:
                        pass
                    result["critic_concerns"] = assessment.get("possible_errors", [])
                    result["critic_suggestion"] = assessment.get("suggestion")
                    result["critic_evidence"] = assessment.get("evidence_frame_numbers", [])
                    result["re_evaluated"] = True

                    # Add judge decision if it exists
                    if "judge_decision" in assessment:
                        result["judge_decision"] = assessment.get("judge_decision")
                        result["judge_reasoning"] = assessment.get("judge_reasoning", "")
                        result["was_reevaluated_by_judge"] = True
                        result["critic_suggested_answer"] = assessment.get("critic_answer_choice", -1)
                    else:
                        result["was_reevaluated_by_judge"] = False
                    
                    # Save critic-response conversation
                    try:
                        conv_path = f"{vid_dir}/{num}/{assessment.get('uid')}_critic_response.json"
                        with open(conv_path, "w") as conv_f:
                            json.dump(model.messages, conv_f, indent=2)
                    except Exception:
                        pass

                    return result
                except Exception as e:
                    print(f"ERROR in solve_question for uid {assessment.get('uid')}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return error result instead of continuing silently
                    continue
            
            # All attempts failed, return error result
            return {
                "uid": assessment.get("uid"),
                "question": assessment.get("question"),
                "answer": assessment.get("answer"),
                "confidence": -1,
                "reasoning": "Failed to re-evaluate after 3 attempts",
                "re_evaluated": False,
                "error": "All re-evaluation attempts failed"
            }
    
    print("questions", questions_to_reevaluate)
    tasks = [solve_question(assessment) for assessment in questions_to_reevaluate]
    results = await asyncio.gather(*tasks, return_exceptions = True)

    for uid, result in zip([question['uid'] for question in questions_to_reevaluate], results):
        if isinstance(result, Exception):
            print(f"uid {uid} has result {result}")
        else:
            print (f"{uid} passes")

    # Combine with high-confidence answers
    final_results = []
    
    # Add high-confidence answers (no re-evaluation needed)
    for assessment in critic_assessments:
        if assessment.get("confidence", 100) >= confidence_threshold:
            assessment["re_evaluated"] = False
            final_results.append(assessment)
    
    # Add re-evaluated answers
    final_results.extend(results)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    with open("answers_logs.json", "a") as f:
        f.write(f"saved critic response final results to {output_file}\n")

    print(f"\n{'='*80}")
    print(f"Re-evaluation complete!")
    print(f"Total questions: {len(critic_assessments)}")
    print(f"Re-evaluated: {len(results)}")
    print(f"Results saved to: {output_file}")

    
    return final_results

async def all_vids(vid_folder, batch_size = 1):
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)
    print(curr_paths)
    all_tasks = []
    task_info = []

    #curr_paths = ["00000031"]

    for num in curr_paths:
        all_tasks.append(re_evaluate_low_confidence_answers(vid_folder, num))
        task_info.append(num)
        
    total_tasks = len(all_tasks)
    print("LEN ALL TASKS", total_tasks)
    failed_tasks = []
    for i in range(0, total_tasks, batch_size):
        batch_tasks = all_tasks[i:i+batch_size]
        batch_info = task_info[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_tasks + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} (videos: {', '.join(batch_info)})")

        try:
            completed = await asyncio.gather(*batch_tasks, return_exceptions=True)
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
        loop.close()