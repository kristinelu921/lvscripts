from model_example_query import query_vlm, query_llm, query_llm_async
from prompts import critic_vlm_prompt, critic_followup_prompt
from critic_response import total_main


import traceback
import json
import os
from together import Together, AsyncTogether
import asyncio
import math

def log(message, file_title):
    if not os.path.exists(file_title):
        os.makedirs(file_title)
    else:
        with open(f"{file_title}/log.log", "a") as f:
            f.write(message + "\n")

with open("env.json", "r") as f:
    env_data = json.load(f)
    together_key_PRIV = env_data["together_key"]

os.environ['TOGETHER_API_KEY'] = together_key_PRIV
client_together = Together()
async_client_together = AsyncTogether(api_key=together_key_PRIV)

# Load subtitle mappings globally
SUBTITLE_MAPPING_PATH = "/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json"
_SUBTITLE_CACHE = None

def load_subtitle_mapping():
    """Load subtitle frame mapping from JSON file"""
    global _SUBTITLE_CACHE
    if _SUBTITLE_CACHE is None:
        try:
            if os.path.exists(SUBTITLE_MAPPING_PATH):
                with open(SUBTITLE_MAPPING_PATH, 'r') as f:
                    _SUBTITLE_CACHE = json.load(f)
                print(f"✓ Loaded subtitle mappings for {len(_SUBTITLE_CACHE)} videos")
            else:
                print(f"⚠ Subtitle mapping not found at {SUBTITLE_MAPPING_PATH}")
                _SUBTITLE_CACHE = {}
        except Exception as e:
            print(f"Error loading subtitle mapping: {e}")
            _SUBTITLE_CACHE = {}
    return _SUBTITLE_CACHE

def get_subtitles_for_frames(video_id, frame_numbers):
    """Get subtitles for specific frame numbers

    Args:
        video_id: Video ID (e.g., "abc123")
        frame_numbers: List of frame paths like ["frames/frame_0123.jpg", ...]

    Returns:
        Dict mapping frame_path -> subtitle_text
    """
    subtitle_mapping = load_subtitle_mapping()
    video_subtitles = subtitle_mapping.get(video_id, {}).get('frames', {})

    result = {}
    for frame_path in frame_numbers:
        # Extract frame number from path like "frames/frame_0123.jpg" -> 123
        import re
        match = re.search(r'frame_(\d+)', frame_path)
        if match:
            frame_num = str(int(match.group(1)))  # Remove leading zeros
            if frame_num in video_subtitles:
                result[frame_path] = video_subtitles[frame_num]

    return result

class CriticPipeline:
    def __init__(self, llm_model_name, vlm_model_name, max_num_iterations=15):
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name

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
        return await query_vlm(self.vlm, image_paths, prompt)

async def third_party_judge(model, question, candidates, original_answer, critic_answer, frames, vid_dir, uid, original_reasoning="", critic_reasoning="", original_messages=None, critic_messages=None):
    """Third-party judge using full pipeline capabilities to decide between conflicting answers

    Can use caption search and VLM query tools to investigate before deciding.
    Now includes FULL conversation history from original model + critic for complete context.

    Args:
        model: CriticPipeline instance
        question: The original question
        candidates: List of answer choices
        original_answer: Original LLM's answer (as number 0-3)
        critic_answer: Critic's answer choice (as number 0-3)
        frames: Frame paths used as evidence
        vid_dir: Video directory
        uid: Question UID
        original_reasoning: Original model's reasoning
        critic_reasoning: Critic's reasoning
        original_messages: Original model's conversation history (optional)
        critic_messages: Critic model's conversation history (optional)

    Returns:
        Dictionary with: final_answer (number), reasoning (string)
    """
    from os_model import query_model_iterative

    candidates_text = "\n\nAnswer Choices:\n"
    for i, choice in enumerate(candidates):
        candidates_text += f"{chr(65+i)}. {choice}\n"

    original_letter = chr(65 + original_answer) if 0 <= original_answer <= 4 else "?"
    critic_letter = chr(65 + critic_answer) if 0 <= critic_answer <= 4 else "?"

    # Try to load original model's conversation history if not provided
    if original_messages is None:
        original_model_path = f"{vid_dir}/{uid}_model.json"
        if os.path.exists(original_model_path):
            try:
                with open(original_model_path, 'r') as f:
                    original_messages = json.load(f)
                print(f"✓ Loaded {len(original_messages)} messages from original model for judge context")
            except Exception as e:
                print(f"⚠ Could not load original model messages: {e}")
                original_messages = []
        else:
            print(f"⚠ Original model conversation not found at {original_model_path}")
            original_messages = []

    # Try to load critic's conversation history if not provided
    if critic_messages is None:
        critic_model_path = f"{vid_dir}/{uid}_critic_model.json"
        if os.path.exists(critic_model_path):
            try:
                with open(critic_model_path, 'r') as f:
                    critic_messages = json.load(f)
                print(f"✓ Loaded {len(critic_messages)} messages from critic model for judge context")
            except Exception as e:
                print(f"⚠ Could not load critic messages: {e}")
                critic_messages = []
        else:
            critic_messages = []

    # Combine conversation histories: original + critic + judge instruction
    combined_messages = []
    if original_messages:
        combined_messages.extend(original_messages)
    if critic_messages:
        # Add a separator message
        combined_messages.append({
            "role": "system",
            "content": "--- CRITIC MODEL ASSESSMENT BEGINS ---"
        })
        combined_messages.extend(critic_messages)

    # Add judge instruction as final user message
    judge_instruction = f"""
--- JUDGE TASK BEGINS ---

Two models disagreed on the answer to this question. You now have the FULL conversation history from both.

Original Question: {question}{candidates_text}

DISAGREEMENT:
- Original Model chose: {original_letter}
  Reasoning: {original_reasoning}

- Critic Model chose: {critic_letter}
  Reasoning: {critic_reasoning}

YOUR TASK AS JUDGE:
Review the ENTIRE conversation history above. You can see:
1. How the original model investigated and reasoned
2. How the critic evaluated the evidence

Now investigate further if needed using caption search and VLM tools, then decide which answer ({original_letter} or {critic_letter}) is MOST CORRECT based on all evidence.

⚠️ CRITICAL: There is ALWAYS a correct answer. Choose the BEST possible answer that most closely matches the evidence.

You MUST choose between ONLY these two options:
A. {original_letter} (Original model's answer)
B. {critic_letter} (Critic model's answer)
"""

    combined_messages.append({
        "role": "user",
        "content": judge_instruction
    })

    # Create simplified candidates for judge
    judge_candidates = [
        f"{original_letter} - {candidates[original_answer] if 0 <= original_answer < len(candidates) else 'Original answer'}",
        f"{critic_letter} - {candidates[critic_answer] if 0 <= critic_answer < len(candidates) else 'Critic answer'}"
    ]

    try:
        # Reuse the full iterative pipeline for the judge WITH FULL CONTEXT
        judge_result = await query_model_iterative(
            model,
            judge_instruction,
            f"{uid}_judge",
            vid_dir,
            candidates=judge_candidates,
            use_no_vlm=False,
            pre_existing_messages=combined_messages  # Pass the full conversation history
        )

        # Map judge's answer (0 or 1) back to original/critic choice
        judge_choice = judge_result.get("answer", 0)
        if judge_choice == 0:
            final_answer = original_answer
        elif judge_choice == 1:
            final_answer = critic_answer
        else:
            # Fallback to original
            final_answer = original_answer

        return {
            "final_answer": final_answer,
            "reasoning": judge_result.get("reasoning", "Judge completed investigation"),
            "judge_frames": judge_result.get("evidence_frame_numbers", [])
        }

    except Exception as e:
        print(f"Error in judge pipeline: {e}")
        import traceback
        traceback.print_exc()
        # Default to original answer if judge fails
        return {
            "final_answer": original_answer,
            "reasoning": f"Judge pipeline error, defaulted to original: {e}"
        }

async def critic_assess(critic_model, question, uid, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context=None, ces_logs=None, criteria=None, answer_criteria=None, candidates=None, max_attempts = 15):
    """Critically assess an answer using VLM and return confidence score with error analysis

    Args:
        critic_model: CriticPipeline instance
        question: The original question
        answer: The proposed answer
        reasoning: The reasoning behind the answer
        evidence_frame_numbers: Frame numbers used as evidence
        general_context: Optional general context about the video
        criteria: Optional list of question-based verification criteria
        answer_criteria: Optional list of answer-specific criteria
        candidates: Optional list of answer choices

    Returns:
        Dictionary with: question, answer, confidence, possible_errors, suggestion, criteria_results (if criteria provided)
    """

    # Get subtitles for evidence frames
    subtitles = get_subtitles_for_frames(num, evidence_frame_numbers)

    # Generate initial critic VLM prompt
    prompt = critic_vlm_prompt(question, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context, ces_logs, criteria=criteria, answer_criteria=answer_criteria, candidates=candidates, subtitles=subtitles)
    for i in range(max_attempts):
        # Query LLM for VLM verification request
        message = "="*10 + " Requesting Critical VLM Assessment " + "="*10
        log(message, f"logs/log_video_{vid_dir}_{uid}")
        print("="*10 + " Requesting Critical VLM Assessment " + "="*10)
        response = await critic_model.llm_query_async(prompt)
        if response:
            message = f"Critic initial response: {response}"
            log(message, f"logs/log_video_{vid_dir}_{uid}")
            print(f"Critic initial response: {response[0:50]}...")
        else:
            print("Critic initial response: None - retrying...")
            continue
        
        critic_model.messages.append({"role": "assistant", "content": response})
        
        # Parse the response to extract VLM query
        parsed_response = None
        
        # Try multiple strategies to extract JSON
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                parsed_response = json.loads(json_str)
                print(f"Parsed response {uid}")
                break
            except:
                pass
        
        if not parsed_response and "{" in response and "}" in response:
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                parsed_response = json.loads(json_str)
                print(f"Parsed response {uid}")
                break
            except:
                pass
        
        if not parsed_response:
            try:
                parsed_response = json.loads(response.strip())
                break
            except:
                print(f"Failed to parse critic VLM request")
                pass
    if parsed_response is None:
        print("="*10 + f" Failed to parse critic VLM request {uid} " + "="*10)
        return {
            "uid": uid,
            "question": question,
            "answer": answer,
            "confidence": 0,
            "possible_errors": ["Could not parse structured assessment"],
            "suggestion": "Manual review recommended",
            "evidence_frame_numbers": evidence_frame_numbers
        }
        
    # Execute VLM query if requested
    try:
        if parsed_response.get("tool") == "VLM_QUERY" or "VLM" in parsed_response.get("tool"):
            frames = parsed_response.get("frames", evidence_frame_numbers)
            message = f"Frames to verify: {frames}"
            log(message, f"logs/log_video_{vid_dir}_{uid}")
            new_frames = [(f"{vid_dir}/{num}/" + frame) for frame in frames]
            vlm_prompt = parsed_response.get("prompt", "")

            # Get subtitles for the frames being queried
            frame_subtitles = get_subtitles_for_frames(num, frames)
            subtitles_text = ""
            if frame_subtitles:
                subtitles_text = "\n\nSubtitles visible in these frames:\n"
                for frame_path, subtitle in frame_subtitles.items():
                    subtitles_text += f"  {frame_path}: \"{subtitle}\"\n"

            # Add general context and subtitles to VLM prompt
            full_vlm_prompt = f"Video context: {general_context}\n\n{vlm_prompt}{subtitles_text}"
            if full_vlm_prompt:
                message = f"VLM Prompt: {full_vlm_prompt}"
                log(message, f"logs/log_video_{vid_dir}_{uid}")
                print(f"VLM Prompt: {full_vlm_prompt[:50]}...")
            else:
                print("VLM Prompt: None")
            print(f"Frames to verify: {new_frames[:1]}...")
            
            # Query VLM with the frames
            vlm_response = await critic_model.vlm_query(new_frames, full_vlm_prompt)
            critic_model.messages.append({"role": "vlm response", "content": vlm_response})
            message = f"VLM response: {vlm_response}"
            log(message, f"logs/log_video_{vid_dir}_{uid}")
            # Get critical assessment
            followup_prompt = critic_followup_prompt(
                vlm_response,
                question,
                answer,
                reasoning,
                evidence_frame_numbers,
                general_context,
                ces_logs,
                criteria=criteria,
                answer_criteria=answer_criteria,
                candidates=candidates,
                subtitles=subtitles
            )
            message = f"Followup prompt: {followup_prompt}"
            log(message, f"logs/log_video_{vid_dir}_{uid}")
            assessment_response = await critic_model.llm_query_async(followup_prompt)
            if assessment_response:
                message = f"Critical assessment response: {assessment_response}"
                log(message, f"logs/log_video_{vid_dir}_{uid}")
                print(f"Critical assessment response: {assessment_response[0:50]}...")
            else:
                print("Critical assessment response: None")
            
            # Robust JSON extraction for assessment response
            assessment_data = None
            
            # Try multiple strategies to extract JSON
            if "```json" in assessment_response:
                try:
                    json_str = assessment_response.split("```json")[1].split("```")[0].strip()
                    assessment_data = json.loads(json_str)
                except:
                    pass
            
            if not assessment_data and "{" in assessment_response:
                try:
                    start = assessment_response.find("{")
                    end = assessment_response.rfind("}") + 1
                    json_str = assessment_response[start:end]
                    assessment_data = json.loads(json_str)
                except:
                    pass
            
            if not assessment_data:
                try:
                    assessment_data = json.loads(assessment_response.strip())
                except:
                    pass
            
            # Extract critical assessment
            if assessment_data:
                message = f"Critical assessment data: {assessment_data}"
                log(message, f"logs/log_video_{vid_dir}_{uid}")

                result = {
                    "uid": uid,
                    "question": question,
                    "answer": answer,
                    "confidence": int(assessment_data.get("confidence", 50)),
                    "possible_errors": assessment_data.get("possible_errors", []),
                    "suggestion": assessment_data.get("suggestion", None),
                    "evidence_frame_numbers": evidence_frame_numbers  # Include original frames
                }

                # Include criteria evaluation results if present
                if "criteria_results" in assessment_data:
                    result["criteria_results"] = assessment_data["criteria_results"]
                    result["criteria_passed"] = assessment_data.get("criteria_passed", 0)
                    result["criteria_total"] = assessment_data.get("criteria_total", len(criteria) if criteria else 0)
                    result["criteria_percentage"] = assessment_data.get("criteria_percentage", 0)
                    result["critic_answer_choice"] = assessment_data.get("critic_answer_choice", -1)
                    result["critic_reasoning"] = assessment_data.get("critic_reasoning", "")

                    # Check if critic's answer differs from original and criteria > 50%
                    if answer in "01234":
                        original_answer_num = int(answer)
                    elif answer in "ABCDE":
                        original_answer_num = ord(answer) - ord('A')
                    elif isinstance(answer, int):
                        original_answer_num = answer
                    else:
                        original_answer_num = -1

                    critic_choice = result.get("critic_answer_choice", -1)
                    criteria_pct = result.get("criteria_percentage", 0)

                    if criteria_pct > 0.5 and critic_choice != original_answer_num and critic_choice != -1:
                        # Critic disagrees with original answer and criteria >50% - call third party judge
                        print(f"⚠ Critic answer ({critic_choice}) differs from original ({original_answer_num}). Calling third-party judge...")

                        vid_path = f"{vid_dir}/{num}"
                        judge_result = await third_party_judge(
                            critic_model,
                            question,
                            candidates,
                            original_answer_num,
                            critic_choice,
                            new_frames,
                            vid_path,
                            uid,
                            original_reasoning=reasoning,
                            critic_reasoning=result.get("critic_reasoning", ""),
                            original_messages=None,  # Will be auto-loaded from {uid}_model.json
                            critic_messages=critic_model.messages  # Pass critic's full conversation history
                        )

                        result["judge_decision"] = judge_result.get("final_answer", original_answer_num)
                        result["judge_reasoning"] = judge_result.get("reasoning", "")
                        result["requires_reeval"] = (judge_result.get("final_answer") != original_answer_num)

                        print(f"→ Third-party judge chose: {judge_result.get('final_answer')}")

                    elif criteria_pct <= 0.5:
                        # Criteria ≤50% - suggest re-evaluation with different frames
                        result["requires_reeval"] = True
                        result["reeval_reason"] = "Frames may not match the right scene (≤50% criteria passed)"
                        print(f"⚠ Only {criteria_pct*100:.1f}% criteria passed. Suggesting re-evaluation with different frames.")

                return result
            else:
                # Fallback: try to extract confidence from response
                import re
                numbers = re.findall(r'\d+', assessment_response)
                confidence_score = int(numbers[0]) if numbers else 50

                return {
                    "uid":uid, 
                    "question": question,
                    "answer": answer,
                    "confidence": confidence_score,
                    "possible_errors": ["Could not parse structured assessment"],
                    "suggestion": "Manual review recommended",
                    "evidence_frame_numbers": evidence_frame_numbers
                }
        else:
            print("No VLM verification requested")
            return {
                "uid": uid,
                "question": question,
                "answer": answer,
                "confidence": 0,
                "possible_errors": ["No VLM verification was performed"],
                "suggestion": "VLM verification required for assessment",
                "evidence_frame_numbers": evidence_frame_numbers
            }
                
    except Exception as e:
        print(f"Error during critical assessment: {e}")
        return {
            "uid": uid,
            "question": question,
            "answer": answer,
            "confidence": 0,
            "possible_errors": [f"Error: {str(e)}"],
            "suggestion": "Error occurred during assessment",
            "evidence_frame_numbers": evidence_frame_numbers
        }

async def batch_critic_assess(answers_data, global_summary, ces_logs, vid_dir, num, llm_model="deepseek-ai/DeepSeek-V3.1",
                             vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                             max_concurrent=10):
    """Critically assess multiple answers concurrently with rate limiting

    Args:
        answers_data: List of dicts with question, answer, reasoning, evidence_frame_numbers
        llm_model: LLM model name for critical assessment
        vlm_model: VLM model name for visual verification
        max_concurrent: Maximum concurrent assessments

    Returns:
        List of critical assessment results
    """
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

    # Add candidates to each answer data
    for data in answers_data:
        if 'candidates' not in data:
            data['candidates'] = candidates_by_uid.get(data.get('uid'), [])

    output_path = f"{vid_dir}/{num}/{num}_critic_assessment.json"
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        # Get UIDs that have already been processed
        processed_uids = {data.get("uid") for data in existing_results}
        # Filter out already processed answers
        original_count = len(answers_data)
        answers_data = [d for d in answers_data if d.get("uid") not in processed_uids]
        removed_count = original_count - len(answers_data)
        print(f"Found {len(existing_results)} already-processed answers")
        print(f"Processing {len(answers_data)} new answers")

    
    async def assess(data, max_retries=10):
        for attempt in range(max_retries):
            try:
                # Create a new critic instance for each assessment
                critic = CriticPipeline(llm_model, vlm_model)
                
                # critic_assess returns the complete assessment
                result = await critic_assess(
                    critic,
                    data.get("question"),
                    data.get("uid"),
                    data.get("answer"),
                    data.get("reasoning", ""), #TODO: RUN EXPERIMENT WITH NO REASONING FOR THE CRITIC
                    data.get("evidence_frame_numbers", data.get("frames", [])),
                    vid_dir,
                    num,
                    global_summary,
                    ces_logs,
                    criteria=data.get("criteria", None),  # Pass question-based criteria if present
                    answer_criteria=data.get("answer_criteria", None),  # Pass answer-specific criteria if present
                    candidates=data.get("candidates", [])  # Pass candidates if present
                )
                # Log per-question assessment completion
                try:
                    with open("answers_logs.json", "a") as log_f:
                        log_f.write(f"critic assessment completed for uid {data.get('uid')} video {num} in {vid_dir}\n")
                except Exception:
                    pass
                
                # Save full critic conversation
                try:
                    conv_path = f"{vid_dir}/{num}/{data.get('uid')}_critic_model.json"
                    with open(conv_path, "w") as conv_f:
                        json.dump(critic.messages, conv_f, indent=2)
                except Exception:
                    pass
                
                return result
                
            except (asyncio.TimeoutError, TimeoutError) as e:
                print(f"Timeout error on attempt {attempt + 1}/{max_retries} for uid {data.get('uid')}: {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed, return error result
                    return {
                        "uid": data.get("uid"),
                        "question": data.get("question"),
                        "answer": data.get("answer"),
                        "confidence": -1,
                        "possible_errors": [f"Timeout after {max_retries} attempts"],
                        "suggestion": "Request timed out",
                        "evidence_frame_numbers": data.get("evidence_frame_numbers", [])
                    }
                # Wait before retrying
                await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff with max 30s
                
            except Exception as e:
                error_msg = str(e)
                print(f"Unexpected error for uid {data.get('uid')}: {error_msg}")
                traceback.print_exc()
                
                # Check if it's a client session error and retry
                if "Unclosed client session" in error_msg or "ClientSession" in error_msg:
                    print(f"Client session error detected, retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                
                return {
                    "uid": data.get("uid"),
                    "question": data.get("question"),
                    "answer": data.get("answer"),
                    "confidence": -1,
                    "possible_errors": [f"Error: {error_msg}"],
                    "suggestion": "Error occurred during assessment",
                    "evidence_frame_numbers": data.get("evidence_frame_numbers", [])
                }

    # Process all assessments concurrently with rate limiting
    if answers_data:  # Only process if there are new answers
        tasks = [assess(data) for data in answers_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any unexpected exceptions from gather
        new_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
                new_results.append({
                    "uid": answers_data[i].get("uid"),
                    "question": answers_data[i].get("question"),
                    "answer": answers_data[i].get("answer"),
                    "confidence": -1,
                    "possible_errors": [f"Task failed: {str(result)}"],
                    "suggestion": "Task execution failed",
                    "evidence_frame_numbers": answers_data[i].get("evidence_frame_numbers", [])
                })
            else:
                new_results.append(result)
        
        # Combine existing and new results
        all_results = existing_results + new_results
    else:
        # No new answers to process, return existing results
        all_results = existing_results
        print("No new answers to process")
    
    return all_results

async def assess_all(video_dir, num,
                     llm_model="deepseek-ai/DeepSeek-V3.1",
                     vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    """Assess all answers and return with confidence scores
    
    Args:
        answers_data: List of answer dictionaries
        llm_model: LLM model for assessment
        vlm_model: VLM model for verification
    
    Returns:
        List of all assessments with confidence scores and error analysis
    """
    # Perform batch assessment

    answers_path = f"{video_dir}/{num}/{num}_answers_reformatted.json"
    global_summary_path = f"{video_dir}/{num}/captions/global_summary.txt"
    ces_logs_path = f"{video_dir}/{num}/captions/CES_logs.txt"

    if os.path.exists(answers_path):
        with open(answers_path, 'r') as f:
            answers_data = json.load(f)

        # Read global summary content
        if os.path.exists(global_summary_path):
            with open(global_summary_path, 'r') as f:
                global_summary = f.read()
        else:
            print(f"Warning: Global summary not found at {global_summary_path}")
            global_summary = ""

        # Read CES logs content
        if os.path.exists(ces_logs_path):
            with open(ces_logs_path, 'r') as f:
                ces_logs = f.read()
        else:
            print(f"Warning: CES logs not found at {ces_logs_path}")
            ces_logs = ""

        assessments = await batch_critic_assess(answers_data, global_summary, ces_logs, video_dir, num, llm_model, vlm_model)
        
        # Calculate summary statistics
        confidences = [a["confidence"] for a in assessments]
        
        print("\n" + "="*60)
        print("CRITICAL ASSESSMENT SUMMARY")
        print("="*60)
        print(f"Total answers assessed: {len(assessments)}")
        print(f"Average confidence: {sum(confidences)/len(confidences):.1f}%")
        print(f"Highest confidence: {max(confidences)}%")
        print(f"Lowest confidence: {min(confidences)}%")
        print(f"Answers below 70% confidence: {sum(1 for c in confidences if c < 70)}")

        output_path = f"{video_dir}/{num}/{num}_critic_assessment.json"
        with open(output_path, "w") as f:
            json.dump(assessments, f, indent = 2)
        


        with open("answers_logs.json", "a") as f:
            f.write(f"saved critic assessments for video {num} in {video_dir}\n")
        
        return assessments
    else:
        with open("failure_log", 'a') as f:
            f.write(f"didn't critic-assess video {num} \n")
        return


async def batch_assess_all(video_dir, batch_size = 2, llm_model="deepseek-ai/DeepSeek-V3.1",
                     vlm_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    results = []
    nums = os.listdir(video_dir)
    num_batches = math.ceil(len(nums)/batch_size)
    open("critic_stats.txt", "w").close()
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1)* batch_size, len(nums))
        print(start, end)
        num_list = nums[start : end]
        print(num_list)
        tasks = [assess_all(video_dir, num) for num in num_list]
        results_batch = await asyncio.gather(*tasks, return_exceptions = True)
        print(results_batch)
        results.append(results_batch)
    for num, result in zip(nums, results):
        if isinstance(result, Exception):
            with open("failure_log", "a") as f:
                f.write(f"Failed critic task for {num} because of {result} \n")
            print(f"Failed critic task for {num} because of {result}")
            print(traceback.format_exception(type(result), result, result.__traceback__))
        else:
            with open("critic_stats.txt", "a") as f:
                f.write(f"CRITIC ASSESSMENT COMPLETED FOR VIDEO {num} \n")
            print("result: ", result)
            print("CRITIC ASSESSMENT COMPLETED FOR VIDEO ", num)
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog = "critic_model.py",
        description = "collects answers, criticizes them"
    ) 

    parser.add_argument('dirname')
    args = parser.parse_args()

    

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(batch_assess_all(args.dirname))
    finally:
        loop.close()
    
    import asyncio
    open('embed_queries.json', 'w').close()
    open('ret_embeddings.json', 'w').close()
    with open('embed_queries.json', 'w') as f:
        json.dump({}, f, indent=2)
    with open('ret_embeddings.json', 'w') as f:
        json.dump({}, f, indent=2)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(total_main(args.dirname))
    finally:
        loop.close()