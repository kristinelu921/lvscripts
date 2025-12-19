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
    gemini_key_PRIV = env_data["gemini_key"]

os.environ['TOGETHER_API_KEY'] = together_key_PRIV
os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
client_together = Together()
async_client_together = AsyncTogether(api_key=together_key_PRIV)

class CriticPipeline:
    def __init__(self, llm_model_name, vlm_model_name):
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        
        self.llm = llm_model_name
        self.vlm = vlm_model_name
        
        self.messages = []
    
    def llm_query(self, prompt):
        return query_llm(self.llm, prompt)
    
    async def llm_query_async(self, prompt):
        return await query_llm_async(self.llm, prompt)
    
    async def vlm_query(self, image_paths, prompt):
        return await query_vlm(self.vlm, image_paths, prompt)

async def critic_assess(critic_model, question, uid, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context=None, max_attempts = 15):
    """Critically assess an answer using VLM and return confidence score with error analysis
    
    Args:
        critic_model: CriticPipeline instance
        question: The original question
        answer: The proposed answer
        reasoning: The reasoning behind the answer
        evidence_frame_numbers: Frame numbers used as evidence
        general_context: Optional general context about the video
    
    Returns:
        Dictionary with: question, answer, confidence, possible_errors, suggestion
    """
    
    # Generate initial critic VLM prompt
    prompt = critic_vlm_prompt(question, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context)
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
            new_frames = [(f"./{vid_dir}/{num}/" + frame) for frame in frames]
            vlm_prompt = parsed_response.get("prompt", "")
            
            # Add general context to VLM prompt
            full_vlm_prompt = f"Video context: {general_context}\n\n{vlm_prompt}"
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
                general_context
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
                
                return {
                    "uid": uid,
                    "question": question,
                    "answer": answer,
                    "confidence": int(assessment_data.get("confidence", 50)),
                    "possible_errors": assessment_data.get("possible_errors", []),
                    "suggestion": assessment_data.get("suggestion", None),
                    "evidence_frame_numbers": evidence_frame_numbers  # Include original frames
                }
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

async def batch_critic_assess(answers_data, global_summary, vid_dir, num, llm_model="openai/gpt-oss-120b", 
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
    output_path = f"./{vid_dir}/{num}/{num}_critic_assessment.json"
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
                    data.get("reasoning", ""),
                    data.get("evidence_frame_numbers", data.get("frames", [])),
                    vid_dir,
                    num,
                    global_summary
                )
                # Log per-question assessment completion
                try:
                    with open("answers_logs.json", "a") as log_f:
                        log_f.write(f"critic assessment completed for uid {data.get('uid')} video {num} in {vid_dir}\n")
                except Exception:
                    pass
                
                # Save full critic conversation
                try:
                    conv_path = f"./{vid_dir}/{num}/{data.get('uid')}_critic_model.json"
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
                     llm_model="openai/gpt-oss-120b",
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

    answers_path = f"./{video_dir}/{num}/{num}_answers_reformatted.json"
    global_summary = f"./{video_dir}/{num}/captions/global_summary.txt"

    if os.path.exists(answers_path):
        with open(answers_path, 'r') as f:
            answers_data = json.load(f)
        assessments = await batch_critic_assess(answers_data, global_summary, video_dir, num, llm_model, vlm_model)
        
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
        
        output_path = f"./{video_dir}/{num}/{num}_critic_assessment.json"
        with open(output_path, "w") as f:
            json.dump(assessments, f, indent = 2)
        


        with open("answers_logs.json", "a") as f:
            f.write(f"saved critic assessments for video {num} in {video_dir}\n")
        
        return assessments
    else:
        with open("failure_log", 'a') as f:
            f.write(f"didn't critic-assess video {num} \n")
        return


async def batch_assess_all(video_dir, batch_size = 2, llm_model="openai/gpt-oss-120b",
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