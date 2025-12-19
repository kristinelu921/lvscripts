import json
import os
from model_example_query import query_llm, query_llm_async

def initial_prompt(question):
    return f"""
You are a reasoning model whose goal is to answer questions about a video. You cannot see the video, but you can use tools to retrieve information about the video.

You will be given a question and answer choices.

Question: {question}
Given the above question and answer choices, please extract the following information:
   1. Temporal: WHAT SCENES/KEY EVENTS in the video do I need to look for? 
   2. Action : What ACTIONS should I look for?
   3. Spatial: WHERE should the scene's setting be? What are some typical CHARACTERISTICS of this setting I might look for?
   4. Semantic: What ITEMS should I look for?
   5. Subject: WHO should I look for, and what do they look like? What is a physical description I can use to identify them?
   6. Length: How long/how many frames do you expect this question to span? How many actions, and predict if you will need clip queries or just static frames. (Eg: If it involves a sequence of actions, we may have to look through A LOT of frames, maybe query videos).

Remember that for TEMPORAL questions asking about length, you should find "starting" and "ending" frames, and keep track of / record timestamps. Time information likely WON'T come from within a video.
Come up with ONE phrase/sentence to look for in the captions. In particular, think about WHAT MIGHT APPEAR in frames that are important to you. 

Return your answer in the following json format:
```json{{
    "tool": "CAPTION_SEARCH",
    "prompt": "your phrase/sentence to look for in the captions"
}}```
"""

def followup_prompt(json_output, question):
    return f"""
Here is the information we received:
{json_output}

The question was: {question}
Now, please choose to do EXACTLY ONE of the following:
1. Choose to query a VLM about a set of frames with a prompt of your choosing. You should choose a WIDE window of frames (eg 30 or 40 frames) at intervals close to key-frames that make sense (you can query ANY frames) or that you selected from the caption-similarity search results, so you get the full context + actions of the scene. You must format the frames as follows: frames/frame_xxxx.jpg, where xxxx is the number of seconds (eg: frames/frame_0102.jpg is the frame at 102 seconds).
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "VLM_QUERY",
            "frames": [Set of frames to query],
            "prompt": "...",
        }}
2. Choose to caption search with a short set of information you need to retrieve. You should CHOOSE this option if you don't have CLEAR evidence that previously chosen frames fit the constraints of the question. Here, you can broadly search for the information you need.
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "CAPTION_SEARCH",   
            "prompt": "...",
        }}
3. Determine your final answer based on the information you have retrieved. ONLY CHOOSE THIS OPTION if you're SURE of your answer. you MUST HAVE VLM/FRAME evidence first. If you are AT ALL unsure, or the answers don't make full sense, you should SEARCH for a different scene or look for more frames. Bias to options 1 and 2 unless you have CLEAR evidence. Make sure the letter and the answer match the answer choices given in the question.
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "FINAL_ANSWER",
            "frames": [list of frame numbers (eg: frames/frame_0001.jpg, frames/frame_0002.jpg)],
            "answer": "...", (eg: "A")
            "reasoning": "...",
        }}
    - IMPORTANT: If you choose the "FINAL_ANSWER" tool, choose AT LEAST as many frames as necessary to answer the questions such that a separate entity could use only these frames to deduce/confirm your answer. Bias towards extra frames if you are unsure. 
    - IMPORTANT: Frame paths must include "frames/" prefix and ".jpg" suffix
    - IMPORTANT: if a question asks you about a specific timestamp, convert it to seconds and QUERY THE VLM about the frames at THOSE seconds, in the right format.
    - IMPORTANT: if you choose caption search, think about what MIGHT VISUALLY APPEAR in frames of relevance.
"""

def finish_prompt(scratchpad):
    return f"""
Given all the information here:
{scratchpad}, please determine a final answer.

Make sure your answer and letter match the answer choices given in the question.
Return your answer in the following json format:
json_output = {{
    "answer": "..." (eg: "A"),
    "frames": [list of frame numbers],
    "reasoning": "...",
}}
- IMPORTANT: If you choose the "FINAL_ANSWER" tool, choose AT LEAST as many frames as necessary to answer the questions such that a separate entity could use only these frames to deduce/confirm your answer. Bias towards extra frames if you are unsure. 
- IMPORTANT: Frame paths must include "frames/" prefix and ".jpg" suffix
"""

def response_parsing_prompt(response):
    return f"""
Extract the JSON object from this response. The response contains ONE of these tools: VLM_QUERY, CAPTION_SEARCH, or FINAL_ANSWER.

Response to parse:
{response}

Return ONLY a JSON object, nothing else. The JSON must follow one of these exact formats:

For VLM_QUERY:
{{
    "tool": "VLM_QUERY",
    "frames": ["frames/frame_0001.jpg", "frames/frame_0002.jpg"],
    "prompt": "describe what you see"
}}

For CAPTION_SEARCH:
{{
    "tool": "CAPTION_SEARCH",
    "input": "search query text",
    "prompt": "optional prompt text"
}}

For FINAL_ANSWER:
{{
    "tool": "FINAL_ANSWER",
    "frames": ["frames/frame_0001.jpg", "frames/frame_0002.jpg", ...],
    "answer": "A",
    "reasoning": "explanation here"
}}

IMPORTANT: 
- Return ONLY valid JSON
- Do NOT add any text before or after the JSON
- For CAPTION_SEARCH, "input" should be a string, not an array
- Frame paths for VLM_QUERY must include "frames/" prefix and ".jpg" suffix
"""


async def reformat_answers(answers_path):
    if (os.path.exists(f'{answers_path[:-5]}_reformatted.json')):
        with open(f'{answers_path[:-5]}_reformatted.json', 'r') as f:
            results = json.load(f)
        return results
    with open(answers_path, "r") as f:
        answers = str(f.read())
    prompt = f"You are a reformatter. Here are the answers you are to reformat: \n {answers} \n Please make sure ALL the questions and answers are in the following format. If you see any questions or answers that are not in the format, please reformat them. Return ONLY a valid JSON array, no other text: "
    prompt += f"""
    [
      {{
        "uid": "...",
        "question": "...",
        "answer": "A", (eg: "A")
        "frames": [list of frame numbers (eg: frames/frame_0001.jpg, frames/frame_0002.jpg)],
        "reasoning": "...",
      }},
      ...
    ]
    IMPORTANT: 
    - Return ONLY valid JSON
    - Do NOT add any text before or after the JSON
    - Frame paths for VLM_QUERY must include "frames/" prefix and ".jpg" suffix

    - The final answer should be a single capitalized letter (eg: "A")
    """
    print("reformatting before inference")
    #print("prompt", prompt)
    result = await query_llm_async("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", prompt)
    print("reformatting post inference")
    
    # Try to parse the result as JSON
    try:
        # Extract JSON if wrapped in markdown code blocks
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Parse and return as proper JSON object
        return json.loads(result)
    except json.JSONDecodeError:
        # If parsing fails, return the raw result
        print(f"Warning: Could not parse reformatted answers as JSON")
        return result



def verifier_prompt(question, answer, reasoning, evidence_frame_numbers, general_context):

    return f"""
You are a verifier model. You are given a question, answer, reasoning, and evidence frame numbers.

Question: {question}
Final Answer: {answer}
Reasoning: {reasoning}
Evidence frame numbers: {evidence_frame_numbers}
General context: {general_context}

Please understand the question, answer, reasoning, and evidence frame numbers. Output a single confidence score between 0 and 100, where 100 means you are 100% confident in the answer being correct, and 0 means you are 0% confident, and the answer is definitely wrong.

Please use the frames to create a request to a VLM model to verify the answer. Make ONE CALL. 

Rate your confidence in the answer based on the question, choices, reasoning, and evidence frame numbers. Be objective and thorough. Don't hallucinate, and also consider the context of the video. 

You should choose to request a VLM model to verify the answer.

Please output your request in the following json format:
{{
    "tool": "VLM_QUERY",
    "frames": [list of frames (eg: frames/frame_0001.jpg, frames/frame_0002.jpg)],
    "prompt": "...",
}}
IMPORTANT:
- Return ONLY valid JSON
- Do NOT add any text before or after the JSON
- Frame paths for VLM_QUERY must include "frames/" prefix and ".jpg" suffix
"""

def verifier_followup_prompt(json_output, question, answer, reasoning, evidence_frame_numbers, general_context):

    return f"""
Based on the VLM verification results above, evaluate the confidence in this specific answer:

Original Question: {question}
Given Answer: {answer}
Original Reasoning: {reasoning}

You should pay close attention to exactly what the question is asking for, how well the presented scenes fit the conditions of the question, and how well the reasoning matches the answer. You must choose a confidence score between 0 and 100, where 100 means you are 100% confident the answer is correct, and 0 means you are 0% confident.

Return ONLY this JSON with the EXACT question and answer provided above:
{{
    "uid": "..."
    "question": "{question}",
    "answer": "{answer}",
    "confidence": <your confidence score 0-100>,
    "confidence_reasoning": "<explain why you gave this confidence score>"
}}

IMPORTANT: 
- Use the EXACT question text provided above
- Use the EXACT answer text provided above
- Do NOT make up your own question or modify the text
- Return ONLY valid JSON, no other text
- "confidence" should be in lower case. The value must be 0 - 100.
- "confidence_reasoning" should be a short explanation of why you gave this confidence score.
"""

def critic_vlm_prompt(question, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context):
    return f"""
You are a critic model evaluating whether an answer to a video question is correct.

Question: {question}
Given Answer: {answer}
Reasoning: {reasoning}
Evidence frames: {evidence_frame_numbers}
General context: {general_context}

Your task is to critically examine whether the answer and reasoning match what the question is asking for. Look for:
- Whether the scenes match the location/setting described in the question
- Whether the evidence time frames make sense given the question (EG: if the question asks about a timestamp, is that timestamp's frame queried?)
- Whether all conditions in the question are satisfied
- Whether the action described actually happened
- Whether the answer choice matches the evidence

Create a VLM query to verify the key claims in the reasoning.

Return your VLM query request in this JSON format:
{{
    "tool": "VLM_QUERY",
    "frames": {evidence_frame_numbers if evidence_frame_numbers else []},
    "prompt": "Please verify: [specific thing to check based on the question and answer]. Look for [specific visual elements]. Does the scene show [key claim from reasoning]?"
}}

IMPORTANT:
- Return ONLY valid JSON
- Use the provided evidence_frame_numbers as the frames list
- Create a specific, targeted prompt that verifies the key claims
- Frame paths must include "frames/" prefix and ".jpg" suffix
"""

def critic_followup_prompt(json_output, question, answer, reasoning, evidence_frame_numbers, general_context):
    return f"""
Critically assess whether the VLM verification supports the given answer.

VLM Verification Results: 
{json_output}

Question Being Answered: {question}
Given Answer: {answer}
Original Reasoning: {reasoning}
Evidence Frames Used: {evidence_frame_numbers}

CRITICAL ASSESSMENT TASK:
1. Check if the VLM results CONFIRM what the reasoning claims
2. Look for CONTRADICTIONS between VLM results and the answer
3. Verify ALL conditions in the question are addressed
4. Check if the correct answer choice was selected

Based ONLY on what the VLM actually reports seeing, determine:
- Does the visual evidence support the answer?
- Are there any discrepancies between claims and visuals?
- Is the answer addressing the right scene/moment?

Return ONLY this JSON:
{{
    "confidence": <0-100, where 100 means VLM fully confirms the answer>,
    "possible_errors": [
        <List ONLY real discrepancies you found, not hypothetical ones>
    ],
    "suggestion": <null if confidence >80, otherwise specific improvement>
}}

IMPORTANT RULES:
- Base confidence ONLY on VLM verification results, not speculation
- Only list errors that are ACTUALLY present in the VLM results
- If VLM confirms the answer, confidence should be high (80-100)
- If VLM contradicts the answer, confidence should be low (0-40)
- Empty possible_errors list [] if no actual errors found
- Do NOT invent errors that aren't supported by the VLM results
- Return ONLY valid JSON
"""