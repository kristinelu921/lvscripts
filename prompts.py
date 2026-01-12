import json
import os
from model_example_query import query_llm, query_llm_async

def initial_prompt(question, candidates=None):
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    return f"""
You are a reasoning model whose goal is to answer questions about a video. You cannot see the video, but you can use tools to retrieve information about the video.

You will be given a question and answer choices.

Question: {question}{candidates_text}

TEMPORAL ATTENTION:
FIRST, check if the question contains time keywords (after, before, during, when, then, while, first, next, later, etc.):
- If present, IDENTIFY and MARK DOWN all temporal relationships and timeframes mentioned
- Note which events should occur before/after/during other events
- This temporal information is CRITICAL for finding the right frames

STEP 1: EXTRACT VERIFICATION CRITERIA
First, carefully analyze the question and extract 4-6 specific, testable criteria that can be verified visually in the video frames. These criteria should be concrete conditions that must be satisfied to answer the question correctly. These criteria should be derived FROM THE QUESTION, NOT THE ANSWER CHOICES. 

For example, if the question asks about "a person in a black shirt holding a metal bucket in a scene with scattered debris", the criteria might be:
- Person is wearing a black shirt
- Person is wearing black pants
- Person is holding a large round metal bucket
- Scene contains scattered debris on the ground
- Scene shows items scattered on the ground

Extract ONLY verifiable visual criteria from the question. DO NOT include criteria about the answer choices.

STEP 2: GENERATE SEARCH QUERIES
Now generate 2-4 search queries to find the relevant frames. Each query should be:
- STRUCTURED like a natural caption (subject-verb-object format)
- FOCUS on VISIBLE, CONCRETE details from the question
- DESCRIBE what would APPEAR in the frame (not abstract concepts)
- Use complete, grammatical phrases (e.g., "woman in yellow dress holds paper")

Guidelines for creating queries:
1. Prioritize VISIBLE traits: clothing colors, physical objects, actions, settings
2. Use natural language structure: "person wearing [clothing] doing [action] in [location]"
3. Each query should target a DIFFERENT aspect or detail from the question
4. Avoid jumbled keywords - write how a caption would describe the scene

Example question: "A woman in a yellow dress is holding papers in an office. What is she reading?"
GOOD queries:
- "woman wearing yellow dress holding papers"
- "person in office setting with documents"
- "woman reading document in indoor workspace"

BAD queries (jumbled/unclear):
- "yellow dress papers office woman"
- "holding reading yellow"

Return your answer in the following json format:
```json{{
    "tool": "CAPTION_SEARCH",
    "criteria": [
        "criterion 1 - a specific visual condition to verify",
        "criterion 2 - another specific visual condition",
        "criterion 3 - another specific visual condition",
        "criterion 4 - another specific visual condition",
        "criterion 5 - (optional) another specific visual condition",
        "criterion 6 - (optional) another specific visual condition"
    ],
    "search_queries": [
        "first natural search query focusing on main subject/action",
        "second query focusing on setting/objects",
        "third query focusing on alternative details (optional)",
        "fourth query for additional context (optional)"
    ]
}}```

IMPORTANT: Generate 2-4 search queries. Each should be a natural, structured phrase describing visible elements.
"""

def followup_prompt(json_output, question, candidates=None):
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    return f"""
Here is the information we received:
{json_output}

The question was: {question}{candidates_text}

⚠️ TEMPORAL ATTENTION ⚠️
- Frame IDs are TIMESTAMPS: frame_0050.jpg = 50 seconds into the video
- INCREASING frame numbers = TIME PASSING (frame_0050 → frame_0100 means 50 seconds passed)
- If question mentions "AFTER" an event: Look 5-30 frames (seconds) AFTER the located event
- If question mentions "BEFORE" an event: Look 5-30 frames (seconds) BEFORE the located event
- If question asks for a "SEQUENCE" of events, you should order events in frame time order. Also, if a question asks for a "DURATION" of an event, you should query the VLM for the frames at the beginning and end of the event.
- Mark down and track ALL timestamps/frame numbers
- Pay special attention to temporal keywords (after, before, during, when, then, etc.)

⚠️ GROUND TRUTH HIERARCHY ⚠️
VLM observations are GROUND TRUTH. Caption searches are only for LOCATING relevant frames.
- If VLM has provided observations, those are FACTS about what's in the frames
- If VLM contradicts caption search results, VLM is ALWAYS correct
- Caption searches are APPROXIMATIONS to find frames; VLM SEES the actual content

⚠️ CRITICAL ORGANIZATIONAL RULE ⚠️
After EVERY VLM_QUERY call where you find relevant information, you MUST use RECORD to log the observations with timestamps. This is MANDATORY for:
- Questions about SEQUENCES of events
- Questions asking about MULTIPLE locations or objects
- ANY question where you need to compare or track information across different parts of the video

RECORD is an ORGANIZATIONAL TOOL - use it liberally to keep your evidence organized!

Now, please choose to do EXACTLY ONE of the following:
1. Choose to query a VLM about a set of frames with a prompt of your choosing. You should choose a WIDE window of frames (eg 30 or 40 frames) at intervals close to key-frames that make sense (you can query ANY frames) or that you selected from the caption-similarity search results, so you get the full context + actions of the scene. You must format the frames as follows: frames/frame_xxxx.jpg, where xxxx is the number of seconds (eg: frames/frame_0102.jpg is the frame at 102 seconds).
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "VLM_QUERY",
            "frames": [Set of frames to query],
            "prompt": "...",
        }}
    - ⚠️ REMINDER: After VLM_QUERY, if you found relevant information, your NEXT step should be RECORD (option 2) to log the observations!

2. Choose to RECORD relevant observations from the VLM results to track evidence across the video. This is CRITICAL for questions requiring you to track SEQUENCES of events across many sections of the video. You MUST use this AFTER VLM_QUERY calls when you find relevant information.
    - ⚠️ MANDATORY after VLM queries that found useful information
    - ⚠️ REQUIRED for sequence/temporal questions before final answer
    - You MUST use this format for EACH entry: "Time: ____ seconds, Event: ______"
    - You can record MULTIPLE entries at once (e.g., multiple events from the same VLM query)
    - Example entries:
      * "Time: 45 seconds, Event: Woman in yellow dress picks up papers"
      * "Time: 67 seconds, Event: Man enters office wearing blue shirt"
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "RECORD",
            "entries": [
                "Time: [seconds] seconds, Event: [description]",
                "Time: [seconds] seconds, Event: [description]",
                ...
            ]
        }}
    - The system will automatically sort and track all recorded entries for you
3. Choose to caption search with 2-4 natural, structured queries. You should CHOOSE this option if you don't have CLEAR evidence that previously chosen frames fit the constraints of the question. Generate multiple queries targeting DIFFERENT visual aspects.

    SEARCH QUERY GUIDELINES:
    - Write queries like natural captions (subject-verb-object)
    - Focus on VISIBLE details: clothing, objects, actions, settings
    - Each query should search for DIFFERENT aspects
    - Use complete phrases, not jumbled keywords

    Example: For "woman in yellow dress with papers in office":
    GOOD: ["woman wearing yellow dress", "person holding papers in office", "indoor workspace with documents"]
    BAD: ["yellow dress papers", "woman office"]

    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "CAPTION_SEARCH",
            "search_queries": [
                "first natural query about main subject/action",
                "second query about setting/objects",
                "optional third query for additional details"
            ]
        }}
4. Choose to VIEW ALL RECORDED ENTRIES to review the evidence you've collected so far. This returns all your recorded observations sorted by time, helping you reason about sequences and relationships. Use this when you want to see all the evidence you've gathered through RECORD calls.
    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "VIEW_RECORDS"
        }}
5. Determine your final answer based on the information you have retrieved. ONLY CHOOSE THIS OPTION if you're SURE of your answer. you MUST HAVE VLM/FRAME evidence first. If you are AT ALL unsure, or the answers don't make full sense, you should SEARCH for a different scene or look for more frames. Bias to options 1-4 unless you have CLEAR evidence. For questions about SEQUENCES or MULTIPLE EVENTS, use RECORD (option 2) and VIEW_RECORDS (option 4) to organize your findings before answering. Make sure the letter and the answer match the answer choices given in the question. Your answer MUST be a LETTER (eg: "A", "B", "C", "D", etc.) ONLY, corresponding to the answer choices given in the question.

⚠️ CRITICAL: There is ALWAYS a correct answer among the choices provided. If all answers seem slightly off or imperfect, you MUST choose the BEST possible answer that most closely matches the evidence. Do not refuse to answer.

IMPORTANT: THE ANSWER OUTPUT MUST BE A SINGLE NUMBER (eg: 0, 1, 2, 3, 4, etc.) where (0 -> A, 1 -> B, 2 -> C, 3 -> D, etc.). ONLY ONE CHARACTER LONG.

    - When providing FINAL_ANSWER, you MUST also generate ONE ANSWER-SPECIFIC CRITERIA that verify YOUR chosen answer choice is correct. These are DIFFERENT from question criteria - they validate whether YOUR FINAL ANSWER matches the evidence.

    Example: If question is "What is the person doing?" with choices A: Reading, B: Cooking, C: Dancing
    And you choose "1" (B: Cooking), your answer_criteria should be the display of the answer choice you chose:
    - "Person is performing cooking actions (stirring, cutting, handling food)"
    - "Kitchen environment or cooking equipment is visible"

    - If you choose this, please return the following json format:
        json_output = {{
            "tool": "FINAL_ANSWER",
            "frames": [list of frame numbers (eg: frames/frame_0001.jpg, frames/frame_0002.jpg)],
            "answer": "...", (eg: "A") Your answer MUST be one character long, a NUMBER with (0 -> A, 1 -> B, 2 -> C, 3 -> D, etc.) according to the answer choices given in the question.
            "reasoning": "...",
            "answer_criteria": [
                "First criterion validating your chosen answer",
                "Second criterion validating your chosen answer (optional)"
            ]
        }}
    - IMPORTANT: If you choose the "FINAL_ANSWER" tool, choose AT LEAST as many frames as necessary to answer the questions such that a separate entity could use only these frames to deduce/confirm your answer. Bias towards extra frames if you are unsure.
    - IMPORTANT: In the "FINAL_ANSWER" tool, your answer MUST be a NUMBER (eg: 0, 1, 2, 3, 4, etc.) ONLY and nothing else, corresponding to the answer choices (0 -> A, 1 -> B, 2 -> C, 3 -> D, etc.) given in the question. 
    - IMPORTANT: Frame paths must include "frames/" prefix and ".jpg" suffix
    - IMPORTANT: if a question asks you about a specific timestamp, convert it to seconds and QUERY THE VLM about the frames at THOSE seconds, in the right format.
    - IMPORTANT: if you choose caption search, think about what MIGHT VISUALLY APPEAR in frames of relevance.
"""

def finish_prompt(scratchpad, candidates=None):
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    return f"""
Given all the information here:
{scratchpad}, please determine a final answer.{candidates_text}

Make sure your answer and letter match the answer choices given above.

⚠️ CRITICAL: There is ALWAYS a correct answer among the choices provided. If all answers seem slightly off or imperfect, you MUST choose the BEST possible answer that most closely matches the evidence. Do not refuse to answer.

IMPORTANT: You must also generate 1-2 ANSWER-SPECIFIC CRITERIA that verify your chosen answer is correct.
These criteria should validate whether the specific answer choice you selected matches the visual evidence.

Return your answer in the following json format:
json_output = {{
    "answer": "..." (eg: "A"),
    "frames": [list of frame numbers],
    "reasoning": "...",
    "answer_criteria": [
        "First criterion that validates your chosen answer",
        "Second criterion that validates your chosen answer (optional)"
    ]
}}
- IMPORTANT: If you choose the "FINAL_ANSWER" tool, choose AT LEAST as many frames as necessary to answer the questions such that a separate entity could use only these frames to deduce/confirm your answer. Bias towards extra frames if you are unsure.
- IMPORTANT: Frame paths must include "frames/" prefix and ".jpg" suffix
"""

def response_parsing_prompt(response):
    return f"""
Extract the JSON object from this response. The response contains ONE of these tools: VLM_QUERY, RECORD, VIEW_RECORDS, CAPTION_SEARCH, or FINAL_ANSWER.

Response to parse:
{response}

Return ONLY a JSON object, nothing else. The JSON must follow one of these exact formats:

For VLM_QUERY:
{{
    "tool": "VLM_QUERY",
    "frames": ["frames/frame_0001.jpg", "frames/frame_0002.jpg"],
    "prompt": "describe what you see"
}}

For RECORD:
{{
    "tool": "RECORD",
    "entries": [
        "Time: 45 seconds, Event: Woman in yellow dress picks up papers",
        "Time: 67 seconds, Event: Man enters office"
    ]
}}

For VIEW_RECORDS:
{{
    "tool": "VIEW_RECORDS"
}}

For CAPTION_SEARCH:
{{
    "tool": "CAPTION_SEARCH",
    "criteria": ["criterion 1", "criterion 2", ...],  // Optional, only on first call
    "search_queries": ["first query", "second query", "optional third query"]  // 2-4 natural search queries
}}

OR (legacy single-query format):
{{
    "tool": "CAPTION_SEARCH",
    "input": "single search query text"
}}

For FINAL_ANSWER:
{{
    "tool": "FINAL_ANSWER",
    "frames": ["frames/frame_0001.jpg", "frames/frame_0002.jpg", ...],
    "answer": "A",
    "reasoning": "explanation here",
    "answer_criteria": ["criterion 1", "criterion 2 (optional)"]
}}

IMPORTANT:
- Return ONLY valid JSON
- Do NOT add any text before or after the JSON
- For CAPTION_SEARCH, prefer "search_queries" as an array of 2-4 natural queries
- For CAPTION_SEARCH legacy format, "input" or "prompt" should be a string
- For CAPTION_SEARCH, "criteria" is an optional array that may be present in the first call
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

def _expand_frames_with_surrounding(evidence_frame_numbers, seconds_before=5, seconds_after=5):
    """Expand frame list to include frames ~5 seconds before and after each frame"""
    expanded_frames = set()
    for frame in evidence_frame_numbers:
        # Extract frame number from path like "frames/frame_0050.jpg"
        try:
            frame_num_str = frame.split('_')[-1].split('.')[0]
            frame_num = int(frame_num_str)

            # Add frames before and after
            for offset in range(-seconds_before, seconds_after + 1):
                new_frame_num = max(0, frame_num + offset)
                new_frame = f"frames/frame_{new_frame_num:04d}.jpg"
                expanded_frames.add(new_frame)
        except (ValueError, IndexError):
            # If parsing fails, just add the original frame
            expanded_frames.add(frame)

    return sorted(list(expanded_frames))

def critic_vlm_prompt(question, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context, ces_logs=None, criteria=None, answer_criteria=None, candidates=None, subtitles=None):
    """
    Critic VLM prompt - evaluates answer based on specific criteria

    Args:
        criteria: Optional list of verification criteria from the question
        answer_criteria: Optional list of criteria specific to the chosen answer
        candidates: Optional list of answer choices
        subtitles: Optional dict mapping frame_path -> subtitle_text
    """
    # Expand frames to include ~5 seconds before and after
    expanded_frames = _expand_frames_with_surrounding(evidence_frame_numbers)

    criteria_section = ""
    if criteria and len(criteria) > 0:
        criteria_list = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(criteria)])
        criteria_section = f"""

QUESTION-BASED VERIFICATION CRITERIA:
The following specific conditions were identified from the question and must be verified:
{criteria_list}
"""

    answer_criteria_section = ""
    if answer_criteria and len(answer_criteria) > 0:
        answer_criteria_list = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(answer_criteria)])
        answer_criteria_section = f"""

ANSWER-SPECIFIC CRITERIA (CRITICAL):
The following criteria validate whether the chosen answer is correct:
{answer_criteria_list}

⚠️ CRITICAL: If ANY answer-specific criterion FAILS, the confidence MUST be ≤50%.
"""

    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    ces_logs_section = ""
    if ces_logs:
        ces_logs_section = f"\nCharacter/Event/Scene logs: {ces_logs}"

    subtitles_section = ""
    if subtitles and len(subtitles) > 0:
        subtitles_list = "\n".join([f"  {frame}: \"{text}\"" for frame, text in subtitles.items()])
        subtitles_section = f"""

SUBTITLES IN EVIDENCE FRAMES:
The following subtitles appear in the video frames (these are visible text/captions):
{subtitles_list}

⚠️ IMPORTANT: Use subtitle information when the question asks about text, words, captions, or specific phrases shown in the video.
"""

    return f"""
You are a critic model evaluating whether an answer to a video question is correct.

Question: {question}{candidates_text}
Given Answer: {answer}
Evidence frames: {evidence_frame_numbers}
Expanded frames (including ~5sec before/after): {expanded_frames}
General context: {general_context}{ces_logs_section}{subtitles_section}{criteria_section}{answer_criteria_section}

Your task is to critically examine whether the answer matches what the question is asking for. Look for:
- Whether the scenes match the location/setting described in the question
- Whether the evidence time frames make sense given the question (EG: if the question asks about a timestamp, is that timestamp's frame queried?)
- Whether all conditions in the question are satisfied
- Whether the answer choice matches the visual evidence
{f"- Whether each of the {len(criteria)} question-based criteria are satisfied" if criteria else ""}
{f"- Whether each of the {len(answer_criteria)} ANSWER-SPECIFIC criteria are satisfied (CRITICAL)" if answer_criteria else ""}
- Check frames BEFORE and AFTER the evidence frames to understand context

Create a VLM query to verify whether the given answer is correct based ONLY on the visual evidence.

Return your VLM query request in this JSON format:
{{
    "tool": "VLM_QUERY",
    "frames": {expanded_frames},
    "prompt": "Based on these frames (including context before/after), verify if the answer '{answer}' is correct for the question: {question}.{f' Check the question criteria: {criteria}.' if criteria else ''}{f' CRITICAL: Check these answer-specific criteria: {answer_criteria}.' if answer_criteria else ''} Pay attention to timestamps and temporal sequences."
}}

IMPORTANT:
- Return ONLY valid JSON
- Use the expanded frames list which includes context frames
- DO NOT rely on any provided reasoning - evaluate purely from visual evidence
- Frame paths must include "frames/" prefix and ".jpg" suffix
"""

def critic_followup_prompt(json_output, question, answer, reasoning, evidence_frame_numbers, general_context, ces_logs=None, criteria=None, answer_criteria=None, candidates=None, subtitles=None):
    """
    Critic followup prompt - evaluates based on criteria pass/fail

    Args:
        criteria: Optional list of question-based verification criteria
        answer_criteria: Optional list of answer-specific criteria
        candidates: Optional list of answer choices
        subtitles: Optional dict mapping frame_path -> subtitle_text
    """
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    subtitles_section = ""
    if subtitles and len(subtitles) > 0:
        subtitles_list = "\n".join([f"  {frame}: \"{text}\"" for frame, text in subtitles.items()])
        subtitles_section = f"""

SUBTITLES IN EVIDENCE FRAMES:
{subtitles_list}

⚠️ IMPORTANT: Use subtitle information to verify text-based questions or questions about visible captions/words.
"""

    # Combine criteria with answer_criteria for evaluation
    has_criteria = (criteria and len(criteria) > 0) or (answer_criteria and len(answer_criteria) > 0)

    if has_criteria:
        # Build criteria sections
        question_criteria_section = ""
        if criteria and len(criteria) > 0:
            criteria_list = "\n".join([f"  Q{i+1}. {c}" for i, c in enumerate(criteria)])
            question_criteria_section = f"""
QUESTION-BASED CRITERIA:
{criteria_list}
"""

        answer_criteria_section = ""
        if answer_criteria and len(answer_criteria) > 0:
            answer_criteria_list = "\n".join([f"  A{i+1}. {c}" for i, c in enumerate(answer_criteria)])
            answer_criteria_section = f"""
ANSWER-SPECIFIC CRITERIA (CRITICAL):
{answer_criteria_list}

⚠️ CRITICAL RULE: If ANY answer-specific criterion FAILS, confidence IMMEDIATELY drops to 50% maximum.
"""

        total_criteria = (len(criteria) if criteria else 0) + (len(answer_criteria) if answer_criteria else 0)

        ces_logs_section = ""
        if ces_logs:
            ces_logs_section = f"\nCharacter/Event/Scene logs: {ces_logs}\n"

        return f"""
Critically assess whether the VLM verification supports the given answer by evaluating each criterion.

VLM Verification Results:
{json_output}

Question Being Answered: {question}{candidates_text}
Given Answer: {answer}
Evidence Frames Used: {evidence_frame_numbers}
{ces_logs_section}{subtitles_section}
{question_criteria_section}{answer_criteria_section}

CRITICAL ASSESSMENT TASK:
For EACH criterion listed above, determine whether it PASSES or FAILS based ONLY on what the VLM actually reports seeing in the frames.

- PASS: The VLM results clearly confirm this criterion is satisfied in the frames
- FAIL: The VLM results indicate this criterion is NOT satisfied, or there is insufficient evidence

CONFIDENCE CALCULATION RULES:
1. Count how many criteria pass (criteria_percentage = criteria_passed / {total_criteria})
2. Calculate base_confidence = criteria_percentage * 100
3. ⚠️ If ANY answer-specific criterion (A1, A2, etc.) FAILS, set confidence = min(50, base_confidence)

⚠️ CRITICAL: There is ALWAYS a correct answer among the choices provided. If all answers seem slightly off or imperfect, you MUST choose the BEST possible answer that most closely matches the evidence.

CRITIC ANSWER CHOICE DECISION:
4. After evaluating criteria, if criteria_percentage > 0.5 (>50% of criteria pass):
   - Review ALL answer choices (A, B, C, D, etc.)
   - Based ONLY on the VLM visual evidence, determine which answer choice is MOST correct
   - Output your critic_answer_choice as a NUMBER (0=A, 1=B, 2=C, 3=D)
5. If criteria_percentage ≤ 0.5 (≤50% of criteria pass):
   - Set critic_answer_choice to -1 (indicating frames may not be the right scene)
   - Set suggestion to "These may not be the right frames/scene. Continue searching for better evidence."

Return ONLY this JSON:
{{
    "criteria_results": [
        {{"criterion": "criterion text", "criterion_type": "QUESTION or ANSWER", "status": "PASS or FAIL", "reasoning": "brief explanation"}},
        ...
    ],
    "question_criteria_passed": <number of question criteria that passed>,
    "answer_criteria_passed": <number of answer criteria that passed>,
    "answer_criteria_total": {len(answer_criteria) if answer_criteria else 0},
    "criteria_passed": <total number of criteria that passed>,
    "criteria_total": {total_criteria},
    "criteria_percentage": <criteria_passed / {total_criteria}>,
    "base_confidence": <(criteria_passed / {total_criteria}) * 100>,
    "confidence": <base_confidence OR 50 if any answer criterion failed>,
    "critic_answer_choice": <NUMBER: 0=A, 1=B, 2=C, 3=D, or -1 if ≤50% criteria pass>,
    "critic_reasoning": "<Brief explanation of why you chose this answer based on VLM evidence>",
    "possible_errors": [
        <List specific criteria that failed or issues found>
    ],
    "suggestion": <"These may not be the right frames" if ≤50%, otherwise null or specific improvement>
}}

IMPORTANT RULES:
- Evaluate EACH criterion independently based on VLM results
- Mark criterion as PASS only if VLM clearly confirms it
- Mark criterion as FAIL if not confirmed or contradicted
- If ANY answer-specific criterion fails, confidence MUST be ≤50%
- Base evaluation ONLY on VLM verification results, not speculation
- Only list errors for criteria that actually failed
- Return ONLY valid JSON
"""
    else:
        # Original non-criteria evaluation (fallback)
        ces_logs_section = ""
        if ces_logs:
            ces_logs_section = f"\nCharacter/Event/Scene logs: {ces_logs}\n"

        return f"""
Critically assess whether the VLM verification supports the given answer.

VLM Verification Results:
{json_output}

Question Being Answered: {question}{candidates_text}
Given Answer: {answer}
Evidence Frames Used: {evidence_frame_numbers}
{ces_logs_section}{subtitles_section}
CRITICAL ASSESSMENT TASK:
1. Check if the VLM results CONFIRM the given answer
2. Look for CONTRADICTIONS between VLM results and the answer
3. Verify ALL conditions in the question are addressed
4. Check if the correct answer choice was selected

⚠️ CRITICAL: There is ALWAYS a correct answer among the choices provided. If all answers seem slightly off or imperfect, you MUST choose the BEST possible answer that most closely matches the evidence.

Based ONLY on what the VLM actually reports seeing, determine:
- Does the visual evidence support the answer?
- Are there any discrepancies between the answer and visuals?
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
- DO NOT consider any reasoning - evaluate purely from visual evidence vs answer
- Only list errors that are ACTUALLY present in the VLM results
- If VLM confirms the answer, confidence should be high (80-100)
- If VLM contradicts the answer, confidence should be low (0-40)
- Empty possible_errors list [] if no actual errors found
- Do NOT invent errors that aren't supported by the VLM results
- Return ONLY valid JSON
"""

######### PROMPT STRINGS #########
def CES_log_prompt(captions_data):
    return f"""
        Here is a SET of frame-by-frame captions data for the long video. Please read it first: \n {captions_data} \n \n  Please keep track of and add to the character-log, an event-log, and a scene/location log with approximate frame numbers in this section of frames for tracking. Note that when the SCENE changes, you should mark the timestamps, so you have rough ideas of locations throughout the whole video. Same for reecurrent characters, and events. 
        
        IMPORTANT: ONLY append to the logs, including NOTHING else in your response.
        IMPORTANT: The logs should be formatted as follows:
        
        CHARACTERS:
        Person 1 (eg: Middle-aged Male Chef): Description: [eg: Heavyset man with a headband, wearing chef's coat]
        
        Person 2 (eg: Conical-Hat Swordbearer): Description:
        
        SCENES:
        Scene 1 (eg: Snowy Forest) : Description: [eg: Snowy winter forest with lots of black trees and cloudy skies] [Timestamp]

        EVENTS:
        Event 1 (eg: Fight Scene): Description: [eg: Confrontation between Person A and person B, swords pulled and fighting commences, Person A is injured.] [Timestamp]

        In the logs, please add a general description of each character/event/scene so that it is identifiable from the visual frames."""

def global_summary_prompt(captions_data):
    return f"""
            Here is a SET of frame-by-frame captions data for a long video. Please read it first: \n {captions_data} \n \n 

            Please output a global summary that depicts main characters, setting, plot line, vibes, and general details. """
