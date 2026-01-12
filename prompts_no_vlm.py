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
You are a reasoning model whose goal is to answer questions about a video. You cannot see the video, but you can use tools to retrieve information about the video through caption search.

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
- If question asks for a "SEQUENCE" of events, you should order events in frame time order
- Mark down and track ALL timestamps/frame numbers
- Pay special attention to temporal keywords (after, before, during, when, then, etc.)

⚠️ CAPTION SEARCH STRATEGY ⚠️
Caption searches help you LOCATE potentially relevant frames based on semantic similarity.
- Search results show which frames have captions most similar to your queries
- Higher similarity scores indicate better matches
- Look for clusters or overlaps across multiple search queries
- Consider temporal ordering when selecting frames

Now, please choose to do EXACTLY ONE of the following:

1. Choose to caption search with 2-4 natural, structured queries. You should CHOOSE this option if you don't have CLEAR evidence that previously identified frames fit the constraints of the question. Generate multiple queries targeting DIFFERENT visual aspects.

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

2. Determine your final answer based on the caption search results you have retrieved. ONLY CHOOSE THIS OPTION if you're SURE of your answer based on the caption similarities and frame information. If you are AT ALL unsure, or the results don't make full sense, you should SEARCH with different queries. Bias to option 1 unless you have CLEAR evidence. Make sure the letter and the answer match the answer choices given in the question.

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
            "reasoning": "Based on caption search results, explain your reasoning here. Reference frame numbers and caption similarities.",
            "answer_criteria": [
                "First criterion validating your chosen answer",
                "Second criterion validating your chosen answer (optional)"
            ]
        }}
    - IMPORTANT: If you choose the "FINAL_ANSWER" tool, choose AT LEAST as many frames as necessary to answer the questions such that a separate entity could use only these frames to deduce/confirm your answer. Bias towards extra frames if you are unsure.
    - IMPORTANT: In the "FINAL_ANSWER" tool, your answer MUST be a NUMBER (eg: 0, 1, 2, 3, 4, etc.) ONLY and nothing else, corresponding to the answer choices (0 -> A, 1 -> B, 2 -> C, 3 -> D, etc.) given in the question.
    - IMPORTANT: Frame paths must include "frames/" prefix and ".jpg" suffix
    - IMPORTANT: if a question asks you about a specific timestamp, convert it to seconds and identify the frames at THOSE seconds, in the right format.
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

IMPORTANT: You must also generate 1-2 ANSWER-SPECIFIC CRITERIA that verify your chosen answer is correct.
These criteria should validate whether the specific answer choice you selected matches the available evidence.

Return your answer in the following json format:
json_output = {{
    "answer": "..." (eg: "A"),
    "frames": [list of frame numbers],
    "reasoning": "Based on the caption search results and evidence gathered, explain your reasoning here.",
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
Extract the JSON object from this response. The response contains ONE of these tools: CAPTION_SEARCH or FINAL_ANSWER.

Response to parse:
{response}

Return ONLY a JSON object, nothing else. The JSON must follow one of these exact formats:

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
    - Frame paths must include "frames/" prefix and ".jpg" suffix

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
