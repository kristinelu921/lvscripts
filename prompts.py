import json
import os
from model_example_query import query_llm, query_llm_async

def initial_prompt(question, candidates=None, use_subtitles=False, subtitles_available=False):
    """Initial prompt for video QA pipeline"""
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    subtitles_note = ""
    if use_subtitles and subtitles_available:
        subtitles_note = "\n\nSUBTITLES AVAILABLE: Use SUBTITLE_SEARCH when the question quotes text, mentions dialogue, or asks about spoken words (e.g., 'when the subtitle says...')."

    return f"""You are answering questions about a video using tools to retrieve information.

Question: {question}{candidates_text}{subtitles_note}

STEP 1: CHECK QUESTION TYPE
- Subtitle question? (quotes text, asks about dialogue) → Use SUBTITLE_SEARCH first
- Temporal question? (after/before/when/first/next/last) → Use RECORD after each VLM_QUERY to track events
- Quick actions? (hand movements, fast motions, sudden reactions) → Use EXTRACT_FINE_FRAMES at 5-10 FPS
- Small details? (small objects, text, fine features) → Use CROP_OBJECT to zoom in

STEP 2: EXTRACT VERIFICATION CRITERIA
List 4-6 specific, testable visual criteria from the QUESTION (not answer choices) that must be verified.

Example: "person in black shirt holding metal bucket with scattered debris"
Criteria:
- Person wearing black shirt
- Person wearing black pants
- Person holding large round metal bucket
- Scene contains scattered debris on ground

STEP 2.5: PLAN TO TEST EACH ANSWER CHOICE
Once you have frames, plan to systematically test EACH answer option with VLM queries:
- Query to verify option A characteristics
- Query to verify option B characteristics
- Query to verify option C characteristics
- Query to verify option D characteristics
This prevents confirmation bias and ensures you select the answer with STRONGEST visual evidence.

STEP 3: GENERATE SEARCH QUERIES
Create 2-4 natural search queries (subject-verb-object format) targeting VISIBLE details.
- Good: "woman wearing yellow dress holding papers", "person in office setting with documents"
- Bad: "yellow dress papers office woman"

IMPORTANT: After getting clip time ranges from CAPTION_SEARCH, you MUST use VLM_QUERY to visually verify frames in those time ranges.
DO NOT repeatedly search captions - once you have candidate clips/time ranges, USE VLM_QUERY to look at the frames!

NOTE: CAPTION_SEARCH returns clip time ranges (e.g., "clip_45_67" = 45-67 seconds).
Use these time ranges to extract frames for VLM_QUERY (e.g., check frames 45-67: frame_0045.jpg to frame_0067.jpg).

Return format:
```json{{
    "tool": "CAPTION_SEARCH",
    "criteria": ["criterion 1", "criterion 2", "criterion 3", "criterion 4"],
    "search_queries": ["natural query 1", "natural query 2", "optional query 3", "optional query 4"]
}}```

For subtitle questions: {{"tool": "SUBTITLE_SEARCH", "query": "exact subtitle text", "topk": 10}}
For video clip extraction: {{"tool": "QUERY_CLIP", "start_frame": 45, "end_frame": 52, "prompt": "Describe the action"}}
"""

def followup_prompt(json_output, question, candidates=None, use_subtitles=False, subtitles_available=False):
    """Followup prompt for video QA pipeline"""
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    subtitles_note = ""
    if use_subtitles and subtitles_available:
        subtitles_note = "\n\nSUBTITLE SEARCH: Use when question quotes text or asks about dialogue."

    return f"""Information received:
{json_output}

Question: {question}{candidates_text}{subtitles_note}

KEY REMINDERS:
- Frame IDs are timestamps: frame_0050.jpg = 50 seconds
- AFTER = larger frame numbers, BEFORE = smaller frame numbers
- Default 1 FPS may miss fast actions → use EXTRACT_FINE_FRAMES for quick motions
- VLM observations are GROUND TRUTH (caption searches locate clip time ranges, then you query frames)
- TEMPORAL QUESTIONS: Use RECORD after each VLM_QUERY to log timestamps and events (improves accuracy +7%)

🔥 CRITICAL: If you just received clip time ranges from CAPTION_SEARCH, USE VLM_QUERY NOW to look at frames in those ranges!
DO NOT do another CAPTION_SEARCH. Caption search is only for FINDING relevant time ranges/clips, VLM_QUERY is for ANALYZING the frames.

TOOL USAGE GUIDE:

1. VLM_QUERY - Visual verification and analysis [USE THIS AFTER GETTING FRAMES!]
   WHEN TO USE:
   ✓✓✓ IMMEDIATELY after receiving frames from CAPTION_SEARCH or SUBTITLE_SEARCH
   ✓ Need to verify what's actually visible in specific frames
   ✓ Have candidate frames from caption/subtitle search
   ✓ Need to answer questions about visual details (colors, actions, objects, settings)
   ✓ Checking temporal sequences (what happens before/after an event)
   ✗ DON'T USE: As first action (search first to find relevant frames)
   ✗ DON'T USE: Without specific frames identified

   Format: {{"tool": "VLM_QUERY", "frames": ["frames/frame_0102.jpg", ...], "prompt": "..."}}

   BEST PRACTICE - Test Each Answer Choice:
   When you have candidate frames, query the VLM to test EACH answer choice systematically:
   - "Is this [option A]?" → Check frames
   - "Is this [option B]?" → Check frames
   - "Is this [option C]?" → Check frames
   This prevents confirmation bias and ensures you pick the BEST match, not just the first plausible one.

   → NEXT: Use RECORD after VLM_QUERY on temporal questions to log timestamps

2. RECORD - Log temporal observations (REQUIRED for temporal questions)
   WHEN TO USE:
   ✓ After each VLM_QUERY on temporal questions (after/before/when/first/next/last)
   ✓ When you observe an event that needs timestamp tracking
   ✓ Building timeline of events across multiple VLM queries
   ✗ DON'T USE: On simple non-temporal questions (e.g., "What color is the shirt?")
   ✗ DON'T USE: Before doing VLM_QUERY (nothing to record yet)

   Format: {{"tool": "RECORD", "entries": ["Time: 45 seconds, Event: Woman picks up papers", ...]}}
   → System auto-displays VIEW_RECORDS after recording

3. CAPTION_SEARCH - Find relevant time ranges by searching clip descriptions
   WHEN TO USE:
   ✓ As FIRST action on most questions (locates candidate time ranges/clips)
   ✓ Current frames don't match the question AFTER verifying with VLM_QUERY
   ✓ Need to find specific scenes, objects, people, or actions
   ✓ Looking for visual concepts (not spoken words)
   ✗ DON'T USE: For subtitle/dialogue questions (use SUBTITLE_SEARCH)
   ✗ DON'T USE: When you already have good candidate frames
   ✗ DON'T USE: Repeatedly without using VLM_QUERY first (you must LOOK at frames before searching again!)

   Format: {{"tool": "CAPTION_SEARCH", "search_queries": ["woman in yellow dress holding papers", "office setting with documents"]}}
   Use 2-4 natural subject-verb-object queries

   HOW CLIP CAPTIONS WORK:
   - Search results return clip time ranges (e.g., "clip_45_67" = 45-67 second clip)
   - Each clip has a detailed caption describing the visual content during that time period
   - Use the time ranges to extract frames for VLM_QUERY (e.g., if clip is 45-67s, check frames 45-67)
   - Clips are automatically extracted at scene boundaries, typically 2-120 seconds long
   - Example: Search finds "clip_120_145" → Use VLM_QUERY on frames 120-145 (e.g., frame_0120.jpg to frame_0145.jpg)

   FRAMES FOR VLM ANALYSIS:
   - Frames are extracted at 1 FPS for the entire video
   - After finding relevant clips (e.g., 45-67s), query frames in that range
   - Frame naming: frame_0050.jpg = 50 seconds into video
   - You can query any frame(s) within the clip's time range for detailed analysis

4. SUBTITLE_SEARCH - Find frames by searching spoken dialogue
   WHEN TO USE:
   ✓ Question quotes exact text ("when subtitle says...", "after the narrator mentions...")
   ✓ Question asks about dialogue, spoken words, or narration
   ✓ Need to find when specific phrases were said
   ✗ DON'T USE: For visual questions (use CAPTION_SEARCH)
   ✗ DON'T USE: When subtitles not available
   ✗ DON'T USE: For on-screen text visible in video (use CROP_OBJECT instead)

   Format: {{"tool": "SUBTITLE_SEARCH", "query": "exact subtitle phrase", "topk": 10}}

5. EXTRACT_FINE_GRAINED_FRAMES - Get higher frame rate for fast actions
   WHEN TO USE:
   ✓ Question about quick hand movements, gestures, or fast motions
   ✓ Facial expressions or quick reactions
   ✓ Fast-paced action scenes where 1 FPS misses critical moments
   ✓ Need to see exact moment of transition or change
   ✗ DON'T USE: For slow-paced scenes (wastes tokens)
   ✗ DON'T USE: As first action (search for time range first)
   ✗ DON'T USE: For entire video (specify 2-10 second ranges only)

   Format: {{"tool": "EXTRACT_FINE_GRAINED_FRAMES", "start_second": 45.0, "end_second": 47.0, "fps": 5}}
   Use 5-10 FPS for short intervals (2-5 seconds)

6. CROP_OBJECT - Zoom in to see small details
   WHEN TO USE:
   ✓ Question about small objects hard to see in full frame
   ✓ Reading text, labels, signs in the video
   ✓ Identifying distant or tiny details (birds, small items, fine features)
   ✓ Need closer view to verify specific characteristics
   ✗ DON'T USE: For large, clearly visible objects
   ✗ DON'T USE: Without knowing which frame contains the object
   ✗ DON'T USE: For spoken subtitles (use SUBTITLE_SEARCH)

   Format: {{"tool": "CROP_OBJECT", "frame": "frames/frame_0050.jpg", "object_query": "bird on branch"}}

7. QUERY_CLIP - Extract and analyze a specific video clip segment
   WHEN TO USE:
   ✓ Need to see continuous motion or action across multiple frames
   ✓ Question about transitions, sequences, or temporal changes
   ✓ Want to analyze video clip instead of static frames
   ✓ Need to see smooth motion between two timestamps
   ✗ DON'T USE: For single-frame analysis (use VLM_QUERY instead)
   ✗ DON'T USE: Without knowing the approximate time range

   Format: {{"tool": "QUERY_CLIP", "start_frame": 45, "end_frame": 52, "prompt": "Describe the action sequence"}}
   Frame numbers are timestamps in seconds (e.g., frame_0045.jpg = 45 seconds)

8. FINAL_ANSWER - Submit answer (only when CERTAIN)
   WHEN TO USE:
   ✓ Have VLM evidence from specific frames
   ✓ Verified all question criteria
   ✓ Can confidently choose between answer choices
   ✓ Have timestamps for temporal questions (use RECORD first)
   ✗ DON'T USE: Without VLM_QUERY verification first
   ✗ DON'T USE: When uncertain (do more searches/queries)
   ✗ DON'T USE: On temporal questions without using RECORD at least once

   CRITICAL ANSWER SELECTION PROCESS:
   → Step 1: REREAD ALL ANSWER CHOICES carefully
   → Step 2: TEST EACH CHOICE against VLM evidence (not just your preferred answer)
      - Did you verify option A? What does VLM show?
      - Did you verify option B? What does VLM show?
      - Did you verify option C? What does VLM show?
      - Did you verify option D? What does VLM show?
   → Step 3: COMPARE which choice has STRONGEST visual support
   → Step 4: Choose the BEST match based on evidence (there is ALWAYS a correct answer)

   Answer must be NUMBER only (0=A, 1=B, 2=C, 3=D, 4=E):
   {{
       "tool": "FINAL_ANSWER",
       "frames": ["frames/frame_0001.jpg", ...],
       "answer": "1",  // NUMBER ONLY: 0=A, 1=B, 2=C, 3=D, 4=E
       "reasoning": "...",
       "answer_criteria": ["Validates my chosen answer", "Confirms answer matches evidence"]
   }}
   → Include enough frames for independent verification of your answer
"""

def finish_prompt(scratchpad, candidates=None):
    """Final answer prompt"""
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    return f"""Given all information:
{scratchpad}

Determine final answer.{candidates_text}

ANSWER SELECTION PROCESS:
1. REREAD all answer choices
2. TEST each choice systematically against VLM evidence:
   - What visual evidence supports option A?
   - What visual evidence supports option B?
   - What visual evidence supports option C?
   - What visual evidence supports option D?
3. COMPARE which option has STRONGEST visual support
4. Select answer with BEST evidence match (there is ALWAYS a correct answer)

Return format:
{{
    "answer": "1",  // NUMBER: 0=A, 1=B, 2=C, 3=D, 4=E
    "frames": ["frames/frame_0001.jpg", ...],
    "reasoning": "...",
    "answer_criteria": ["Validates chosen answer"]
}}
"""

def response_parsing_prompt(response):
    """Parse JSON from LLM response"""
    return f"""Extract JSON from this response. Must be ONE of: VLM_QUERY, RECORD, VIEW_RECORDS, CAPTION_SEARCH, SUBTITLE_SEARCH, EXTRACT_FINE_GRAINED_FRAMES, CROP_OBJECT, QUERY_CLIP, or FINAL_ANSWER.

Response: {response}

Return ONLY valid JSON. Examples:

VLM_QUERY: {{"tool": "VLM_QUERY", "frames": ["frames/frame_0001.jpg"], "prompt": "..."}}
RECORD: {{"tool": "RECORD", "entries": ["Time: 45 seconds, Event: ..."]}}
SUBTITLE_SEARCH: {{"tool": "SUBTITLE_SEARCH", "query": "...", "topk": 10}}
EXTRACT_FINE_GRAINED_FRAMES: {{"tool": "EXTRACT_FINE_GRAINED_FRAMES", "start_second": 45.0, "end_second": 47.0, "fps": 5}}
CROP_OBJECT: {{"tool": "CROP_OBJECT", "frame": "frames/frame_0050.jpg", "object_query": "bird on branch"}}
QUERY_CLIP: {{"tool": "QUERY_CLIP", "start_frame": 45, "end_frame": 52, "prompt": "Describe the action sequence"}}
CAPTION_SEARCH: {{"tool": "CAPTION_SEARCH", "search_queries": ["query 1", "query 2"]}}
FINAL_ANSWER: {{"tool": "FINAL_ANSWER", "frames": ["frames/frame_0001.jpg"], "answer": "0", "reasoning": "...", "answer_criteria": ["..."]}}

Frame paths must include "frames/" prefix and ".jpg" suffix.
"""

async def reformat_answers(answers_path):
    """Reformat answers to standard JSON format"""
    if os.path.exists(f'{answers_path[:-5]}_reformatted.json'):
        with open(f'{answers_path[:-5]}_reformatted.json', 'r') as f:
            return json.load(f)

    with open(answers_path, "r") as f:
        answers = str(f.read())

    prompt = f"""Reformat these answers to valid JSON array:
{answers}

Format:
[
  {{"uid": "...", "question": "...", "answer": "A", "frames": ["frames/frame_0001.jpg", ...], "reasoning": "..."}},
  ...
]

Return ONLY valid JSON. Answer must be single capitalized letter (A, B, C, etc.).
"""

    result = await query_llm_async("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", prompt)

    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        return json.loads(result)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse reformatted answers as JSON")
        return result

def verifier_prompt(question, answer, reasoning, evidence_frame_numbers, general_context):
    """Verifier initial prompt"""
    return f"""Verify this answer using VLM.

Question: {question}
Answer: {answer}
Reasoning: {reasoning}
Evidence frames: {evidence_frame_numbers}
Context: {general_context}

Create VLM request to verify answer correctness:
{{
    "tool": "VLM_QUERY",
    "frames": ["frames/frame_0001.jpg", ...],
    "prompt": "..."
}}

Return ONLY valid JSON.
"""

def verifier_followup_prompt(json_output, question, answer, reasoning, evidence_frame_numbers, general_context):
    """Verifier followup prompt"""
    return f"""Based on VLM verification, evaluate confidence in answer.

Original Question: {question}
Given Answer: {answer}
Original Reasoning: {reasoning}

Return ONLY JSON:
{{
    "uid": "...",
    "question": "{question}",
    "answer": "{answer}",
    "confidence": <0-100>,
    "confidence_reasoning": "..."
}}

Use EXACT question and answer text provided. Confidence must be 0-100.
"""

def _expand_frames_with_surrounding(evidence_frame_numbers, seconds_before=5, seconds_after=5):
    """Expand frame list to include frames ~5 seconds before and after each frame"""
    expanded_frames = set()
    for frame in evidence_frame_numbers:
        try:
            frame_num_str = frame.split('_')[-1].split('.')[0]
            frame_num = int(frame_num_str)
            for offset in range(-seconds_before, seconds_after + 1):
                new_frame_num = max(0, frame_num + offset)
                new_frame = f"frames/frame_{new_frame_num:04d}.jpg"
                expanded_frames.add(new_frame)
        except (ValueError, IndexError):
            expanded_frames.add(frame)
    return sorted(list(expanded_frames))

def critic_vlm_prompt(question, answer, reasoning, evidence_frame_numbers, vid_dir, num, general_context, ces_logs=None, criteria=None, answer_criteria=None, candidates=None, subtitles=None):
    """Critic VLM prompt - evaluates answer based on criteria"""
    expanded_frames = _expand_frames_with_surrounding(evidence_frame_numbers)

    criteria_section = ""
    if criteria:
        criteria_list = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(criteria)])
        criteria_section = f"\n\nQUESTION CRITERIA:\n{criteria_list}"

    answer_criteria_section = ""
    if answer_criteria:
        answer_criteria_list = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(answer_criteria)])
        answer_criteria_section = f"\n\nANSWER CRITERIA (CRITICAL - if ANY fails, confidence ≤50%):\n{answer_criteria_list}"

    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    ces_logs_section = f"\n\nLogs: {ces_logs}" if ces_logs else ""

    subtitles_section = ""
    if subtitles:
        subtitles_list = "\n".join([f"  {frame}: \"{text}\"" for frame, text in subtitles.items()])
        subtitles_section = f"\n\nSUBTITLES:\n{subtitles_list}"

    return f"""Evaluate whether answer is correct.

Question: {question}{candidates_text}
Given Answer: {answer}
Evidence frames: {evidence_frame_numbers}
Expanded frames (±5sec): {expanded_frames}
Context: {general_context}{ces_logs_section}{subtitles_section}{criteria_section}{answer_criteria_section}

Verify:
- Scene matches question location/setting
- Evidence timeframes make sense
- All question conditions satisfied
- Answer choice matches visual evidence
{f"- All {len(criteria)} question criteria satisfied" if criteria else ""}
{f"- All {len(answer_criteria)} ANSWER criteria satisfied (CRITICAL)" if answer_criteria else ""}
- Context before/after frames

Return:
{{
    "tool": "VLM_QUERY",
    "frames": {expanded_frames},
    "prompt": "Verify if '{answer}' is correct for: {question}.{f' Check question criteria: {criteria}.' if criteria else ''}{f' CRITICAL: Check answer criteria: {answer_criteria}.' if answer_criteria else ''} Check timestamps and temporal sequences."
}}

Evaluate from visual evidence only, ignore provided reasoning.
"""

def critic_followup_prompt(json_output, question, answer, reasoning, evidence_frame_numbers, general_context, ces_logs=None, criteria=None, answer_criteria=None, candidates=None, subtitles=None):
    """Critic followup - evaluates criteria pass/fail"""
    candidates_text = ""
    if candidates:
        candidates_text = "\n\nAnswer Choices:\n"
        for i, choice in enumerate(candidates):
            candidates_text += f"{chr(65+i)}. {choice}\n"

    subtitles_section = ""
    if subtitles:
        subtitles_list = "\n".join([f"  {frame}: \"{text}\"" for frame, text in subtitles.items()])
        subtitles_section = f"\n\nSUBTITLES:\n{subtitles_list}"

    has_criteria = (criteria and len(criteria) > 0) or (answer_criteria and len(answer_criteria) > 0)

    if has_criteria:
        question_criteria_section = ""
        if criteria:
            criteria_list = "\n".join([f"  Q{i+1}. {c}" for i, c in enumerate(criteria)])
            question_criteria_section = f"\nQUESTION CRITERIA:\n{criteria_list}"

        answer_criteria_section = ""
        if answer_criteria:
            answer_criteria_list = "\n".join([f"  A{i+1}. {c}" for i, c in enumerate(answer_criteria)])
            answer_criteria_section = f"\nANSWER CRITERIA (if ANY fails, confidence ≤50%):\n{answer_criteria_list}"

        total_criteria = (len(criteria) if criteria else 0) + (len(answer_criteria) if answer_criteria else 0)
        ces_logs_section = f"\nLogs: {ces_logs}" if ces_logs else ""

        return f"""Assess VLM verification by evaluating each criterion.

VLM Results: {json_output}
Question: {question}{candidates_text}
Answer: {answer}
Frames: {evidence_frame_numbers}{ces_logs_section}{subtitles_section}{question_criteria_section}{answer_criteria_section}

TASK: For EACH criterion, determine PASS or FAIL based on VLM results.

CONFIDENCE RULES:
1. criteria_percentage = criteria_passed / {total_criteria}
2. base_confidence = criteria_percentage * 100
3. If ANY answer criterion (A1, A2) FAILS → confidence = min(50, base_confidence)

CRITIC ANSWER CHOICE:
4. If criteria_percentage > 0.5: Review ALL choices, pick MOST correct based on VLM evidence (NUMBER: 0=A, 1=B, 2=C, 3=D, 4=E)
5. If criteria_percentage ≤ 0.5: Still pick best guess, set suggestion to "May not be right frames/scene"

There is ALWAYS a correct answer. Pick the BEST match even if imperfect.

Return ONLY JSON:
{{
    "criteria_results": [{{"criterion": "...", "criterion_type": "QUESTION or ANSWER", "status": "PASS or FAIL", "reasoning": "..."}}],
    "question_criteria_passed": <num>,
    "answer_criteria_passed": <num>,
    "answer_criteria_total": {len(answer_criteria) if answer_criteria else 0},
    "criteria_passed": <num>,
    "criteria_total": {total_criteria},
    "criteria_percentage": <float>,
    "base_confidence": <float>,
    "confidence": <float>,
    "critic_answer_choice": <NUMBER: 0=A, 1=B, 2=C, 3=D, 4=E>,
    "critic_reasoning": "...",
    "possible_errors": [...],
    "suggestion": <"May not be right frames" if ≤50%, else null>
}}

Mark PASS only if VLM clearly confirms. Base evaluation on VLM results only.
"""
    else:
        # Fallback non-criteria evaluation
        ces_logs_section = f"\nLogs: {ces_logs}" if ces_logs else ""

        return f"""Assess VLM verification.

VLM Results: {json_output}
Question: {question}{candidates_text}
Answer: {answer}
Frames: {evidence_frame_numbers}{ces_logs_section}{subtitles_section}

Check:
1. VLM confirms given answer
2. Contradictions between VLM and answer
3. All question conditions addressed
4. Correct answer choice selected

There is ALWAYS a correct answer. Pick BEST match even if imperfect.

Return ONLY JSON:
{{
    "confidence": <0-100>,
    "possible_errors": [<actual discrepancies found>],
    "suggestion": <null if >80, else specific improvement>
}}

Base confidence on VLM results only. List only ACTUAL errors, not hypothetical. Empty [] if no errors.
"""

def CES_log_prompt(captions_data):
    """Generate character/event/scene logs from clip captions"""
    return f"""Clip captions with timestamps: {captions_data}

Track characters, events, and scenes with their time ranges from the clip captions.
Each caption represents a continuous clip segment (e.g., [45s-67s] means 45 to 67 seconds).

Format:
CHARACTERS:
Person 1 (e.g., Middle-aged Male Chef): Description: [e.g., Heavyset man with headband, chef's coat] [Time ranges where character appears]

SCENES:
Scene 1 (e.g., Snowy Forest): Description: [e.g., Snowy winter forest, black trees, cloudy] [Time range: start-end seconds]

EVENTS:
Event 1 (e.g., Fight Scene): Description: [e.g., Confrontation, swords drawn, Person A injured] [Time range: start-end seconds]

Use the timestamp ranges from the clip captions to accurately track when characters, scenes, and events occur.
ONLY append to logs, nothing else.
"""

def global_summary_prompt(captions_data):
    """Generate global video summary from clip captions"""
    return f"""Clip captions with timestamps: {captions_data}

Each caption describes a continuous video clip segment with a time range (e.g., [45s-67s]).
Use these clip descriptions to create a comprehensive summary of the entire video.

Output global summary including:
- Main characters and their roles
- Setting and locations
- Plot/narrative arc
- Overall mood and atmosphere
- Key events and their timing
- General details about the video content
"""
