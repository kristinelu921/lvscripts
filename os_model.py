from model_example_query import (
    query_vlm,
    query_llm,
    query_llm_async,
    query_vlm_kimi_video,
    trim_video_for_kimi
)
from search_frame_captions import batch_embed_query_async, search_captions, search_clip_captions
from search_subtitles import search_subtitles
from extract_fine_grained_frames import extract_fine_grained_for_pipeline, get_video_path_from_id
from prompts import initial_prompt, followup_prompt, response_parsing_prompt, finish_prompt, _expand_frames_with_surrounding
from subtitle_utils import SubtitleLoader
import math
import json
import os
import tempfile
from together import AsyncTogether, Together
#from google import genai
import asyncio
import ast
json_file_lock = asyncio.Lock()

# Import prompts
import prompts as default_prompts
import re

CAPTION_SEARCH_TOPK = 4
CAPTION_SEARCH_MAX_QUERIES = 3
MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_SNIPPET_CHARS = 1000
QUERY_ITERATION_TIMEOUT_SECONDS = 1800
QUERY_CLIP_DEFAULT_FPS = 10
QUERY_CLIP_MIN_FPS = 2
QUERY_CLIP_MAX_FPS = 10
QUERY_CLIP_MAX_DURATION_SECONDS = 12.0
PIPELINE_STATUS_PATH = "tmp/pipeline_status"
VISION_REFUSAL_MARKERS = [
    "i don't see",
    "i do not see",
    "no image",
    "can't see",
    "can you upload",
    "cannot see",
    "could you upload",
    "please upload",
    "need.*image",
    "no video",
    "not visible"
]

def _contains_viz_refusal(text):
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(re.search(marker, lower) for marker in VISION_REFUSAL_MARKERS)


def _coerce_tool_name(tool_name):
    if tool_name is None:
        return None
    if isinstance(tool_name, str):
        tool_clean = tool_name.strip().upper().replace("-", "_")
        if not tool_clean:
            return None
        # Normalize common spelling/spacing variants from model output.
        tool_clean = " ".join(tool_clean.split())
        tool_clean = tool_clean.replace(" ", "_")
        if tool_clean == "QUERYCLIP":
            tool_clean = "QUERY_CLIP"
        elif tool_clean == "CAPTIONSEARCH":
            tool_clean = "CAPTION_SEARCH"
        elif tool_clean == "SUBTITLESEARCH":
            tool_clean = "SUBTITLE_SEARCH"
        elif tool_clean == "FINALANSWER":
            tool_clean = "FINAL_ANSWER"
        elif tool_clean == "EXTRACTFINEGRAINEDFRAMES":
            tool_clean = "EXTRACT_FINE_GRAINED_FRAMES"
        elif tool_clean == "CROPOBJECT":
            tool_clean = "CROP_OBJECT"
        elif tool_clean == "VIEWRECORDS":
            tool_clean = "VIEW_RECORDS"
        return tool_clean
    return str(tool_name).strip().upper()


def _extract_tool_from_loose_text(text):
    """Best-effort extraction of a tool call from non-strict model output."""
    if not isinstance(text, str):
        return None

    # Normalize to a smaller window to avoid regex failures on long outputs.
    content = text.strip()
    if not content:
        return None

    # Direct structured tool-only output inside text.
    tool_match = re.search(r'"?tool"?\s*[:=]\s*["\']?([A-Za-z_\s]+)["\']?', content, flags=re.IGNORECASE)
    if not tool_match:
        return None

    parsed = {
        "tool": _coerce_tool_name(tool_match.group(1))
    }
    if parsed["tool"] is None:
        return None

    # Capture common scalar fields.
    for key in ("start_frame", "end_frame", "fps", "topk", "answer"):
        match = re.search(rf'"?{key}"?\s*[:=]\s*([0-9]+(?:\\.[0-9]+)?)', content, flags=re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                if key in {"start_frame", "end_frame", "fps", "topk"}:
                    num = int(num) if float(num).is_integer() else num
                parsed[key] = num
            except (ValueError, TypeError):
                pass

    # Capture answer text/label if present.
    ans_match = re.search(r'"?answer"?:?\s*["\']?([^",}\\]]+)["\']?', content, flags=re.IGNORECASE)
    if ans_match:
        parsed["answer"] = ans_match.group(1).strip()

    # Capture prompt.
    prompt_match = re.search(r'"?prompt"?:?\s*"([^"]+)"', content, flags=re.IGNORECASE | re.DOTALL)
    if prompt_match:
        parsed["prompt"] = prompt_match.group(1).strip()

    # Capture list-like fields.
    def _extract_list(field):
        match = re.search(
            rf'"?{field}"?\s*:\s*(\[[^\]]*\])',
            content,
            flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            return None
        raw = match.group(1)
        try:
            parsed_list = json.loads(raw)
            if isinstance(parsed_list, list):
                return parsed_list
        except Exception:
            pass
        return None

    for field in ("search_queries", "entries", "frames"):
        items = _extract_list(field)
        if items is not None:
            parsed[field] = items

    query_match = re.search(r'"?query"?\s*:\s*"([^"]+)"', content, flags=re.IGNORECASE)
    if query_match:
        parsed["query"] = query_match.group(1).strip()

    if parsed["tool"] == "CAPTION_SEARCH" and "search_queries" not in parsed and "query" in parsed:
        parsed["search_queries"] = [parsed.get("query")]

    return parsed


def _extract_candidate_answer(answer, candidates):
    if not candidates or not isinstance(answer, str):
        return None

    ans = str(answer).lower()
    stop_words = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "is",
        "are",
        "has",
        "have",
        "were",
        "was",
        "to",
        "it",
        "its",
        "in",
        "on",
        "for",
        "this",
        "that",
        "these",
        "those",
        "same",
        "number"
    }

    def _token_match(token, text):
        t = token.strip().lower()
        if not t:
            return False
        if t in text:
            return True
        if t.endswith("s") and t[:-1] in text:
            return True
        if t.endswith("ies") and t[:-3] + "y" in text:
            return True
        return False

    # Structured "label: count" format; prefer explicit numeric evidence over free text.
    mentions = re.findall(r"([a-z][a-z0-9\s\-]{2,80})\s*[:：]\s*(\d+)", ans)
    if mentions:
        scores = {}
        for segment, _count in mentions:
            segment = segment.strip()
            for idx, candidate in enumerate(candidates):
                candidate_tokens = [
                    t for t in re.findall(r"[a-z0-9]+", str(candidate).lower()) if t not in stop_words
                ]
                if not candidate_tokens:
                    continue
                score = 0
                for token in candidate_tokens:
                    if _token_match(token, segment):
                        score += 1
                if score > 0:
                    scores[idx] = max(scores.get(idx, 0), score + 0.01 * int(_count))
        if scores:
            best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            return best[0][0]

    # Explicit option labels such as "A", "option B", "The answer is C"
    explicit = re.search(r"\b(?:option|answer)\s*([a-e])\b", ans, flags=re.IGNORECASE)
    if explicit:
        idx = ord(explicit.group(1).upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return idx

    # Bracketed / parenthesized/numbered option formats like "A) ...", "b) ..." etc.
    for match in re.finditer(r"\b([a-e])\b", ans):
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return idx

    # Direct candidate text match.
    direct_scores = {}
    for idx, candidate in enumerate(candidates):
        candidate_tokens = [
            t for t in re.findall(r"[a-z0-9]+", str(candidate).lower()) if t not in stop_words
        ]
        if not candidate_tokens:
            continue
        score = 0
        for token in candidate_tokens:
            if _token_match(token, ans):
                score += 1
        if score > 0:
            direct_scores[idx] = score

    if direct_scores:
        best = sorted(direct_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0]

    return None


def _looks_like_tool_response(text):
    return isinstance(text, str) and ("\"tool\"" in text or "'tool'" in text)

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

def _extract_frame_token_paths(token):
    """Normalize frame-like tokens from search results into frame-relative paths."""
    if token is None:
        return []

    if isinstance(token, int):
        token = max(1, int(token))
        return [f"frames/frame_{token:04d}.jpg"]

    if not isinstance(token, str):
        return []

    s = token.strip().strip('"').strip("'")
    if not s:
        return []

    # Allow tokens like frames/frame_0001.jpg directly
    if "frame_" in s and ".jpg" in s:
        return [s if s.startswith("frames/") else s]

    # Timestamp-like tokens such as 00:01:05 or 00:01:05.000 should map to seconds-based frame ids.
    if ":" in s:
        parts = s.split(":")
        if 2 <= len(parts) <= 3:
            try:
                if len(parts) == 3:
                    hh, mm, ss = parts
                    frame_num = int(float(hh)) * 3600 + int(float(mm)) * 60 + int(float(ss))
                else:
                    mm, ss = parts
                    frame_num = int(float(mm)) * 60 + int(float(ss))
                return [f"frames/frame_{max(1, frame_num):04d}.jpg"]
            except (ValueError, TypeError):
                pass

    # Allow clip/range tokens only when explicitly requested as timestamps; clip IDs are handled by QUERY_CLIP, not VLM_QUERY.
    nums = []
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        elif cur:
            nums.append(int(cur))
            cur = ""
    if cur:
        nums.append(int(cur))
    if ("clip" in s and len(nums) >= 2) or (len(nums) >= 2 and "_" in s):
        # Clip-style IDs (clip_45_67 / 45_67) must be resolved via QUERY_CLIP.
        return []

    # Allow plain timestamp numbers
    if nums:
        return [f"frames/frame_{max(1, int(nums[-1])):04d}.jpg"]

    return [s]

def _normalize_vlm_frames(frames, max_frames=48):
    """Normalize user-provided frame names into a bounded list of frame paths."""
    if not frames:
        return []

    normalized = []
    if isinstance(frames, str):
        candidates = [frames]
    else:
        candidates = frames if isinstance(frames, list) else list(frames)

    for token in candidates:
        normalized.extend(_extract_frame_token_paths(token))

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered[:max_frames]


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


def _trim_message_content(message, max_chars=MAX_MESSAGE_SNIPPET_CHARS):
    """Trim long message content while preserving role."""
    content = message.get("content")
    if not isinstance(content, str):
        if isinstance(content, (dict, list)):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except TypeError:
                content = str(content)
        else:
            content = str(content)

    if len(content) <= max_chars:
        return message

    trimmed = content[:max_chars] + f"\n... [truncated {len(content) - max_chars} chars]"
    trimmed_message = dict(message)
    trimmed_message["content"] = trimmed
    return trimmed_message


def _trim_messages_for_prompt(messages, max_messages=MAX_HISTORY_MESSAGES, max_chars=MAX_MESSAGE_SNIPPET_CHARS):
    """Keep only the most recent messages and truncate long message content."""
    selected = messages[-max_messages:] if len(messages) > max_messages else messages
    return [_trim_message_content(m, max_chars=max_chars) for m in selected]


def _trim_retrieved_info(value, max_chars=MAX_MESSAGE_SNIPPET_CHARS):
    """Convert tool outputs to a bounded text payload for follow-up prompts."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)

    return _trim_message_content({"content": value}, max_chars=max_chars)["content"]


def _safe_read_text(path, default=""):
    """Read a context file if present; otherwise return a safe fallback."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _write_pipeline_probe(v_id, question_uid, status, tool=None, note=None, attempt=None, last_error=None):
    """Persist lightweight runtime status for monitoring/debugging long runs."""
    try:
        os.makedirs(PIPELINE_STATUS_PATH, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "video_id": v_id,
            "question_uid": question_uid,
            "status": status,
            "tool": tool,
            "attempt": attempt,
            "note": note,
            "last_error": str(last_error) if last_error is not None else None
        }
        with open(f"{PIPELINE_STATUS_PATH}/pipeline_status_{os.getpid()}.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # Probe writes must never block the pipeline.
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
    
    async def vlm_query(self, image_paths, prompt, batch_size=15):
        result = await query_vlm(self.vlm, image_paths, prompt, batch_size=batch_size)
        return result
        

my_model = Pipeline("moonshotai/Kimi-K2.5", "kimi-k2.5")

async def query_model_iterative_with_retry(model, question, uid, vid_path, output_file, max_retries=15, candidates=None, use_no_vlm=False, videos_dir="/mnt/ssd/data/longvideobench/videos", pass_all_subtitles_to_llm=False, subtitles_dir=None, embeddings_path=None):
    """Wrapper to retry query_model_iterative if it hangs"""
    # Default to frame captions embeddings if not specified
    if embeddings_path is None:
        clip_embeddings = f"{vid_path}/captions/clip_embeddings.jsonl"
        frame_embeddings = f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl"
        if os.path.exists(clip_embeddings):
            embeddings_path = clip_embeddings
        elif os.path.exists(frame_embeddings):
            embeddings_path = frame_embeddings
        else:
            raise FileNotFoundError(f"No caption embedding file found for {vid_path}: expected clip or frame embeddings")
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
                if item.get("uid") == uid:
                    print(f"already completed question {uid}")
                    return item
        except Exception as e:
            print("Checking if q already completed error")

    for attempt in range(max_retries):
        try:
            message = f"Attempt {attempt + 1}/{max_retries} for question: {question[:50]}..."
            log(message, f"logs/log_video_{vid_path}_{uid}")
            # Timeout for the entire iterative process (LLM + tools + retries)
            result = await asyncio.wait_for(
                query_model_iterative(model, question, uid, vid_path, candidates, use_no_vlm=use_no_vlm, videos_dir=videos_dir, pass_all_subtitles_to_llm=pass_all_subtitles_to_llm, subtitles_dir=subtitles_dir, embeddings_path=embeddings_path),
                timeout=QUERY_ITERATION_TIMEOUT_SECONDS
            )
            print(f"Successfully completed on attempt {attempt + 1}")
            if output_file:
                await append_to_json_file(output_file, result)
                await append_to_json_file('completed_uid.json', {"uid": uid})
            return result
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}, retrying...")
            _write_pipeline_probe(os.path.basename(vid_path.rstrip('/')), uid, status="attempt_timeout", attempt=attempt + 1, last_error=f"Timeout after {QUERY_ITERATION_TIMEOUT_SECONDS}s")
            # Reset model state for retry
            model.messages = []
            model.scratchpad = []
            model.records = []
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

                return result
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            _write_pipeline_probe(os.path.basename(vid_path.rstrip('/')), uid, status="attempt_error", attempt=attempt + 1, last_error=str(e))
            # Reset model state for retry
            model.messages = []
            model.scratchpad = []
            model.records = []
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
    # Default to clip captions first, then frame captions
    if embeddings_path is None:
        clip_embeddings = f"{vid_path}/captions/clip_embeddings.jsonl"
        frame_embeddings = f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl"
        if os.path.exists(clip_embeddings):
            embeddings_path = clip_embeddings
        elif os.path.exists(frame_embeddings):
            embeddings_path = frame_embeddings
        else:
            raise FileNotFoundError(f"No caption embedding file found for {vid_path}: expected clip or frame embeddings")

    global_sum_path = vid_path + "/captions/global_summary.txt"
    CES_logs_path = vid_path + "/captions/CES_logs.txt"

    if not os.path.exists(global_sum_path):
        raise FileNotFoundError(f"Missing required context file: {global_sum_path}")
    if not os.path.exists(CES_logs_path):
        raise FileNotFoundError(f"Missing required context file: {CES_logs_path}")

    with open(global_sum_path, "r") as f:
        global_summary = f.read()
    with open(CES_logs_path, "r") as f:
        CES_log = f.read()

    question = question.strip()

    # Variable to store criteria extracted from initial response
    question_criteria = None
    query_clip_used = False
    empty_response_count = 0
    parse_retry_count = 0
    invalid_final_attempts = 0
    last_tool_request = {}

    def _set_followup_prompt_with_note(note):
        return (
            str(_trim_messages_for_prompt(model.messages, max_chars=MAX_MESSAGE_SNIPPET_CHARS))
            + prompts.followup_prompt(
                note,
                question,
                candidates,
                use_subtitles=use_subtitles,
                subtitles_available=subtitles_available
            )
        )

    # Check if subtitle embeddings are available for this video
    video_id = os.path.basename(vid_path.rstrip('/'))
    _write_pipeline_probe(video_id, question_uid, status="question_start", note=f"question_len={len(question)}")

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
        _write_pipeline_probe(video_id, question_uid, status="llm_query_start", attempt=i+1)
        try:
            #print("reached this thing")
            response = await model.llm_query_async(prompt)
            #print("response reached", response)
        except Exception as e:
            error_message = str(e).lower()
            print(f"Failed to get model response: {e}")
            print(f"Error type: {type(e).__name__}")
            if (
                "500" in error_message
                or "server_error" in error_message
                or "internal server error" in error_message
            ):
                print("❗ Model backend returned 5xx; stopping retries for this question to avoid repeated overload.")
                return {
                    "uid": question_uid,
                    "question": question,
                    "answer": "ERROR",
                    "reasoning": f"LLM request failed: {e}",
                    "evidence_frame_numbers": []
                }
            prompt = _set_followup_prompt_with_note(
                (
                    f"Could not get a valid planner response ({str(e)[:180]}). "
                    "Return strict JSON with one valid tool call."
                )
            )
            continue

        # Skip logging/parsing if LLM returned an empty or None response
        if not response:
            reason = "LLM returned empty response"
            print(f"Failed to get response at iteration {i+1}: {reason}")
            empty_response_count += 1
            _write_pipeline_probe(video_id, question_uid, status="tool_llm_empty", attempt=i+1, last_error=reason)
            if empty_response_count >= 2:
                return {
                    "uid": question_uid,
                    "question": question,
                    "answer": "ERROR",
                    "reasoning": reason,
                    "evidence_frame_numbers": []
                }
            model.messages.append({
                "role": "system",
                "content": (
                    "Your previous response was empty. "
                    "Reply with strict JSON that includes a valid tool call in the format your parser expects."
                )
            })
            prompt = _set_followup_prompt_with_note("Your previous response was empty. Return STRICT JSON with a tool call.")
            continue

        # If the model response itself contains server-side failure text, stop immediately.
        response_error = str(response).lower()
        if "500" in response_error or "server_error" in response_error or "internal server error" in response_error:
            print("❗ Model backend returned server-side error text; stopping retries for this question to avoid overload.")
            return {
                "uid": question_uid,
                "question": question,
                "answer": "ERROR",
                "reasoning": f"LLM response indicates server error: {response}",
                "evidence_frame_numbers": []
            }

        model.messages.append({"role": "assistant", "content": response})
        print(f"✓ Added assistant response to messages (total: {len(model.messages)})")
        
        message = response
        log(message, f"logs/log_video_{vid_path}_{question_uid}")
            
        # Parse the response
        if response and "</think>" in response:
            response = response.split("</think>")[1].strip()

        print("reached here")

        if _contains_viz_refusal(response):
            refusal_note = (
                "The previous response is a model artifact about missing uploads. "
                "You do have clip context. Do not ask for images/videos. "
                "Return the next strict JSON tool call immediately."
            )
            model.messages.append({"role": "system", "content": refusal_note})
            _write_pipeline_probe(video_id, question_uid, status="tool_parse_refusal", attempt=i+1, note="response indicated missing attachment")
            prompt = _set_followup_prompt_with_note(refusal_note)
            continue

        def extract_json(text):
            if not text:
                return None

            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
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
                                try:
                                    return json.loads(json_str)
                                except Exception:
                                    pass
                                try:
                                    parsed = ast.literal_eval(json_str)
                                    if isinstance(parsed, (dict, list)):
                                        return parsed
                                except (ValueError, SyntaxError):
                                    pass

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

            # Strategy 7: Parse JSON-like Python dict/list output
            try:
                parsed = ast.literal_eval(text.strip())
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (ValueError, SyntaxError):
                pass

            return None

        # Prefer direct JSON extraction from the model response to avoid extra parser calls.
        original_response_text = response
        parsed_response = extract_json(response)
        if parsed_response is None:
            # Heuristic recovery first to avoid parser LLM timeouts/failures.
            recovered = _extract_tool_from_loose_text(response)
            if isinstance(recovered, dict) and recovered.get("tool"):
                parsed_response = recovered
            else:
                # Last-resort parser LLM call if strict recovery fails.
                parsing_prompt = prompts.response_parsing_prompt(response)
                try:
                    _write_pipeline_probe(video_id, question_uid, status="tool_parse_retry", attempt=i+1)
                    parser_resp = await model.llm_query_async(parsing_prompt)
                except Exception as e:
                    parser_error = str(e).strip()
                    if not parser_error:
                        parser_error = type(e).__name__
                    print(f"Failed to parse assistant response at iteration {i+1}: {parser_error}")
                    parse_retry_count += 1
                    if parse_retry_count >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": f"Parser LLM failed: {parser_error}",
                            "evidence_frame_numbers": []
                        }
                    _write_pipeline_probe(video_id, question_uid, status="tool_parse_error", attempt=i+1, last_error=f"Parser LLM failed: {parser_error}")
                    model.messages.append({
                        "role": "system",
                        "content": (
                            "Your response was not machine-parsable. "
                            "Please reply with STRICT JSON that includes a 'tool' field only. "
                            "Do not claim missing images/videos if context exists."
                        )
                    })
                    prompt = _set_followup_prompt_with_note(
                        "Parser failed on assistant response. Reply with strict JSON containing a valid 'tool' field."
                    )
                    continue
                if not parser_resp:
                    parsed_response = None
                else:
                    parsed_response = extract_json(parser_resp)
                    if parsed_response is None:
                        recovered = _extract_tool_from_loose_text(parser_resp)
                        if isinstance(recovered, dict):
                            parsed_response = recovered
                if parsed_response is None:
                    print(f"Parser response could not be converted to JSON at iteration {i+1}")
                    parse_retry_count += 1
                    if parse_retry_count >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": "LLM parser returned unparsable response",
                            "evidence_frame_numbers": []
                        }
                    model.messages.append({
                        "role": "system",
                        "content": (
                            "Could not parse the parser response. "
                            "Reply with strict JSON containing only a valid tool object."
                        )
                    })
                    prompt = _set_followup_prompt_with_note(
                        "Parser response was not valid JSON. Return strict JSON with one valid tool call."
                    )
                    continue

            if not parsed_response:
                print(f"Failed to get parsing response at iteration {i+1}")
                parse_retry_count += 1
                if parse_retry_count >= 2:
                    return {
                        "uid": question_uid,
                        "question": question,
                        "answer": "ERROR",
                        "reasoning": "LLM parser returned empty response",
                        "evidence_frame_numbers": []
                    }
                model.messages.append({
                    "role": "system",
                    "content": (
                        "Your previous response to the parser prompt was empty. "
                        "Reply with strict JSON only, containing a valid 'tool' field."
                    )
                })
                prompt = _set_followup_prompt_with_note(
                    "Parser output was empty. Return strict JSON with a valid tool call."
                )
                continue
        if parsed_response is None:
            print(f"Failed to extract JSON from response")
            print(f"Raw response: {original_response_text[:500]}...")  # Print first 500 chars
            _write_pipeline_probe(video_id, question_uid, status="tool_parse_error", attempt=i+1, last_error=f"extract_json failed: {original_response_text[:200]}")

            # Try one more lightweight heuristic pass before failing.
            recovered = _extract_tool_from_loose_text(original_response_text)
            if isinstance(recovered, dict) and recovered.get("tool"):
                parsed_response = recovered
            else:
                parse_retry_count += 1
                if parse_retry_count >= 2:
                    return {
                        "uid": question_uid,
                        "question": question,
                        "answer": "ERROR",
                        "reasoning": "Failed to extract JSON from assistant response",
                        "evidence_frame_numbers": []
                    }
                if _looks_like_tool_response(original_response_text):
                    model.messages.append({
                        "role": "assistant",
                        "content": original_response_text
                    })
                    model.messages.append({
                        "role": "system",
                        "content": (
                            "Parser can't read your tool call. Return the same JSON tool object only "
                            "(no explanation, no markdown)."
                        )
                    })
                else:
                    model.messages.append({
                        "role": "system",
                        "content": "Could not extract JSON from planner output. Return strict JSON with a valid tool call."
                    })
                prompt = _set_followup_prompt_with_note(
                    "Parser could not extract JSON. Return strict JSON with one valid tool call."
                )
                continue

        if not isinstance(parsed_response, dict):
            print(f"Parser produced non-dict response (type={type(parsed_response).__name__}): {str(parsed_response)[:200]}")
            parse_retry_count += 1
            if parse_retry_count >= 2:
                return {
                    "uid": question_uid,
                    "question": question,
                    "answer": "ERROR",
                    "reasoning": f"Parser returned non-dict response type={type(parsed_response).__name__}",
                    "evidence_frame_numbers": []
                }
            model.messages.append({
                "role": "system",
                "content": f"Tool call parser returned invalid JSON object shape ({type(parsed_response).__name__}). Please output strict JSON with at least a tool field next time."
            })
            prompt = _set_followup_prompt_with_note(
                f"Parser returned invalid type {type(parsed_response).__name__}. Return strict JSON with valid tool."
            )
            continue

        # Debug: Print detected tool and normalize it
        detected_tool = parsed_response.get("tool", "UNKNOWN")
        detected_tool = _coerce_tool_name(detected_tool)
        if detected_tool is not None:
            parsed_response["tool"] = detected_tool

        # Recovery: infer a clip query when tool tag is missing but query fields are present.
        if (parsed_response.get("tool") in (None, "", "UNKNOWN") and
                parsed_response.get("start_frame") is not None and parsed_response.get("end_frame") is not None):
            parsed_response["tool"] = "QUERY_CLIP"
            _write_pipeline_probe(
                video_id,
                question_uid,
                status="tool_inferred_query_clip",
                attempt=i + 1,
                note="inferred query_clip_from_missing_tool"
            )
            detected_tool = "QUERY_CLIP"
        print(f"🔧 Detected tool: {detected_tool}")

        try:
            if parsed_response.get("tool") == "FINAL_ANSWER":
                _write_pipeline_probe(video_id, question_uid, status="tool_final_answer", attempt=i+1)
                # The parsed response has "frames" field, not "evidence_frame_numbers"
                message = f"FINAL_ANSWER: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # Convert answer to int format (handle both letter and number responses)
                parsed_answer = parsed_response.get("answer")
                inferred_answer = _extract_candidate_answer(parsed_answer, candidates)
                if isinstance(inferred_answer, int):
                    parsed_answer = inferred_answer
                if isinstance(parsed_answer, str):
                    parsed_answer = parsed_answer.strip().upper()
                    if parsed_answer in ["NO IMAGE OR VIDEO PROVIDED", "NO IMAGE PROVIDED", "NO VIDEO PROVIDED", "I CANNOT SEE THE VIDEO"]:
                        invalid_final_attempts += 1
                        if invalid_final_attempts >= 2:
                            return {
                                "uid": question_uid,
                                "question": question,
                                "answer": "ERROR",
                                "reasoning": "Rejected invalid FINAL_ANSWER claims without image/video",
                                "evidence_frame_numbers": []
                            }
                        model.messages.append({
                            "role": "system",
                            "content": "Invalid FINAL_ANSWER: do not claim missing image/video. You already have access to QUERY_CLIP outputs and must provide a numeric answer."
                        })
                        _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error="Final answer claimed missing image/video")
                        continue
                    # Normalize punctuation and extract a clean token.
                    parsed_answer_clean = re.sub(r"[^A-Z0-4]", "", parsed_answer)
                    if len(parsed_answer_clean) == 1 and parsed_answer_clean in ['A', 'B', 'C', 'D', 'E']:
                        parsed_answer = parsed_answer_clean
                    else:
                        match = re.search(r"\b([0-4])\b", parsed_answer)
                        if match:
                            parsed_answer = match.group(1)
                        else:
                            parsed_answer = parsed_answer_clean

                    # If it's a letter (A, B, C, D, E), convert to number
                    if parsed_answer in ['A', 'B', 'C', 'D', 'E']:
                        parsed_answer = ord(parsed_answer) - ord('A')
                    # If it's already a number string, convert to int
                    elif parsed_answer.isdigit():
                        parsed_answer = int(parsed_answer)
                    else:
                        invalid_final_attempts += 1
                        if invalid_final_attempts >= 2:
                            return {
                                "uid": question_uid,
                                "question": question,
                                "answer": "ERROR",
                                "reasoning": "Final answer must be a number index (0-4) or letter A-E",
                                "evidence_frame_numbers": []
                            }
                        model.messages.append({
                            "role": "system",
                            "content": "Invalid FINAL_ANSWER: answer must be a number index (0-4) or letter A-E."
                        })
                        _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error="Final answer not numeric/letter format")
                        continue
                elif isinstance(parsed_answer, int):
                    # Already an int, keep as is
                    pass
                elif inferred_answer is not None:
                    parsed_answer = inferred_answer
                else:
                    invalid_final_attempts += 1
                    if invalid_final_attempts >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": "Final answer must be a number index (0-4) or letter A-E",
                            "evidence_frame_numbers": []
                        }
                    model.messages.append({
                        "role": "system",
                        "content": "Invalid FINAL_ANSWER: answer must be a number index (0-4) or letter A-E."
                    })
                    _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error="Final answer type invalid")
                    continue

                if isinstance(parsed_answer, int) and not (0 <= parsed_answer <= 4):
                    invalid_final_attempts += 1
                    if invalid_final_attempts >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": f"Final answer index out of range: {parsed_answer}",
                            "evidence_frame_numbers": []
                        }
                    model.messages.append({
                        "role": "system",
                        "content": "Invalid FINAL_ANSWER: numeric answer must be between 0 and 4."
                    })
                    _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error=f"Final answer out of range: {parsed_answer}")
                    continue
                elif not isinstance(parsed_answer, int):
                    invalid_final_attempts += 1
                    if invalid_final_attempts >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": "Final answer must be a number index (0-4) or letter A-E",
                            "evidence_frame_numbers": []
                        }
                    model.messages.append({
                        "role": "system",
                        "content": "Invalid FINAL_ANSWER: answer must be a number index (0-4) or letter A-E."
                    })
                    _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error="Final answer type invalid")
                    continue

                new_response = {
                    "uid": question_uid,
                    "question": question,
                    "answer": parsed_answer,
                    "reasoning": parsed_response.get("reasoning"),
                    "evidence_frame_numbers": parsed_response.get("frames")  # Map "frames" to "evidence_frame_numbers"
                }

                if query_clip_used and not new_response["evidence_frame_numbers"]:
                    invalid_final_attempts += 1
                    if invalid_final_attempts >= 2:
                        return {
                            "uid": question_uid,
                            "question": question,
                            "answer": "ERROR",
                            "reasoning": "Final answer missing evidence frames after QUERY_CLIP usage",
                            "evidence_frame_numbers": []
                        }
                    model.messages.append({
                        "role": "system",
                        "content": "Invalid FINAL_ANSWER: you used QUERY_CLIP. Return frames/evidence from the clip rather than an empty evidence list."
                    })
                    _write_pipeline_probe(video_id, question_uid, status="tool_invalid_final_answer", attempt=i+1, last_error="No evidence frames despite QUERY_CLIP")
                    continue

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
                _write_pipeline_probe(video_id, question_uid, status="tool_vlm_query", attempt=i+1, note=f"frames={len(parsed_response.get('frames', [])) if parsed_response.get('frames') else 0}")
                message = f"VLM_QUERY: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print("parsed response: ", parsed_response)
                print("="*60 + "Querying VLM" + "="*60)

                # Get requested frames and expand them to include ±5 seconds of context
                frames = parsed_response.get("frames")
                if not frames:
                    frames = []
                frames = _normalize_vlm_frames(frames)
                if not frames:
                    message = (
                        "Invalid VLM_QUERY: no usable frame filenames were provided. "
                        "For CAPTION_SEARCH clip IDs (e.g., clip_45_67), use QUERY_CLIP instead."
                    )
                    model.messages.append({"role": "system", "content": message})
                    log(message, f"logs/log_video_{vid_path}_{question_uid}")
                    continue

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
                _write_pipeline_probe(video_id, question_uid, status="tool_caption_search", attempt=i+1, note=f"queries={search_queries if 'search_queries' in locals() else 'n/a'}")
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

                if search_queries and len(search_queries) > CAPTION_SEARCH_MAX_QUERIES:
                    print(f"Trimming CAPTION_SEARCH queries from {len(search_queries)} to {CAPTION_SEARCH_MAX_QUERIES}")
                    search_queries = search_queries[:CAPTION_SEARCH_MAX_QUERIES]

                print(f"Performing {len(search_queries)} search queries: {search_queries}")
                message = f"Search queries ({len(search_queries)}): {search_queries}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")

                # Perform all searches and collect results
                all_results = []
                caption_search_failed = False
                # Detect if using clip captions based on embeddings path
                use_clip_search = 'clip_embeddings' in str(embeddings_path)

                for idx, query in enumerate(search_queries):
                    print(f"  Query {idx+1}/{len(search_queries)}: {query}")
                    # Use appropriate search function based on caption type
                    try:
                        if use_clip_search:
                            results = await search_clip_captions(vid_path, question_uid, query, embeddings_path, CAPTION_SEARCH_TOPK)
                        else:
                            results = await search_captions(vid_path, question_uid, query, embeddings_path, CAPTION_SEARCH_TOPK)
                    except Exception as e:
                        message = f"CAPTION_SEARCH failed for query '{query}': {e}"
                        print(f"⚠️ {message}")
                        retrieved_info = (
                            "Caption search encountered an error: "
                            f"{e}\n"
                            "You can either try a different query, switch to QUERY_CLIP to sample a short segment, "
                            "or continue with direct VLM reasoning using the global summary."
                        )
                        model.messages.append({"role": "system", "content": retrieved_info})
                        _write_pipeline_probe(video_id, question_uid, status="tool_caption_search_error", attempt=i+1, last_error=str(e))
                        caption_search_failed = True
                        break

                    all_results.append({
                        "query": query,
                        "results": results if isinstance(results, list) else [results]
                    })

                if caption_search_failed:
                    continue

                # Format results for LLM to process (trimmed to top-k highlights)
                retrieved_info_parts = [f"Retrieved top {CAPTION_SEARCH_TOPK} results from each of {len(search_queries)} caption queries.\n"]
                retrieved_info_parts.append("These are the only caption hits added to model context. If you need more, run another CAPTION_SEARCH.\n")

                for idx, query_result in enumerate(all_results):
                    retrieved_info_parts.append(f"\n--- Query {idx+1}: \"{query_result['query']}\" ---")
                    results = query_result['results']
                    if isinstance(results, list) and len(results) > 0:
                        top_results = results[:CAPTION_SEARCH_TOPK]
                        highlighted = []
                        for rank, hit in enumerate(top_results, start=1):
                            item = dict(hit) if isinstance(hit, dict) else {"result": hit}
                            item["rank"] = rank
                            if "similarity score" in item:
                                item["ranked_similarity"] = round(float(item["similarity score"]), 4)
                            highlighted.append(item)
                        retrieved_info_parts.append(json.dumps(highlighted, indent=2))
                    else:
                        retrieved_info_parts.append("No results")

                retrieved_info = "\n".join(retrieved_info_parts)

                message = f"Caption search completed: {len(search_queries)} queries executed"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                model.messages.append({"role": "caption search results", "content": retrieved_info})

            elif parsed_response.get("tool") == "RECORD":
                _write_pipeline_probe(video_id, question_uid, status="tool_record", attempt=i+1)
                # Record relevant observations for organizational tracking
                entries = parsed_response.get("entries", [])
                if not isinstance(entries, list):
                    entries = [entries]  # Convert single entry to list

                print(f"Recording {len(entries)} event(s)")
                for entry in entries:
                    # Parse time from entry (format: "Time: XX seconds, Event: ...")
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
                _write_pipeline_probe(video_id, question_uid, status="tool_view_records", attempt=i+1)
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
                _write_pipeline_probe(video_id, question_uid, status="tool_subtitle_search", attempt=i+1, note=f"query={parsed_response.get('query', '')}")
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
                _write_pipeline_probe(video_id, question_uid, status="tool_extract_fine_grained", attempt=i+1)
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

            elif parsed_response.get("tool") == "QUERY_CLIP":
                query_clip_used = True
                _write_pipeline_probe(video_id, question_uid, status="tool_query_clip", attempt=i+1, note=f"start={parsed_response.get('start_frame')} end={parsed_response.get('end_frame')}")
                message = f"QUERY_CLIP: {parsed_response}"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print("="*60 + "Querying clip segment" + "="*60)

                start_frame = parsed_response.get("start_frame")
                end_frame = parsed_response.get("end_frame")
                prompt = parsed_response.get("prompt", "Describe the action sequence in this clip segment.")
                fps = parsed_response.get("fps", QUERY_CLIP_DEFAULT_FPS)

                # Validate required inputs
                try:
                    start_sec = float(start_frame)
                    end_sec = float(end_frame)
                except (TypeError, ValueError):
                    model.messages.append({"role": "system", "content": "Invalid QUERY_CLIP: 'start_frame' and 'end_frame' must be numeric."})
                    continue

                if start_sec < 0 or end_sec < 0:
                    model.messages.append({"role": "system", "content": "Invalid QUERY_CLIP: start_frame/end_frame must be non-negative."})
                    continue
                if end_sec <= start_sec:
                    model.messages.append({"role": "system", "content": f"Invalid QUERY_CLIP range: end_frame ({end_sec}) must be greater than start_frame ({start_sec})."})
                    continue

                try:
                    fps = int(fps)
                except (TypeError, ValueError):
                    fps = QUERY_CLIP_DEFAULT_FPS
                fps = max(QUERY_CLIP_MIN_FPS, min(QUERY_CLIP_MAX_FPS, fps))

                # Keep clip analysis short so VLM payload remains tractable.
                duration = end_sec - start_sec
                if duration > QUERY_CLIP_MAX_DURATION_SECONDS:
                    end_sec = start_sec + QUERY_CLIP_MAX_DURATION_SECONDS
                    model.messages.append({
                        "role": "system",
                        "content": f"QUERY_CLIP duration was truncated from {duration:.2f}s to {QUERY_CLIP_MAX_DURATION_SECONDS:.1f}s."
                    })

                if "kimi" in str(model.vlm).lower() and "/" not in str(model.vlm):
                    clip_prompt = (
                        f"Question: {question}\n\n"
                        f"{prompt}\n\n"
                        f"Important context: the provided video is a trimmed standalone segment from the original "
                        f"timeline, covering approximately {start_sec:.2f}s to {end_sec:.2f}s. "
                        f"Treat this as the only evidence window."
                    )

                    try:
                        print(f"QUERY_CLIP request: video={video_id}, segment=({start_sec:.2f}, {end_sec:.2f}), fps={fps}")
                        source_video = get_video_path_from_id(video_id, videos_dir)
                        if not source_video:
                            raise FileNotFoundError(f"Raw video not found for {video_id} in {videos_dir}")

                        clip_output = tempfile.NamedTemporaryFile(
                            suffix=".mp4",
                            prefix=f"{video_id}_",
                            delete=False
                        )
                        clip_output.close()
                        raw_clip_path = trim_video_for_kimi(
                            input_file=source_video,
                            start_second=start_sec,
                            end_second=end_sec,
                            output_file=clip_output.name
                        )
                        try:
                            retrieved_info = await query_vlm_kimi_video(model.vlm, raw_clip_path, clip_prompt, temperature=1.0)
                            print(f"QUERY_CLIP VLM preview: {retrieved_info[:160]}...")
                            print(f"QUERY_CLIP VLM response length: {len(retrieved_info)}")
                        finally:
                            if os.path.exists(raw_clip_path):
                                os.remove(raw_clip_path)
                        model.messages.append({"role": "query clip results", "content": retrieved_info})
                    except Exception as e:
                        model.messages.append({"role": "system", "content": f"QUERY_CLIP VLM analysis failed: {e}"})
                    continue

                try:
                    frame_paths = extract_fine_grained_for_pipeline(
                        video_id=video_id,
                        start_second=start_sec,
                        end_second=end_sec,
                        fps=fps,
                        videos_dir=videos_dir,
                        output_base=os.path.dirname(os.path.dirname(vid_path)),
                        vid_path=vid_path
                    )
                except FileNotFoundError as e:
                    model.messages.append({"role": "system", "content": f"QUERY_CLIP failed: video file not found ({e})"})
                    continue
                except RuntimeError as e:
                    model.messages.append({"role": "system", "content": f"QUERY_CLIP failed: {e}"})
                    continue
                except Exception as e:
                    model.messages.append({"role": "system", "content": f"QUERY_CLIP failed with unexpected error: {e}"})
                    continue

                if not frame_paths:
                    model.messages.append({"role": "system", "content": "QUERY_CLIP extracted no frames from this range. Try a smaller range."})
                    continue

                clip_frames = [(f"{vid_path}/" + frame_path) for frame_path in frame_paths]
                clip_prompt = (
                    f"{prompt}\n\n"
                    f"You are viewing {len(clip_frames)} clip frames sampled at {fps} FPS "
                    f"covering approximately {start_sec:.2f}s to {end_sec:.2f}s."
                )

                try:
                    retrieved_info = await model.vlm_query(clip_frames, clip_prompt)
                    model.messages.append({"role": "query clip results", "content": retrieved_info})
                except Exception as e:
                    model.messages.append({"role": "system", "content": f"QUERY_CLIP VLM analysis failed: {e}"})

            elif parsed_response.get("tool") == "CROP_OBJECT":
                _write_pipeline_probe(video_id, question_uid, status="tool_crop_object", attempt=i+1, note=f"frame={parsed_response.get('frame','')}")
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
                _write_pipeline_probe(video_id, question_uid, status="tool_unknown", attempt=i+1, note=f"tool={parsed_response.get('tool')}")
                tool_name = parsed_response.get('tool')
                message = f"Invalid or unrecognized tool: '{tool_name}' (type: {type(tool_name)}, repr: {repr(tool_name)})"
                log(message, f"logs/log_video_{vid_path}_{question_uid}")
                print(f"❌ {message}")
                print(f"   Valid tools: FINAL_ANSWER, VLM_QUERY, CAPTION_SEARCH, QUERY_CLIP, RECORD, VIEW_RECORDS, SUBTITLE_SEARCH, EXTRACT_FINE_GRAINED_FRAMES, CROP_OBJECT")
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
            elif parsed_response.get("tool") == "QUERY_CLIP":
                retrieved_info = "QUERY_CLIP analyzed a short sampled segment and returns motion-aware observations for that time window.\n" + str(model.messages[-1].get("content", ""))
            elif parsed_response.get("tool") == "CROP_OBJECT":
                retrieved_info = "Objects have been detected and cropped from the frame. These cropped images show ONLY the detected objects with fine details. Use VLM_QUERY with the cropped image paths to analyze specific object details.\n" + str(model.messages[-1].get("content", ""))
            else:
                retrieved_info = str(model.messages[-1].get("content", ""))

            retrieved_info = _trim_retrieved_info(retrieved_info)

            # CRITICAL: Include full conversation history in prompt
            trimmed_messages = _trim_messages_for_prompt(model.messages)
            prompt = str(trimmed_messages) + prompts.followup_prompt(retrieved_info, question, candidates, use_subtitles=use_subtitles, subtitles_available=subtitles_available)
            print(f"✓ Updated prompt for next iteration (length: {len(prompt)} chars, messages: {len(model.messages)})")
        except Exception as e:
            message = f"Error updating prompt: {e}"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")
            print(f"Error updating prompt: {e}")
            continue
    
    # Return final formatted scratchpad
    final_prompt = finish_prompt(_trim_messages_for_prompt(model.messages, max_chars=MAX_MESSAGE_SNIPPET_CHARS), candidates)
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
                inferred = _extract_candidate_answer(answer, candidates)
                if isinstance(inferred, int):
                    answer = inferred

                # Try to match answer against candidates if provided
                if candidates:
                    # Try exact match first
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
            inferred = _extract_candidate_answer(answer, candidates)
            if isinstance(inferred, int):
                answer = inferred

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

    answer = final_answer
    inferred = _extract_candidate_answer(final_answer, candidates)
    if isinstance(inferred, int):
        answer = inferred

    message = model.messages
    log(message, f"logs/log_video_{vid_path}_{question_uid}")

    # Try to match answer against candidates if provided
    if candidates:
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

async def answer_question(question_uid, question, vid_folder, vid_num, candidates=None, vlm_model="kimi-k2.5", llm_model="moonshotai/Kimi-K2.5", use_no_vlm=False, videos_dir="/mnt/ssd/data/longvideobench/videos", pass_all_subtitles_to_llm=False, subtitles_dir=None, embeddings_path=None):
    try:
        # Create a separate Pipeline instance for each question to avoid shared state
        #qwen model : Qwen/Qwen3-235B-A22B-Instruct-2507-tput
        model = Pipeline(llm_model, vlm_model)
        curr_folder = str(vid_folder)
        num = vid_num
        vid_path = curr_folder + "/" + num
        required_context = [
            f"{vid_path}/captions/global_summary.txt",
            f"{vid_path}/captions/CES_logs.txt",
        ]
        missing_context = [path for path in required_context if not os.path.exists(path)]
        if missing_context:
            raise FileNotFoundError(f"Missing required context file(s): {', '.join(missing_context)}")
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
