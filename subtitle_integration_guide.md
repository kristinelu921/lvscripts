# Subtitle Integration Guide

## Overview
This guide explains how to integrate subtitles into the video QA pipeline with a True/False parameter.

## Files Modified
1. **prompts.py** - Updated to support subtitle parameters
2. **os_model.py** - Needs to handle SUBTITLE_SEARCH tool (see below)

## Integration Steps

### 1. Load Subtitle Frame Mapping

Add to the beginning of `answer_question()` function in os_model.py:

```python
def load_subtitle_frames(video_id, subtitles_path="/mnt/ssd/data/longvideobench/subtitles_frame_mapping.json"):
    """Load subtitle frame mapping for a video

    Returns:
        dict: Frame number -> subtitle text mapping, or empty dict if not available
    """
    try:
        with open(subtitles_path, 'r') as f:
            all_subtitles = json.load(f)

        if video_id in all_subtitles and 'frames' in all_subtitles[video_id]:
            # Convert string frame keys to integers
            frames_dict = all_subtitles[video_id]['frames']
            return {int(k): v for k, v in frames_dict.items()}
        return {}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}
```

### 2. Update answer_question() Function

Add subtitle support parameters:

```python
async def answer_question(question_uid, question, vid_folder, vid_num,
                         candidates=None, vlm_model="...", llm_model="...",
                         use_no_vlm=False, use_subtitles=True):  # Add use_subtitles parameter
    """
    Answer a question about a video

    Args:
        ...
        use_subtitles: Whether to enable subtitle search (default: True)
    """
    # Load subtitle frames for this video
    subtitle_frames = load_subtitle_frames(vid_num) if use_subtitles else {}
    subtitles_available = len(subtitle_frames) > 0

    # Log subtitle availability
    if use_subtitles:
        if subtitles_available:
            print(f"✓ Subtitles available: {len(subtitle_frames)} frames with subtitles")
        else:
            print(f"⚠️  No subtitles found for video {vid_num}")
```

### 3. Pass Subtitle Parameters to Prompts

Update the prompt calls in os_model.py to include subtitle parameters:

```python
# In query_model_iterative function, update:
prompt = str(model.messages) + prompts.initial_prompt(
    question, candidates,
    use_subtitles=use_subtitles,
    subtitles_available=subtitles_available
)

# Later in the loop, update followup_prompt:
prompt = prompts.followup_prompt(
    retrieved_info, question, candidates,
    use_subtitles=use_subtitles,
    subtitles_available=subtitles_available
)
```

### 4. Handle SUBTITLE_SEARCH Tool

Add handler in the tool processing section of os_model.py:

```python
elif parsed_response.get("tool") == "SUBTITLE_SEARCH":
    query = parsed_response.get("query", "")
    print(f"SUBTITLE_SEARCH: {query}")

    if not use_subtitles or not subtitle_frames:
        # Subtitles not available, fallback to caption search
        print("⚠️  Subtitles not available, using caption search instead")
        retrieved_info = "Subtitles are not available for this video. Please use CAPTION_SEARCH instead."
        model.messages.append({"role": "system", "content": retrieved_info})
    else:
        # Search subtitles for matching frames
        from subtitle_search_tool import search_video_subtitles, format_results_for_llm

        try:
            # Search using the subtitle search tool
            results = search_video_subtitles(
                video_id=vid_num,
                query=query,
                topk=10,
                fps=1.0
            )

            # Format results for LLM
            retrieved_info = format_results_for_llm(results)

            # Log the search
            message = f"SUBTITLE_SEARCH completed: found {len(results)} results"
            log(message, f"logs/log_video_{vid_path}_{question_uid}")

            model.messages.append({"role": "subtitle search results", "content": retrieved_info})
        except Exception as e:
            print(f"❌ Subtitle search error: {e}")
            retrieved_info = f"Subtitle search failed: {str(e)}. Please use CAPTION_SEARCH instead."
            model.messages.append({"role": "system", "content": retrieved_info})
```

### 5. Update test_pipeline.py

Add subtitle support to the test pipeline:

```python
parser.add_argument('--use-subtitles', action='store_true', default=False,
                   help='Enable subtitle search functionality')

# In PipelineTester.__init__:
self.use_subtitles = use_subtitles

# When calling answer_question:
answer = await answer_question(
    question_uid=q['uid'],
    question=q['question'],
    vid_folder=self.video_folder,
    vid_num=video_id,
    candidates=q.get('candidates'),
    vlm_model=self.vlm_model,
    llm_model=self.llm_model,
    use_no_vlm=self.use_no_vlm,
    use_subtitles=self.use_subtitles  # Pass subtitle flag
)
```

## Usage Examples

### Command Line Usage

```bash
# Run pipeline with subtitles enabled
python test_pipeline.py /path/to/videos --use-subtitles

# Caption generation with subtitles
python caption_frames_query_aware.py /path/to/videos --use-subtitles --query-aware

# Test subtitle search
python subtitle_search_tool.py Y0IaijKNGX8 "master chief arrives" --topk 5
```

### Programmatic Usage

```python
# With subtitles
result = await answer_question(
    question_uid="test_001",
    question="When the subtitle says 'temples scattered throughout', what is the woman wearing?",
    vid_folder="/mnt/ssd/data/longvideobench/videos_processed_1",
    vid_num="9PD3ciudpIE",
    candidates=["red", "blue", "yellow", "green"],
    use_subtitles=True
)

# Without subtitles
result = await answer_question(
    question_uid="test_002",
    question="What is the person doing in the video?",
    vid_folder="/mnt/ssd/data/longvideobench/videos_processed_1",
    vid_num="9PD3ciudpIE",
    candidates=["running", "walking", "sitting", "jumping"],
    use_subtitles=False
)
```

## Data Format

### Subtitle Frame Mapping (`subtitles_frame_mapping.json`)

```json
{
  "video_id_1": {
    "frames": {
      "3": "Ibiza Spain the world's most famous",
      "5": "Party Island didn't think I'd ever make",
      "7": "a video about this but here we go this"
    }
  },
  "video_id_2": {
    "frames": {
      "10": "Another subtitle text",
      "15": "More subtitle content"
    }
  }
}
```

### Subtitle Search Results Format

```
Subtitle Search Results:

[1] Time: 3:42 (Frame 222-225)
    Score: 0.892
    Text: the conference happening about central

[2] Time: 5:18 (Frame 318-320)
    Score: 0.845
    Text: you're interested in the latest developments
```

## Implementation Status

- ✅ Subtitle extraction and frame mapping (extract_subtitles_and_questions.py)
- ✅ Subtitle search tool (subtitle_search_tool.py)
- ✅ Subtitle embeddings (embed_subtitles.py)
- ✅ Prompts updated with subtitle support (prompts.py)
- ⚠️  OS model integration needed (os_model.py) - see steps above
- ⚠️  Test pipeline integration needed (test_pipeline.py) - see steps above

## Notes

- Subtitle search requires embeddings to be generated first: `python embed_subtitles.py`
- Not all videos have subtitles - the system gracefully falls back to caption search
- Frame numbers in subtitle mapping correspond to seconds (frame_0050 = 50 seconds)
- Query-aware captions can include subtitles inline: use `--use-subtitles` flag
