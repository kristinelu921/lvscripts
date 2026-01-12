# Long Video Understanding Scripts

## Quick Start

### 1. Extract Subtitles and Questions
```bash
python extract_subtitles_and_questions.py
```
This creates:
- `subtitles_frame_mapping.json` - Frame-to-subtitle mappings
- `downloaded_videos_questions.json` - Questions for your videos

### 2. Choose Your Captioning Method

#### Option A: Standard Captioning (General Purpose)
```bash
python caption_frames_os.py <video_folder>
```

#### Option B: Query-Aware Captioning (Recommended for QA)
```bash
python caption_frames_query_aware.py <video_folder>
```

#### Option C: With Subtitles (Best Performance)
```bash
# Standard + subtitles
python caption_frames_os.py <video_folder> --use-subtitles

# Query-aware + subtitles (BEST)
python caption_frames_query_aware.py <video_folder> --use-subtitles
```

### 3. Answer Questions
```bash
python os_model.py <video_folder>
```

### 4. Verify Answers (Optional)
```bash
python critic_model_os.py <video_folder>
```

---

## What's New

### ✨ Subtitle Integration
- Subtitles are now appended to frame captions
- Use `--use-subtitles` flag with captioning scripts
- Format: `[VLM caption] | Subtitle: [subtitle text]`

### ✨ Query-Aware Captioning
- New script: `caption_frames_query_aware.py`
- Feeds all questions to VLM before captioning
- Focuses on question-relevant details
- Reduces noise, improves QA performance

### ✨ Automated Data Extraction
- `extract_subtitles_and_questions.py` handles:
  - Subtitle→frame mapping
  - Question extraction for downloaded videos
  - Statistics generation

---

## File Summary

| File | Purpose |
|------|---------|
| `caption_frames_os.py` | Standard frame captioning |
| `caption_frames_query_aware.py` | **NEW** Query-aware captioning |
| `extract_subtitles_and_questions.py` | **NEW** Subtitle & question extraction |
| `embed_frame_captions.py` | Embed captions for search |
| `search_frame_captions.py` | Semantic caption search |
| `os_model.py` | Main QA pipeline |
| `critic_model_os.py` | Answer verification |
| `prompts.py` | Prompt templates |
| `model_example_query.py` | API interface |

See `CODEBASE_DOCUMENTATION.md` for complete details.

---

## Configuration

Create `env.json`:
```json
{
  "together_key": "your_together_api_key",
  "openai_key": "your_openai_api_key",
  "gemini_key": "your_gemini_api_key"
}
```

---

## Performance Comparison

| Method | Pros | Cons |
|--------|------|------|
| Standard | General-purpose, reusable | May miss question-specific details |
| Query-Aware | Focused, less noise | Less general, needs questions first |
| +Subtitles | Captures speech/dialogue | Only helps when subtitles exist |
| **Query-Aware + Subtitles** | **Best QA performance** | **Requires questions + subtitles** |

---

## Tips

1. **Always run extraction first**: `extract_subtitles_and_questions.py`
2. **Use query-aware for benchmarks**: Better performance on test sets
3. **Check subtitle coverage**: Not all videos have subtitles
4. **Monitor confidence scores**: Critic scores below 70% may need review
5. **Keep caption files separate**: Don't mix standard and query-aware

---

## Questions?

See `CODEBASE_DOCUMENTATION.md` for:
- Detailed architecture
- Function descriptions
- Data flow diagrams
- Troubleshooting guide
- Extension examples
