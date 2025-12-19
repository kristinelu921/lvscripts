# VideoMME Dataset Integration

This guide explains how to use the `run_videomme.py` script to evaluate the long-context video understanding pipeline on the [VideoMME dataset](https://huggingface.co/datasets/lmms-lab/Video-MME).

## Overview

The `run_videomme.py` script adapts the existing three-stage pipeline (OS Model → Critic → Critic Response) to work with VideoMME's format:
- Automatically downloads videos from YouTube
- Extracts frames at 1 FPS
- Generates captions and embeddings
- Runs the full pipeline on multiple-choice questions
- Reports accuracy with and without the critic module

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies

**FFmpeg** (required for frame extraction):
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

**yt-dlp** (for downloading YouTube videos):
```bash
pip install yt-dlp
```

### 3. Setup API Keys

Create `env.json` in the scripts directory:
```json
{
  "openai_key": "sk-...",
  "together_key": "..."
}
```

## Usage

### Basic Usage

Run on first 10 videos from validation set:
```bash
python run_videomme.py --subset validation --max_videos 10 --output_dir ./videomme_results
```

### Quick Test

Process just 1 video with 3 questions:
```bash
python run_videomme.py --max_videos 1 --max_questions_per_video 3 --output_dir ./test_run
```

### Without Critic Module

Test OS model only (faster):
```bash
python run_videomme.py --no_critic --max_videos 5 --output_dir ./no_critic_test
```

### Full Evaluation

Run on entire validation set:
```bash
python run_videomme.py --subset validation --output_dir ./full_evaluation
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--subset` | `validation` | Dataset subset to use |
| `--output_dir` | `./videomme_results` | Directory to store results |
| `--llm_model` | `deepseek-ai/DeepSeek-V3.1` | LLM model for reasoning |
| `--vlm_model` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | VLM model for vision |
| `--max_videos` | `None` (all) | Maximum number of videos to process |
| `--max_questions_per_video` | `None` (all) | Maximum questions per video |
| `--no_critic` | `False` | Disable critic module |
| `--reprocess` | `False` | Reprocess videos even if already processed |

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── videomme_results.json         # Detailed results for each question
├── videomme_summary.json          # Overall statistics and accuracy
└── <video_id>/                    # One directory per video
    ├── <video_id>.mp4             # Downloaded video
    ├── <video_id>.json            # Video metadata
    ├── <video_id>_answers.json    # OS model answers
    ├── <video_id>_critic_assessment.json  # Critic assessments
    ├── <video_id>_re_evaluated.json       # Re-evaluated answers
    ├── frames/
    │   ├── frame_0001.jpg         # Extracted frames (1 FPS)
    │   └── ...
    └── captions/
        ├── frame_captions_sorted.json
        ├── frame_captions_sorted_embeddings.jsonl
        ├── CES_logs.txt
        └── global_summary.txt
```

### Result Format

**videomme_results.json** contains a list of results:
```json
[
  {
    "video_id": "video_001",
    "question_id": "q1",
    "question": "What color is the car?",
    "options": ["Red", "Blue", "Green", "Yellow"],
    "ground_truth": "A",
    "os_prediction": "A",
    "os_reasoning": "The car appears red in frames 12-45...",
    "os_frames": [12, 23, 34, 45],
    "correct": true,
    "critic_confidence": 85,
    "critic_feedback": "Evidence frames clearly show...",
    "final_prediction": "A",
    "final_correct": true
  }
]
```

**videomme_summary.json** contains overall statistics:
```json
{
  "dataset_subset": "validation",
  "total_questions": 100,
  "os_correct": 60,
  "os_accuracy": 60.0,
  "final_correct": 65,
  "final_accuracy": 65.0,
  "llm_model": "deepseek-ai/DeepSeek-V3.1",
  "vlm_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
  "use_critic": true
}
```

## How It Works

### Pipeline Flow

```
1. Load VideoMME dataset from HuggingFace
   ↓
2. For each video:
   a. Download from YouTube
   b. Extract frames at 1 FPS
   c. Generate frame captions using VLM
   d. Generate CES logs (Characters/Events/Scenes)
   e. Generate global summary
   f. Embed captions for semantic search
   ↓
3. For each question:
   a. Format as multiple choice
   b. Run OS Model (reasoning agent with tools)
   c. Run Critic (assess answer quality)
   d. Run Critic Response (re-evaluate if needed)
   e. Compare with ground truth
   ↓
4. Calculate accuracy and save results
```

### Question Format Adaptation

VideoMME questions are automatically formatted for the pipeline:

**Original VideoMME format:**
```python
{
  "question": "What color is the car?",
  "options": ["Red", "Blue", "Green", "Yellow"],
  "answer": "A"
}
```

**Formatted for pipeline:**
```
What color is the car?

Options:
A. Red
B. Blue
C. Green
D. Yellow

Answer with a single letter (A, B, C, or D).
```

## Performance Notes

### Processing Time

Approximate times per video (depends on video length):
- **Video download**: 30s - 5min
- **Frame extraction**: 10s - 2min
- **Caption generation**: 2-10min (depends on frame count)
- **CES logs + summary**: 1-3min
- **Embedding**: 30s - 2min
- **Per question**: 1-3min (OS model + critic)

**Total per video**: ~15-30 minutes for setup + 2-5 minutes per question

### Caching

The script intelligently caches all preprocessing:
- Videos are downloaded once
- Frames are extracted once
- Captions/embeddings are generated once
- Subsequent runs reuse cached data

Use `--reprocess` flag to force regeneration.

### Resource Requirements

- **Disk space**: ~500MB - 2GB per video (video + frames + captions)
- **Memory**: 4-8GB recommended
- **GPU**: Optional but recommended for faster VLM inference
- **API costs**: Depends on Together AI and OpenAI usage

## Troubleshooting

### Common Issues

**1. yt-dlp not found**
```bash
pip install yt-dlp
```

**2. ffmpeg not found**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

**3. Video download fails**
- Check internet connection
- Some videos may be region-restricted
- Try updating yt-dlp: `pip install -U yt-dlp`

**4. API errors**
- Verify API keys in `env.json`
- Check API rate limits
- Ensure sufficient API credits

**5. Out of memory**
- Reduce batch size in model queries
- Process fewer videos at once using `--max_videos`
- Close other applications

### Resuming Failed Runs

The script automatically skips processed videos. If a run fails:
```bash
# Simply re-run the same command
python run_videomme.py --output_dir ./videomme_results --max_videos 10

# It will skip already completed videos and continue from where it stopped
```

To force reprocessing:
```bash
python run_videomme.py --output_dir ./videomme_results --reprocess
```

## Expected Results

Based on the research paper, expected accuracy on LV-Bench:
- **OS Model only**: ~60%
- **OS Model + Critic**: ~65%

VideoMME is a different benchmark, so results may vary.

## Example Run

```bash
$ python run_videomme.py --max_videos 2 --output_dir ./test

Loading VideoMME dataset (subset: validation)...
Dataset loaded: 900 examples
Found 150 unique videos
Processing first 2 videos

================================================================================
Video 1/2: video_001
================================================================================
Processing 6 questions for this video
Downloading video video_001...
Extracting frames from video_001.mp4 at 1 FPS...
Extracted 180 frames
Generating captions for video_001...
Generating CES logs for video_001...
Generating global summary for video_001...
Embedding captions for video_001...

Processing question q1 for video video_001...
OS Model: Answer A (confidence: high)
Critic confidence: 85% - Evidence frames match reasoning well
Final answer: A ✓ Correct!

...

================================================================================
FINAL RESULTS
================================================================================
Total questions processed: 12
OS Model Accuracy: 7/12 (58.33%)
Final Accuracy (with Critic): 8/12 (66.67%)
Improvement: +8.33%

Results saved to:
  - ./test/videomme_results.json
  - ./test/videomme_summary.json
```

## Integration with Existing Pipeline

The VideoMME script reuses all existing pipeline components:
- `caption_frames_os.py` - Frame captioning
- `embed_frame_captions.py` - Embedding generation
- `os_model.py` - Original Solution model
- `critic_model_os.py` - Critic assessment
- `critic_response.py` - Critic response
- `search_frame_captions.py` - Semantic search

No modifications to core pipeline code are required.

## Citation

If you use this VideoMME integration, please cite:

```bibtex
@inproceedings{videomme2024,
  title={Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis},
  author={VideoMME Team},
  year={2024}
}
```

And the original pipeline paper (if applicable).

## License

Same license as the parent project. VideoMME dataset has its own license terms.
