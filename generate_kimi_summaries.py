#!/usr/bin/env python3
"""Generate CES logs and global summaries from clip- or frame-style captions."""

import argparse
import asyncio
import json
from pathlib import Path

from together import AsyncTogether

# Import prompt templates
from prompts import CES_log_prompt, global_summary_prompt


def load_env():
    """Load API keys from env.json."""
    env_path = Path(__file__).parent / "env.json"
    with open(env_path, "r") as f:
        return json.load(f)


async def generate_summary_with_kimi(client, prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    Error generating summary: {e}")
        return None


def _caption_file(video_files_dir: Path, caption_style: str) -> Path:
    if caption_style == "clip":
        return video_files_dir / "captions" / "clip_captions.json"
    if caption_style == "frame":
        return video_files_dir / "clip_frame_captions.json"
    raise ValueError(f"Unsupported caption_style: {caption_style}")


def _format_caption_lines(captions, caption_style: str):
    lines = []
    for clip in captions:
        start = int(clip["start"])
        end = int(clip["end"])
        if caption_style == "frame":
            lines.append(f"[{start}s-{end}s] {clip['caption']}")
        else:
            lines.append(f"[{start}s-{end}s]: {clip['caption']}")
    return "\n".join(lines)


async def generate_summaries_for_video(video_id, dataset_dir, client, model, caption_style="clip"):
    """Generate CES_logs and global_summary for a single video."""
    video_files_dir = dataset_dir / "video_files" / video_id
    captions_file = _caption_file(video_files_dir, caption_style)

    if not captions_file.exists():
        print(f"  ✗ {video_id}: No {caption_style} captions file found")
        return {"video_id": video_id, "status": "error", "error": "No captions file"}

    captions_dir = video_files_dir / "captions"
    ces_log_file = captions_dir / "CES_logs.txt"
    global_summary_file = captions_dir / "global_summary.txt"

    if ces_log_file.exists() and global_summary_file.exists():
        print(f"  ✓ {video_id}: Summaries already exist")
        return {"video_id": video_id, "status": "skipped"}

    with open(captions_file, "r") as f:
        captions_data_json = json.load(f)

    if not captions_data_json:
        print(f"  ✗ {video_id}: No captions in file")
        return {"video_id": video_id, "status": "error", "error": "Empty captions file"}

    print(f"  Processing {video_id}: {len(captions_data_json)} captions")

    captions_text = _format_caption_lines(captions_data_json, caption_style)

    print("    Generating CES logs...")
    ces_prompt = CES_log_prompt(captions_text)
    ces_log = await generate_summary_with_kimi(client, ces_prompt, model)

    if ces_log:
        ces_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ces_log_file, "w") as f:
            f.write(ces_log)
        print(f"    ✓ CES_logs.txt generated")
    else:
        print(f"    ✗ Failed to generate CES logs")

    print("    Generating global summary...")
    summary_prompt = global_summary_prompt(captions_text)
    global_summary = await generate_summary_with_kimi(client, summary_prompt, model)

    if global_summary:
        with open(global_summary_file, "w") as f:
            f.write(global_summary)
        print(f"    ✓ global_summary.txt generated")
    else:
        print(f"    ✗ Failed to generate global summary")

    if ces_log and global_summary:
        print(f"  ✓ {video_id}: Summaries generated")
        return {"video_id": video_id, "status": "success"}
    return {"video_id": video_id, "status": "partial", "error": "Some summaries failed"}


async def process_dataset(dataset_name, base_dir, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", caption_style="clip"):
    dataset_dir = Path(base_dir) / "kimi" / dataset_name
    video_files_dir = dataset_dir / "video_files"

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    video_ids = [d.name for d in video_files_dir.iterdir() if d.is_dir()]
    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'=' * 60}")

    env = load_env()
    client = AsyncTogether(api_key=env["together_key"])

    results = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}]")
        result = await generate_summaries_for_video(video_id, dataset_dir, client, model, caption_style=caption_style)
        results.append(result)
        await asyncio.sleep(0.5)

    successful = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = len(results) - successful - skipped

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Generate summaries from captions")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                        help='Dataset to process')
    parser.add_argument('--base-dir', type=str, default='/mnt/ssd/data',
                        help='Base directory')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                        help='Model endpoint for summary generation')
    parser.add_argument(
        '--caption-style',
        choices=['clip', 'frame'],
        default='clip',
        help='Caption layout to summarize (clip_captions.json or clip_frame_captions.json)',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Kimi Summary Generation")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model}")
    print(f"Caption style: {args.caption_style}")

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        asyncio.run(process_dataset(dataset, args.base_dir, args.model, caption_style=args.caption_style))

    print("\n" + "=" * 60)
    print("SUMMARY GENERATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Review generated CES_logs.txt and global_summary.txt")
    print("=" * 60)


if __name__ == '__main__':
    main()
