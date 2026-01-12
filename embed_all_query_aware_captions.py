#!/usr/bin/env python3
"""
Script to embed all frame_captions_query_aware.json files using Together AI API
Uses BAAI/bge-large-en-v1.5 (1024 dimensions) - much faster with API batching

Output: Creates frame_captions_query_aware_embeddings.jsonl for each video
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from tqdm import tqdm
from together import Together

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from embed_frame_captions import iter_records, write_jsonl


def load_together_client():
    """Load Together API client"""
    with open("env.json", "r") as f:
        env = json.load(f)
        together_key = env["together_key"]
    os.environ["TOGETHER_API_KEY"] = together_key
    return Together(api_key=together_key)


def collect_all_captions(vid_folder, caption_filename='frame_captions_sorted.json'):
    """Collect all captions from all videos"""
    video_data = []
    skipped = []

    video_dirs = sorted(os.listdir(vid_folder))
    print(f"Scanning {len(video_dirs)} video directories...")

    for video_id in tqdm(video_dirs, desc="Collecting"):
        input_path = os.path.join(vid_folder, video_id, 'captions', caption_filename)
        output_filename = caption_filename.replace('.json', '_embeddings.jsonl')
        output_path = os.path.join(vid_folder, video_id, 'captions', output_filename)

        if not os.path.exists(input_path):
            print("input path", input_path)
            print("video id file not found")
            skipped.append((video_id, "no input"))
            continue

        if os.path.exists(output_path):
            print("fiile already exists")
            skipped.append((video_id, "already done"))
            continue

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            records = list(iter_records(data))
            texts = [str(r["text"]) for r in records]

            if not texts:
                skipped.append((video_id, "no captions"))
                print("texts not found")
                continue
            video_data.append({
                'video_id': video_id,
                'output_path': output_path,
                'records': records,
                'texts': texts,
                'num_captions': len(texts)
            })
        except Exception as e:
            skipped.append((video_id, f"error: {e}"))

    print(f"\n✓ Found {len(video_data)} videos to process")
    if skipped:
        print(f"Skipped {len(skipped)} videos (first 5: {[s[0] for s in skipped[:5]]})")

    return video_data


def truncate_text(text, max_chars=2000):
    """Truncate text to roughly fit within 512 token limit

    512 tokens ≈ 2000 characters (conservative estimate)
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch_together(client, texts, model="BAAI/bge-large-en-v1.5", max_retries=3):
    """Embed batch using Together API with retries and truncation"""
    # Truncate all texts to fit within token limit
    max_chars = 3500

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            # Check if it's a token limit error
            if "maximum context length" in str(e):
                # Further truncate and retry
                max_chars=max_chars//2
                texts = [truncate_text(t, max_chars=max_chars) for t in texts]
                print(f"\nTruncating texts further to {max_chars} characters due to token limit")
                continue

            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\nRetry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def embed_all_captions(client, all_texts, model="BAAI/bge-large-en-v1.5", batch_size=100):
    """Embed all texts using Together API"""
    print(f"\nEmbedding {len(all_texts):,} captions with {model}")
    print(f"Batch size: {batch_size}")

    all_embeddings = []

    with tqdm(total=len(all_texts), desc="Embedding") as pbar:
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embeddings = embed_batch_together(client, batch, model)
            all_embeddings.extend(embeddings)
            pbar.update(len(batch))

            # Small delay to avoid rate limits
            if i + batch_size < len(all_texts):
                time.sleep(0.1)

    return np.array(all_embeddings, dtype=np.float32)


def write_all_embeddings(video_data, all_embeddings):
    """Write embeddings back to individual video files"""
    print("\nWriting embeddings to files...")

    current_idx = 0
    failed = []

    for video_info in tqdm(video_data, desc="Writing"):
        video_id = video_info['video_id']
        num = video_info['num_captions']

        try:
            video_embs = all_embeddings[current_idx:current_idx + num]
            current_idx += num

            write_jsonl(video_info['output_path'], video_info['records'], video_embs)
        except Exception as e:
            print(f"\nError writing {video_id}: {e}")
            failed.append(video_id)

    return failed


def main():
    """Main entry point"""
    videos_folder = '/mnt/ssd/data/lvbench/videos_processed'
    caption_filename = 'frame_captions_sorted.json'
    model = 'BAAI/bge-large-en-v1.5'
    batch_size = 100

    print("="*70)
    print("TOGETHER AI EMBEDDING - OPTIMIZED BATCH PROCESSING")
    print(f"Model: {model} (1024 dimensions)")
    print(f"Folder: {videos_folder}")
    print(f"Caption file: {caption_filename}")
    print("="*70)

    # Initialize API client
    print("\nInitializing Together AI client...")
    client = load_together_client()
    print("✓ Client ready")

    # Collect all captions
    video_data = collect_all_captions(videos_folder, caption_filename)

    if not video_data:
        print("\nNo videos to process!")
        return 0

    # Flatten all texts
    all_texts = []
    for v in video_data:
        all_texts.extend(v['texts'])

    total = len(all_texts)
    print(f"\nTotal captions to embed: {total:,}")

    # Embed everything
    start = time.time()
    all_embeddings = embed_all_captions(client, all_texts, model, batch_size)
    elapsed = time.time() - start

    print(f"\n✓ Generated {all_embeddings.shape[0]:,} embeddings (dim={all_embeddings.shape[1]})")
    print(f"✓ Speed: {total/elapsed:.1f} captions/sec ({elapsed:.1f}s total)")

    # Write results
    failed = write_all_embeddings(video_data, all_embeddings)

    # Summary
    print("\n" + "="*70)
    print("COMPLETE")
    print(f"Videos: {len(video_data)}")
    print(f"Captions: {total:,}")
    print(f"Speed: {total/elapsed:.1f} captions/sec")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("✓ All successful!")
    print("="*70)

    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
