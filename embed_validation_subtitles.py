#!/usr/bin/env python3
"""
Embed validation subtitle JSON files using Together AI API with BAAI/bge-large-en-v1.5
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
from together import Together


def load_together_client():
    """Load Together API client from env.json"""
    env_path = Path("/mnt/ssd/data/lvscripts/env.json")
    with open(env_path, "r") as f:
        env = json.load(f)
        together_key = env["together_key"]
    os.environ["TOGETHER_API_KEY"] = together_key
    return Together(api_key=together_key)


def parse_timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds"""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    ms = int(s_parts[1]) if len(s_parts) > 1 else 0
    return h * 3600 + m * 60 + s + ms / 1000.0


def load_subtitle_file(path: str):
    """Load subtitle JSON and parse timestamps"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for i, entry in enumerate(data):
        start_sec = parse_timestamp_to_seconds(entry["start"])
        end_sec = parse_timestamp_to_seconds(entry["end"])
        text = entry["line"]

        records.append({
            "id": i,
            "start": entry["start"],
            "end": entry["end"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text
        })

    return records


def truncate_text(text: str, max_chars: int = 3500) -> str:
    """Truncate text to fit within token limit"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch_together(client, texts: List[str], model: str = "BAAI/bge-large-en-v1.5", max_retries: int = 3):
    """Embed batch using Together API with retries"""
    max_chars = 3500
    texts = [truncate_text(t, max_chars) for t in texts]

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            embeddings = [item.embedding for item in response.data]

            # L2 normalize
            embeddings = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

            return embeddings.tolist()
        except Exception as e:
            if "maximum context length" in str(e):
                max_chars = max_chars // 2
                texts = [truncate_text(t, max_chars) for t in texts]
                print(f"\nTruncating to {max_chars} chars due to token limit")
                continue

            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\nRetry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def write_embeddings_jsonl(output_path: str, records: List[dict], embeddings: List[list]):
    """Write records with embeddings to JSONL format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec, emb in zip(records, embeddings):
            rec_with_emb = dict(rec)
            rec_with_emb["embedding"] = emb
            f.write(json.dumps(rec_with_emb, ensure_ascii=False) + "\n")


def embed_subtitle_file(client, subtitle_path: str, output_path: str, model: str, batch_size: int = 100):
    """Embed a single subtitle file"""
    # Load subtitle records
    records = load_subtitle_file(subtitle_path)
    texts = [r["text"] for r in records]

    if not texts:
        print(f"  No subtitles found in {subtitle_path}")
        return 0

    # Embed in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch_together(client, batch_texts, model)
        all_embeddings.extend(batch_embeddings)

        # Small delay to avoid rate limits
        if i + batch_size < len(texts):
            time.sleep(0.1)

    # Write to JSONL
    write_embeddings_jsonl(output_path, records, all_embeddings)

    return len(all_embeddings)


def main():
    """Embed all validation subtitle files"""
    subtitles_folder = "/mnt/ssd/data/longvideobench/subtitles"
    val_videos_folder = "/mnt/ssd/data/longvideobench/videos_processed_val"
    model = "BAAI/bge-large-en-v1.5"
    batch_size = 100

    print("=" * 70)
    print("VALIDATION SUBTITLE EMBEDDING - Together AI with BAAI/bge-large-en-v1.5")
    print(f"Model: {model} (1024 dimensions)")
    print(f"Subtitles folder: {subtitles_folder}")
    print(f"Validation videos: {val_videos_folder}")
    print("=" * 70)

    # Initialize client
    print("\nInitializing Together AI client...")
    client = load_together_client()
    print("✓ Client ready")

    # Get list of validation videos
    val_videos = sorted(os.listdir(val_videos_folder))
    print(f"\nFound {len(val_videos)} validation videos")

    # Find subtitle files that need embedding
    to_embed = []
    for video_id in val_videos:
        subtitle_file = f"{video_id}_en.json"
        subtitle_path = os.path.join(subtitles_folder, subtitle_file)
        embeddings_file = f"{video_id}_en_embeddings.jsonl"
        embeddings_path = os.path.join(subtitles_folder, embeddings_file)

        # Check if subtitle exists
        if not os.path.exists(subtitle_path):
            continue

        # Check if already embedded
        if os.path.exists(embeddings_path):
            continue

        to_embed.append((video_id, subtitle_path, embeddings_path))

    print(f"Subtitles to embed: {len(to_embed)}")
    print(f"Already embedded: {len(val_videos) - len(to_embed)}")

    if len(to_embed) == 0:
        print("\n✓ All validation subtitles already embedded!")
        return 0

    # Process each file
    total_embedded = 0
    failed = []

    for video_id, subtitle_path, embeddings_path in tqdm(to_embed, desc="Embedding"):
        try:
            count = embed_subtitle_file(client, subtitle_path, embeddings_path, model, batch_size)
            total_embedded += count
        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            failed.append((video_id, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Videos processed: {len(to_embed) - len(failed)}/{len(to_embed)}")
    print(f"Total subtitle entries embedded: {total_embedded:,}")
    if failed:
        print(f"Failed: {len(failed)} files")
        for video_id, reason in failed[:5]:
            print(f"  - {video_id}: {reason}")
    else:
        print("✓ All files processed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
