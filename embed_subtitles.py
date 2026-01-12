#!/usr/bin/env python3
"""
Embed all subtitle JSON files using Together AI API with BAAI/bge-large-en-v1.5 (1024 dims)

Input: Subtitle JSON files in /mnt/ssd/data/longvideobench/subtitles/*_en.json
Output: Creates subtitle_embeddings.jsonl for each subtitle file

Each subtitle entry has: {"start": "HH:MM:SS.mmm", "end": "HH:MM:SS.mmm", "line": "text"}
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
    env_path = Path(__file__).parent / "env.json"
    with open(env_path, "r") as f:
        env = json.load(f)
        together_key = env["together_key"]
    os.environ["TOGETHER_API_KEY"] = together_key
    return Together(api_key=together_key)


def parse_timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds

    Args:
        ts: Timestamp string like "00:00:02.790"

    Returns:
        Float seconds like 2.79
    """
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    ms = int(s_parts[1]) if len(s_parts) > 1 else 0

    return h * 3600 + m * 60 + s + ms / 1000.0


def load_subtitle_file(path: str):
    """Load subtitle JSON and parse timestamps

    Returns:
        List of dicts with 'start_sec', 'end_sec', 'text'
    """
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
    """Embed all subtitle files in the subtitles folder"""
    subtitles_folder = "/mnt/ssd/data/longvideobench/subtitles_filtered"
    model = "BAAI/bge-large-en-v1.5"
    batch_size = 100

    print("=" * 70)
    print("SUBTITLE EMBEDDING - Together AI with BAAI/bge-large-en-v1.5")
    print(f"Model: {model} (1024 dimensions)")
    print(f"Folder: {subtitles_folder}")
    print("=" * 70)

    # Initialize client
    print("\nInitializing Together AI client...")
    client = load_together_client()
    print("✓ Client ready")

    # Find all subtitle files
    subtitle_files = sorted([f for f in os.listdir(subtitles_folder) if f.endswith("_en.json")])
    print(f"\nFound {len(subtitle_files)} subtitle files")

    # Process each file
    total_embedded = 0
    skipped = []

    for filename in tqdm(subtitle_files, desc="Processing"):
        subtitle_path = os.path.join(subtitles_folder, filename)
        output_filename = filename.replace(".json", "_embeddings.jsonl")
        output_path = os.path.join(subtitles_folder, output_filename)

        # Skip if already processed
        if os.path.exists(output_path):
            skipped.append((filename, "already done"))
            continue

        try:
            count = embed_subtitle_file(client, subtitle_path, output_path, model, batch_size)
            total_embedded += count
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped.append((filename, f"error: {e}"))

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Files processed: {len(subtitle_files) - len(skipped)}/{len(subtitle_files)}")
    print(f"Total subtitle entries embedded: {total_embedded:,}")
    if skipped:
        print(f"Skipped: {len(skipped)} files")
        for fname, reason in skipped[:5]:
            print(f"  - {fname}: {reason}")
    else:
        print("✓ All files processed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
