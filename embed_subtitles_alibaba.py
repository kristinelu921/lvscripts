#!/usr/bin/env python3
"""
Generate subtitle embeddings using nvidia/NV-Embed-v2 (1024 dims).
Processes subtitles in subtitles_val directories.

Usage:
    python embed_subtitles_alibaba.py --dataset longvideobench
    python embed_subtitles_alibaba.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from together import Together
from tqdm import tqdm


def load_env():
    """Load API keys from env.json."""
    env_path = Path(__file__).parent / 'env.json'
    with open(env_path, 'r') as f:
        return json.load(f)


def parse_timestamp_to_seconds(timestamp):
    """Convert timestamp HH:MM:SS.mmm to seconds."""
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def embed_texts_batch(client, texts, model="nvidia/NV-Embed-v2", batch_size=32):
    """
    Embed texts using Together AI in batches.

    Args:
        client: Together client
        texts: List of text strings
        model: Embedding model
        batch_size: Batch size for API calls

    Returns:
        List of embedding vectors
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]

        try:
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"    Error embedding batch {i//batch_size}: {e}")
            # Retry with smaller batch if failed
            if len(batch) > 1:
                print(f"    Retrying with smaller batches...")
                for text in batch:
                    try:
                        response = client.embeddings.create(model=model, input=[text])
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as e2:
                        print(f"    Error embedding single text: {e2}")
                        all_embeddings.append(None)
            else:
                all_embeddings.append(None)

    return all_embeddings


def embed_subtitle_file(video_id, subtitles_dir, client, model="nvidia/NV-Embed-v2", force=False):
    """Embed subtitles for a single video."""

    # Check if subtitles exist (try both formats)
    subtitle_file_jsonl = subtitles_dir / f'{video_id}_en.jsonl'
    subtitle_file_json = subtitles_dir / f'{video_id}_en.json'

    if subtitle_file_jsonl.exists():
        subtitle_file = subtitle_file_jsonl
    elif subtitle_file_json.exists():
        subtitle_file = subtitle_file_json
    else:
        return {'video_id': video_id, 'status': 'error', 'error': 'No subtitle file'}

    # Check if embeddings already exist
    embeddings_file = subtitles_dir / f'{video_id}_en_embeddings_alibaba.jsonl'
    if embeddings_file.exists() and not force:
        with open(embeddings_file, 'r') as f:
            num_embeddings = sum(1 for line in f)
        print(f"  ✓ {video_id}: Embeddings already exist ({num_embeddings} captions)")
        return {'video_id': video_id, 'status': 'skipped', 'embeddings_generated': num_embeddings}

    # Load subtitles
    subtitles = []
    with open(subtitle_file, 'r') as f:
        if subtitle_file.suffix == '.json':
            # Load as JSON array
            subtitles = json.load(f)
        else:
            # Load as JSONL
            for line in f:
                subtitles.append(json.loads(line))

    if not subtitles:
        print(f"  ✗ {video_id}: No subtitles in file")
        return {'video_id': video_id, 'status': 'error', 'error': 'Empty subtitles file'}

    print(f"  Processing {video_id}: {len(subtitles)} subtitle captions")

    # Extract caption texts (handle both 'text' and 'line' fields)
    texts = [sub.get('text') or sub.get('line') for sub in subtitles]

    # Generate embeddings in batches
    print(f"    Generating embeddings...")
    embeddings = embed_texts_batch(client, texts, model=model, batch_size=32)

    # Create JSONL records
    records = []
    for i, (sub, embedding) in enumerate(zip(subtitles, embeddings)):
        if embedding is None:
            print(f"    Warning: No embedding for subtitle {i}")
            continue

        text = sub.get('text') or sub.get('line')
        start = sub.get('start')
        end = sub.get('end')

        # Parse timestamps to seconds
        start_sec = parse_timestamp_to_seconds(start) if start else 0.0
        end_sec = parse_timestamp_to_seconds(end) if end else 0.0

        records.append({
            'id': sub.get('id', i),
            'text': text,
            'embedding': embedding,
            'start': start,
            'end': end,
            'start_sec': start_sec,
            'end_sec': end_sec
        })

    # Save embeddings as JSONL
    with open(embeddings_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"  ✓ {video_id}: Generated {len(records)} embeddings")

    return {
        'video_id': video_id,
        'status': 'success',
        'embeddings_generated': len(records)
    }


def process_dataset(dataset_name, base_dir, model="nvidia/NV-Embed-v2", force=False):
    """Process all videos in a dataset."""
    subtitles_dir = Path(base_dir) / dataset_name / 'subtitles_val'

    if not subtitles_dir.exists():
        print(f"Error: {subtitles_dir} does not exist")
        return

    # Get list of subtitle files (try both .jsonl and .json)
    subtitle_files = list(subtitles_dir.glob('*_en.jsonl')) + list(subtitles_dir.glob('*_en.json'))
    video_ids = [f.stem.replace('_en', '') for f in subtitle_files]

    if not video_ids:
        print(f"No subtitle files found in {subtitles_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    # Load API key
    env = load_env()
    client = Together(api_key=env['together_key'])

    results = []
    for i, video_id in enumerate(sorted(video_ids), 1):
        print(f"[{i}/{len(video_ids)}]")
        result = embed_subtitle_file(video_id, subtitles_dir, client, model, force=force)
        results.append(result)

    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{dataset_name} Summary:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description='Generate subtitle embeddings with Alibaba model')
    parser.add_argument('--dataset', choices=['longvideobench', 'lvbench', 'videomme'],
                       help='Dataset to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets')
    parser.add_argument('--base-dir', default='/mnt/ssd/data',
                       help='Base directory (default: /mnt/ssd/data)')
    parser.add_argument('--model', default='nvidia/NV-Embed-v2',
                       help='Embedding model')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing subtitle embeddings')

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error('Must specify either --dataset or --all')

    print("=" * 60)
    print("Subtitle Embedding Generation (Alibaba Model)")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model}")
    print()

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.all else [args.dataset]

    for dataset in datasets:
        process_dataset(dataset, args.base_dir, args.model, force=args.force)

    print("\n" + "=" * 60)
    print("SUBTITLE EMBEDDING GENERATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
