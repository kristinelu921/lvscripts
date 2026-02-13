#!/usr/bin/env python3
"""
Generate embeddings for clip_frame_captions.json using Together AI.
Reads from video_files/{video_id}/clip_frame_captions.json

Usage:
    python embed_clip_frame_captions.py --dataset longvideobench
    python embed_clip_frame_captions.py --dataset all
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from together import Together
from tqdm import tqdm


def load_env():
    """Load API keys from env.json."""
    env_path = Path(__file__).parent / 'env.json'
    with open(env_path, 'r') as f:
        return json.load(f)


def embed_texts_batch(client, texts, model="Alibaba-NLP/gte-modernbert-base", batch_size=32):
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


def embed_video_frame_captions(video_id, dataset_dir, client, model="Alibaba-NLP/gte-modernbert-base"):
    """Embed clip_frame_captions for a single video."""
    video_files_dir = dataset_dir / 'video_files' / video_id

    # Check if clip_frame_captions exist
    frame_captions_file = video_files_dir / 'clip_frame_captions.json'
    if not frame_captions_file.exists():
        print(f"  ✗ {video_id}: No clip_frame_captions.json found")
        return {'video_id': video_id, 'status': 'error', 'error': 'No frame captions file'}

    # Check if embeddings already exist
    embeddings_file = video_files_dir / 'clip_frame_embeddings.jsonl'
    if embeddings_file.exists():
        with open(embeddings_file, 'r') as f:
            num_embeddings = sum(1 for line in f)
        print(f"  ✓ {video_id}: Embeddings already exist ({num_embeddings} clips)")
        return {'video_id': video_id, 'status': 'skipped', 'embeddings_generated': num_embeddings}

    # Load frame captions
    with open(frame_captions_file, 'r') as f:
        frame_captions = json.load(f)

    if not frame_captions:
        print(f"  ✗ {video_id}: No captions in file")
        return {'video_id': video_id, 'status': 'error', 'error': 'Empty captions file'}

    print(f"  Processing {video_id}: {len(frame_captions)} frame clips")

    # Extract caption texts
    texts = [clip['caption'] for clip in frame_captions]

    # Generate embeddings in batches
    print(f"    Generating embeddings...")
    embeddings = embed_texts_batch(client, texts, model=model, batch_size=32)

    # Create JSONL records
    records = []
    for i, (clip, embedding) in enumerate(zip(frame_captions, embeddings)):
        if embedding is None:
            print(f"    Warning: No embedding for clip {i}")
            continue

        # Create clip ID from start and end times
        start = int(clip['start'])
        end = int(clip['end'])
        clip_id = f"clip_{start}_{end}"

        records.append({
            'id': clip_id,
            'text': clip['caption'],
            'embedding': embedding,
            'start': clip['start'],
            'end': clip['end']
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


def process_dataset(dataset_name, base_dir, model="Alibaba-NLP/gte-modernbert-base"):
    """Process all videos in a dataset."""
    dataset_dir = Path(base_dir) / 'kimi' / dataset_name
    video_files_dir = dataset_dir / 'video_files'

    if not video_files_dir.exists():
        print(f"Error: {video_files_dir} does not exist")
        return

    # Get list of video IDs
    video_ids = sorted([d.name for d in video_files_dir.iterdir() if d.is_dir()])

    if not video_ids:
        print(f"No video directories found in {video_files_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}: {len(video_ids)} videos")
    print(f"{'='*60}")

    # Load API key
    env = load_env()
    client = Together(api_key=env['together_key'])

    results = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}]")
        result = embed_video_frame_captions(video_id, dataset_dir, client, model)
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
    parser = argparse.ArgumentParser(description='Generate embeddings for clip_frame_captions')
    parser.add_argument('--dataset', choices=['all', 'longvideobench', 'lvbench', 'videomme'],
                       default='all', help='Dataset to process')
    parser.add_argument('--base-dir', default='/mnt/ssd/data',
                       help='Base directory (default: /mnt/ssd/data)')
    parser.add_argument('--model', default='Alibaba-NLP/gte-modernbert-base',
                       help='Embedding model')

    args = parser.parse_args()

    print("=" * 60)
    print("Clip Frame Caption Embedding Generation")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model}")
    print()

    datasets = ['longvideobench', 'lvbench', 'videomme'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        process_dataset(dataset, args.base_dir, args.model)

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Generate summaries using clip_frame_captions")
    print("  2. Run pipeline with --use-frame-captions flag")


if __name__ == '__main__':
    main()
