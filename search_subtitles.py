#!/usr/bin/env python3
"""
Search subtitle embeddings using semantic similarity

Uses Together AI with BAAI/bge-large-en-v1.5 (1024 dims) to search subtitle content.
Returns matching subtitle segments with exact frame timestamps.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from together import Together


def load_together_client():
    """Load Together API client from env.json"""
    from pathlib import Path
    env_path = Path(__file__).parent / "env.json"
    with open(env_path, "r") as f:
        env = json.load(f)
        together_key = env["together_key"]
    os.environ["TOGETHER_API_KEY"] = together_key
    return Together(api_key=together_key)


def load_jsonl_embeddings(path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load subtitle embeddings from JSONL file

    Returns:
        records: List of subtitle record dicts
        matrix: numpy array of embeddings (n_records, embedding_dim)
    """
    records: List[Dict] = []
    vectors: List[np.ndarray] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "embedding" not in rec:
                continue
            vec = np.asarray(rec["embedding"], dtype=np.float32)
            records.append(rec)
            vectors.append(vec)

    if not vectors:
        raise RuntimeError(f"No embeddings found in {path}")

    matrix = np.vstack(vectors)
    return records, matrix


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def embed_query(query: str, client: Together, model: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """Embed a query string using Together AI

    Returns:
        L2-normalized embedding vector
    """
    response = client.embeddings.create(model=model, input=query)
    vec = np.asarray(response.data[0].embedding, dtype=np.float32)

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def cosine_topk(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find top-k most similar vectors using cosine similarity

    Args:
        query_vec: Query embedding vector
        corpus: Matrix of corpus embeddings
        k: Number of top results to return

    Returns:
        idx: Indices of top-k results
        scores: Cosine similarity scores for top-k results
    """
    q = query_vec.reshape(1, -1)
    qn = l2_normalize(q)
    cn = l2_normalize(corpus)
    scores = (cn @ qn.T).ravel()

    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        idx = part[np.argsort(-scores[part])]

    return idx, scores[idx]


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format

    Args:
        seconds: Time in seconds (float)

    Returns:
        Formatted string like "0:02" or "1:23:45"
    """
    total_sec = int(seconds)
    h, remainder = divmod(total_sec, 3600)
    m, s = divmod(remainder, 60)

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def seconds_to_frame(seconds: float, fps: float = 1.0) -> int:
    """Convert seconds to frame number

    Args:
        seconds: Time in seconds
        fps: Frames per second (default 1.0)

    Returns:
        Frame number (int)
    """
    return int(seconds * fps)


def search_subtitles(embeddings_path: str, query: str, topk: int = 10, fps: float = 1.0) -> List[Dict]:
    """Search subtitles using semantic similarity

    Args:
        embeddings_path: Path to subtitle embeddings JSONL file
        query: Search query text
        topk: Number of top results to return
        fps: Frames per second for frame number conversion (default 1.0)

    Returns:
        List of search results with similarity scores and timestamps
    """
    # Load client
    client = load_together_client()

    # Embed query
    query_vec = embed_query(query, client)

    # Load subtitle embeddings
    records, matrix = load_jsonl_embeddings(embeddings_path)

    # Check dimension compatibility
    if query_vec.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query_vec.shape[0]} dims "
            f"but subtitles have {matrix.shape[1]} dims"
        )

    # Find top-k similar subtitles
    idx, scores = cosine_topk(query_vec, matrix, k=topk)

    # Format results
    results = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        rec = dict(records[int(i)])

        # Calculate frame numbers from timestamps
        start_frame = seconds_to_frame(rec["start_sec"], fps)
        end_frame = seconds_to_frame(rec["end_sec"], fps)

        result = {
            "rank": rank,
            "score": float(score),
            "text": rec["text"],
            "start": rec["start"],
            "end": rec["end"],
            "start_sec": rec["start_sec"],
            "end_sec": rec["end_sec"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "time_formatted": format_time(rec["start_sec"])
        }
        results.append(result)

    return results


def main():
    """CLI interface for subtitle search"""
    parser = argparse.ArgumentParser(
        description="Search subtitles using Together AI with BAAI/bge-large-en-v1.5"
    )
    parser.add_argument("embeddings", help="Path to subtitle embeddings JSONL file")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for frame conversion")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("SUBTITLE SEARCH - Together AI with BAAI/bge-large-en-v1.5")
    print(f"Query: {args.query}")
    print(f"Top-k: {args.topk}")
    print(f"{'='*70}\n")

    # Search
    results = search_subtitles(args.embeddings, args.query, args.topk, args.fps)

    # Output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(results)} results:\n")
        for res in results:
            print(f"#{res['rank']} | Score: {res['score']:.4f} | Time: {res['time_formatted']} ({res['start']} - {res['end']})")
            print(f"  Frame: {res['start_frame']} - {res['end_frame']}")
            print(f"  Text: {res['text']}")
            print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
