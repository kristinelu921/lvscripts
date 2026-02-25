import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple, Optional, Sequence
try:
    from openai import AsyncOpenAI, OpenAI
except Exception:  # pragma: no cover - optional dependency in this environment
    AsyncOpenAI = None
    OpenAI = None
import numpy as np
import time
import fcntl
import asyncio

with open("env.json", "r") as f:
    env = json.load(f)
    together_key = env["together_key"]
os.environ["TOGETHER_API_KEY"] = together_key

def log(message, file_title):
    if not os.path.exists(file_title):
        os.makedirs(file_title)
    else:
        with open(f"{file_title}/log.log", "a") as f:
            f.write(message + "\n")
def safe_access_and_remove(filepath):
    while True:
        try:
            with open(filepath, 'r+') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    data = json.load(f)
                    #print("retrieved data: ", data)
                    f.seek(0)
                    json.dump({}, f)
                    f.truncate()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            break
        except IOError:
            time.sleep(0.1)
    return data #data is in the format: {"uid": "phrase to embed"}

def safe_write(filepath, items_to_add): #items to add is a dict of {"uid": "phrase"}
    max_attempts = 50  # 5 seconds max wait
    
    # Ensure file exists with empty JSON if it doesn't
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump({}, f)
    
    for attempt in range(max_attempts):
        try:
            with open(filepath, 'r+') as f:
                print(f"opened filepath: {filepath}")
                # Try to acquire lock - simplified approach
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking lock
                    print("acquired lock successfully")
                except IOError as e:
                    print(f"Could not acquire lock: {e}")
                    # Try to force unlock in case of stale lock
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        print("Released potential stale lock, retrying...")
                    except:
                        pass
                    time.sleep(0.1)
                    continue
                    
                try:
                    f.seek(0)  # Make sure we're at the beginning
                    content = f.read()
                    if content:
                        data = json.loads(content)
                    else:
                        data = {}
                    #print("data before", data)
                    data.update(items_to_add)
                    #print("data updated:", data)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    print("safe_write completed successfully")
                    return data
                finally:
                    print("Releasing lock in safe_write")
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception as e:
                        print(f"Warning: Could not release lock: {e}")
        except Exception as e:
            print(f"Exception in safe_write: {e}, attempt {attempt + 1}/{max_attempts}")
            time.sleep(0.1)
    
    raise TimeoutError(f"Could not write to {filepath} after {max_attempts} attempts") 


def load_jsonl_embeddings(path: str) -> Tuple[List[Dict], np.ndarray]:
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
        raise RuntimeError("No embeddings found in JSONL.")
    matrix = np.vstack(vectors)
    return records, matrix


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def _maybe_load_env_keys(env_path: str = None) -> None:
    if not env_path:
        return
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Env file not found: {env_path}")
    with open(env_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "openai_key" in data and data["openai_key"]:
        os.environ.setdefault("OPENAI_API_KEY", str(data["openai_key"]))


def embed_query(
    query: str,
    provider: str,
    model_name: str,
    device: str = None,
    normalize: bool = True,
) -> np.ndarray:
    """Synchronous version of embed_query for CLI usage"""
    if provider == "together":
        from together import Together
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string, got: {query}")
        client = Together(api_key=together_key)
        resp = client.embeddings.create(model=model_name, input=query)

        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        # L2 normalize
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec
    elif provider == "openai":
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required for provider=openai. Install with: pip install openai"
            )
        client = OpenAI()
        # Ensure query is a non-empty string
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string, got: {query}")
        resp = client.embeddings.create(model=model_name, input=query)

        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        vec = model.encode([query], normalize_embeddings=normalize)
        return np.asarray(vec[0], dtype=np.float32)

async def embed_query_async(
    query: str,
    provider: str,
    model_name: str,
    device: str = None,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a query string asynchronously

    Args:
        query: Query text to embed
        provider: Provider to use ("together", "openai", or "sbert")
        model_name: Model identifier
        device: Device for local models (sbert only)
        normalize: Whether to L2 normalize the embedding

    Returns:
        np.ndarray: L2-normalized embedding vector
    """
    if provider == "openai":
        if AsyncOpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required for provider=openai. "
                "Install with: pip install openai"
            )
        client = AsyncOpenAI()
        # Ensure query is a non-empty string
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string, got: {query}")
        resp = await client.embeddings.create(model=model_name, input=query)

        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec
    elif provider == "together":
        # Use Together AI's async API
        from together import AsyncTogether
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string, got: {query}")

        client = AsyncTogether(api_key=together_key)
        resp = await client.embeddings.create(model=model_name, input=query)

        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        # L2 normalize
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        vec = model.encode([query], normalize_embeddings=normalize)
        return np.asarray(vec[0], dtype=np.float32)

def embed_texts_openai(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
    max_chars: Optional[int] = None,
) -> np.ndarray:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is required for provider=openai. Install with: pip install openai"
        ) from e

    client = OpenAI()

    def _truncate(text: str) -> str:
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = [_truncate(t) for t in texts[start : start + batch_size]]
        start = time.time()
        resp = client.embeddings.create(model=model_name, input=chunk)
        end = time.time()
        elapsed = end - start
        #print(f"time for one {elapsed}")
        for item in resp.data:
            vectors.append(np.asarray(item.embedding, dtype=np.float32))

    mat = np.vstack(vectors)
    if normalize:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
    #print("embedded and this is the mat: ", mat)
    return mat

def embed_texts_together(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
    max_chars: Optional[int] = None,
) -> np.ndarray:
    """Embed texts using Together AI API with BAAI models"""
    try:
        from together import Together
    except Exception as e:
        raise RuntimeError(
            "The 'together' package is required for provider=together. Install with: pip install together"
        ) from e

    client = Together(api_key=together_key)

    def _truncate(text: str) -> str:
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = [_truncate(t) for t in texts[start : start + batch_size]]
        start_time = time.time()
        resp = client.embeddings.create(model=model_name, input=chunk)
        end_time = time.time()
        elapsed = end_time - start_time
        #print(f"time for one {elapsed}")
        for item in resp.data:
            vectors.append(np.asarray(item.embedding, dtype=np.float32))

    mat = np.vstack(vectors)
    if normalize:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
    #print("embedded and this is the mat: ", mat)
    return mat

async def batch_embed_query_async(query_path: str, output_path: str, provider: str, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None, normalize: bool = True):
    """Batch embed queries using Together AI with BAAI embedder"""
    while True:
        await asyncio.sleep(2)
        #("batch task running")
        query_dict = safe_access_and_remove(query_path)


        # Skip if no queries to process
        if not query_dict:
            continue

        query_list = []
        uid_list = []
        for uid, query in query_dict.items():
            query_list.append(query)
            uid_list.append(uid)

        #print("query list made", query_list)

        # Only process if we have queries
        if query_list:
            # Use Together AI with BAAI embedder
            if provider == "together":
                embeds = embed_texts_together(query_list, model_name, normalize=normalize)
            else:
                # Fallback to OpenAI if explicitly specified
                embeds = embed_texts_openai(query_list, model_name, normalize=normalize)
            embed_list = embeds.tolist()
            res_dict = {}
            for i in range(len(uid_list)):
                res_dict[uid_list[i]] = embed_list[i]
            #print("made to res_dict", res_dict)
            safe_write(output_path, res_dict)

async def wait_embedding(uid, query, provider, model_name, normalize):
    # Wait for the embedding to be processed
    max_attempts = 30  # Maximum 60 seconds wait (30 * 2)
   #print(f"this embed fctn reached")
    for attempt in range(max_attempts):
        #print(attempt)
        await asyncio.sleep(2)
        try:
            # Single file operation with proper locking
                
            print(f"wait_embedding: Attempting to open ret_embeddings.json for uid {uid}")
            with open('ret_embeddings.json', 'r+') as f:
                #print(f"wait_embedding: File opened, acquiring lock for uid {uid}")
                lock_acquired = False
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    #print(f"wait_embedding: Lock acquired for uid {uid}")
                    
                    data = json.load(f)
                    #print("data loaded", data)
                    if uid in data:
                        embedding = data[uid]
                        # Remove it from the file
                        del data[uid]
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                        print(f"wait_embedding: Found and returning embedding for {uid}")
                        return np.array(embedding, dtype=np.float32)
                except IOError as e:
                    if lock_acquired:
                        # Lock was acquired but something else failed
                        print(f"wait_embedding: IOError after lock acquired: {e}")
                    else:
                        # Couldn't acquire lock
                        print(f"wait_embedding: Could not acquire lock for {uid}, will retry")
                finally:
                    if lock_acquired:
                        try:
                            fcntl.flock(f, fcntl.LOCK_UN)
                            print(f"wait_embedding: Lock released for {uid}")
                        except Exception as e:
                            print(f"wait_embedding: Warning - could not release lock: {e}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # File doesn't exist or uid not found yet, continue waiting
            continue
        except IOError as e:
            # Lock is held by another process, wait and retry
            await asyncio.sleep(0.5)
            continue
    raise TimeoutError(f"Embedding for uid {uid} not available after {max_attempts * 2} seconds")



def cosine_topk(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure 2D
    q = query_vec.reshape(1, -1)
    # Normalize both sides for cosine
    qn = l2_normalize(q)
    cn = l2_normalize(corpus)
    scores = (cn @ qn.T).ravel()
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        idx = part[np.argsort(-scores[part])]
    return idx, scores[idx]


def _select_clip_caption_embedder(dim: int) -> Sequence[Tuple[str, str]]:
    """Return candidate embedder backends/models for a caption embedding dimension."""
    if dim == 1024:
        # Try requested nvidia model first, then fallback to Together's available 1024-d alternative.
        return (
            ("together", "nvidia/NV-Embed-v2"),
            ("together", "BAAI/bge-large-en-v1.5"),
        )
    if dim == 768:
        # Default for the pipeline's frame/clip caption embeddings.
        # Prefer Together API to avoid local SBERT dependency.
        return (
            ("together", "Alibaba-NLP/gte-modernbert-base"),
        )
    # Safe fallback for unexpected dims.
    return (
        ("together", "Alibaba-NLP/gte-modernbert-base"),
        ("together", "BAAI/bge-large-en-v1.5"),
    )


async def _embed_query_with_fallback(query: str, target_dim: int) -> np.ndarray:
    """Try multiple embedding models until one works and returns the target dimensionality."""
    last_error = None
    for provider, model_name in _select_clip_caption_embedder(target_dim):
        try:
            query_vec = await embed_query_async(
                query,
                provider=provider,
                model_name=model_name,
                normalize=True,
            )
            if query_vec.shape[0] != target_dim:
                print(
                    f"⚠️ Skipped {provider}:{model_name} due to dimension mismatch "
                    f"({query_vec.shape[0]} != {target_dim})"
                )
                continue
            print(f"✓ Query embedded with {provider}:{model_name}")
            return query_vec
        except Exception as e:
            last_error = e
            print(f"⚠️ Embedding with {provider}:{model_name} failed: {e}")

    raise RuntimeError(
        f"No available clip-caption embedder could produce {target_dim} dims. "
        f"Last error: {last_error}"
    )


def format_time_s(seconds: int) -> str:
    if seconds is None or not isinstance(seconds, (int, np.integer)):
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def safe_write_single(filepath, contents):
    while True:
        try:
            with open(filepath, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    data = json.load(f)
                    data[contents["uid"]] = contents["query"]
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            break
        except IOError:
            time.sleep(0.1)
    print("written to file!")

def safe_remove(filepath, items_to_remove):
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            with open(filepath, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    data = json.load(f)
                    if items_to_remove["uid"] in data:
                        embedding = data[items_to_remove["uid"]]
                        del data[items_to_remove["uid"]]
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                        return embedding
                    else:
                        # UID not found, wait and retry
                        pass
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except (IOError, json.JSONDecodeError) as e:
            if attempt == max_attempts - 1:
                raise e
        time.sleep(2)
    raise KeyError(f"UID {items_to_remove['uid']} not found after {max_attempts} attempts")
    

async def search_captions(vid_path, question_uid, query, embeddings_path, topk=30):
    """Search captions using semantic similarity with Together AI BAAI embedder

    Args:
        vid_path: Path to video directory
        question_uid: Unique question identifier
        query: Search query text
        embeddings_path: Path to embeddings JSONL file
        topk: Number of top results to return

    Returns:
        List of search results with similarity scores
    """
    import os

    print(f"Searching captions with semantic embeddings")
    print(f"Embeddings path: {embeddings_path}")

    # Load pre-computed caption embeddings
    records, matrix = load_jsonl_embeddings(embeddings_path)
    print(f"  Loaded {len(records)} caption embeddings (shape: {matrix.shape})")

    # Embed query with a provider/model that matches embedding dimensions.
    # This defaults to Together and falls back only if needed.
    query_emb = await _embed_query_with_fallback(query, matrix.shape[1])
    print(f"✓ Query embedded: {query[:100]}..." if len(query) > 100 else f"✓ Query embedded: {query}")
    print(f"  Query embedding shape: {query_emb.shape}")

    # Check dimension compatibility
    if query_emb.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query embedding has {query_emb.shape[0]} dims "
            f"but caption embeddings have {matrix.shape[1]} dims. "
            f"Ensure query and captions use the same embedding model."
        )

    # Find most similar captions using cosine similarity
    idx, scores = cosine_topk(query_emb, matrix, k=topk)

    results = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        rec = dict(records[int(i)])
        id_cap_score = {}
        id_cap_score[rec["id"]] = rec["id"]
        id_cap_score["text"] = rec["text"]
        id_cap_score["similarity score"] = float(score)
        results.append(id_cap_score)

    print(f"✓ Found top {len(results)} similar captions (scores: {scores[0]:.3f} to {scores[-1]:.3f})")
    message = f"Caption search results: {results}"
    log(message, f"logs/log_video_{vid_path}_{question_uid}")
    return results


async def search_clip_captions(vid_path, question_uid, query, embeddings_path, topk=30):
    """Search clip captions using semantic similarity with nvidia/NV-Embed-v2

    Args:
        vid_path: Path to video directory
        question_uid: Unique question identifier
        query: Search query text
        embeddings_path: Path to clip embeddings JSONL file
        topk: Number of top results to return

    Returns:
        List of search results with clip time ranges and similarity scores
    """
    print(f"Searching clip captions with nvidia/NV-Embed-v2 (fallbacks enabled)")
    print(f"Embeddings path: {embeddings_path}")

    # Load pre-computed clip embeddings
    try:
        records, matrix = load_jsonl_embeddings(embeddings_path)
    except Exception as e:
        print(f"✗ Failed to load clip embeddings from {embeddings_path}: {e}")
        return []

    if matrix is None or matrix.size == 0:
        print(f"✗ Clip embeddings file is empty or invalid: {embeddings_path}")
        return []

    print(f"  Loaded {len(records)} clip embeddings (shape: {matrix.shape})")

    try:
        query_emb = await _embed_query_with_fallback(query, matrix.shape[1])
        query_emb = np.array(query_emb).reshape(1, -1)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []

    # Check dimension compatibility
    if query_emb.shape[1] != matrix.shape[1]:
        print(
            f"Dimension mismatch: query has {query_emb.shape[1]} dims "
            f"but caption embeddings have {matrix.shape[1]} dims. "
            f"Ensure query and captions use the same embedding model."
        )
        return []

    # Find most similar clips using cosine similarity
    idx, scores = cosine_topk(query_emb, matrix, k=topk)

    results = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        rec = dict(records[int(i)])
        result = {
            "id": rec["id"],  # e.g., "clip_0_120"
            "text": rec["text"],
            "similarity score": float(score),
            "start": rec.get("start", 0),
            "end": rec.get("end", 0)
        }
        results.append(result)

    print(f"✓ Found top {len(results)} similar clips (scores: {scores[0]:.3f} to {scores[-1]:.3f})")
    message = f"Clip caption search results: {results}"
    log(message, f"logs/log_video_{vid_path}_{question_uid}")
    return results


async def search_multi_caption_types(vid_path, question_uid, query, caption_types=None, topk=30):
    """Search multiple caption types using semantic similarity

    Args:
        vid_path: Path to video directory
        question_uid: Unique question identifier
        query: Search query text
        caption_types: List of caption types to search. Options: ['characters_actions', 'objects', 'scene_setting_mood', 'frames']
                      If None, searches all available types.
        topk: Number of top results to return per caption type

    Returns:
        Dictionary mapping caption types to search results
    """
    import os

    # Default to searching all types if none specified
    if caption_types is None:
        caption_types = ['characters_actions', 'objects', 'scene_setting_mood', 'frames']

    # Map caption types to their embedding file paths
    caption_type_files = {
        'characters_actions': 'captions/clip_captions_characters_embeddings.jsonl',
        'objects': 'captions/clip_captions_objects_embeddings.jsonl',
        'scene_setting_mood': 'captions/clip_captions_scene_embeddings.jsonl',
        'frames': 'captions/frame_captions_sorted_embeddings.jsonl'
    }

    results_by_type = {}

    for caption_type in caption_types:
        if caption_type not in caption_type_files:
            print(f"⚠ Warning: Unknown caption type '{caption_type}', skipping")
            continue

        embeddings_path = os.path.join(vid_path, caption_type_files[caption_type])

        if not os.path.exists(embeddings_path):
            print(f"⚠ Warning: Embeddings not found for {caption_type} at {embeddings_path}, skipping")
            continue

        print(f"\n--- Searching {caption_type} captions ---")

        try:
            # Search this caption type
            results = await search_captions(
                vid_path=vid_path,
                question_uid=question_uid,
                query=query,
                embeddings_path=embeddings_path,
                topk=topk
            )
            results_by_type[caption_type] = results
        except Exception as e:
            print(f"✗ Error searching {caption_type}: {e}")
            results_by_type[caption_type] = []

    return results_by_type
    
def main():
    parser = argparse.ArgumentParser(
        description="Search most similar captions using Together AI with BAAI/bge-large-en-v1.5 (1024 dims)"
    )
    parser.add_argument("embeddings", help="Path to *_embeddings.jsonl produced by embed_frame_captions.py")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    parser.add_argument("--provider", choices=["together", "sbert", "openai"], default="together",
                       help="Embedding provider (default: together)")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                       help="Model name for query embedding (default: BAAI/bge-large-en-v1.5 for Together AI)")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (sbert only)")
    parser.add_argument("--env", default=None, help="Path to env.json with API keys")
    parser.add_argument("--print-json", action="store_true", help="Output results as JSONL to stdout")

    args = parser.parse_args()

    _maybe_load_env_keys(args.env)

    print(f"\n{'='*60}")
    print(f"Searching with {args.provider} provider: {args.model}")
    print(f"{'='*60}\n")

    records, matrix = load_jsonl_embeddings(args.embeddings)
    print(f"Loaded {len(records)} embeddings with dimension {matrix.shape[1]}")

    # Verify dimension compatibility with BAAI/bge-large-en-v1.5 (1024 dims)
    if args.provider == "together" and matrix.shape[1] != 1024:
        print(f"⚠ Warning: Embeddings have {matrix.shape[1]} dimensions, but BAAI/bge-large-en-v1.5 outputs 1024 dimensions")
        print(f"  Make sure your embeddings were created with the same model!")

    # Set default models for each provider if not specified
    if args.provider == "openai" and args.model == "BAAI/bge-large-en-v1.5":
        args.model = "text-embedding-3-small"
        print(f"Switched to OpenAI model: {args.model}")
    elif args.provider == "sbert" and args.model == "BAAI/bge-large-en-v1.5":
        args.model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Switched to SBERT model: {args.model}")

    qvec = embed_query(args.query, provider=args.provider, model_name=args.model, device=args.device, normalize=True)
    print(f"Query embedding shape: {qvec.shape}")

    idx, scores = cosine_topk(qvec, matrix, k=args.topk)
    

    results: List[Dict] = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        rec = dict(records[int(i)])
        rec["rank"] = rank
        rec["score"] = float(score)
        rec["frame_path"] = rec["text"].split(" second")[0]
        # convenience: alias timestamp fields if present
        if "frame_second" in rec and isinstance(rec["frame_second"], (int, float)):
            rec["time_str"] = format_time_s(int(rec["frame_second"]))
        results.append(rec)

    if args.print_json:
        print("reached")
        for rec in results:
            print(json.dumps(rec, ensure_ascii=False))
    else:
        for rec in results:
            rid = rec.get("id", "")
            rtime = rec.get("time_str", "")
            rpath = rec.get("frame_path", "")
            rnum = rec.get("frame_number", "")
            #text = rec.get("text", "")
            score = rec.get("score", 0.0)
            print(f"#{rec['rank']} score={score:.4f} id={rid} time={rtime} frame_number={rnum} path={rpath}\n")


if __name__ == "__main__":
    main()
