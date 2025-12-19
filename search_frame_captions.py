import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple, Optional
from openai import AsyncOpenAI, OpenAI
import numpy as np
import time
import fcntl
import asyncio

with open("env.json", "r") as f:
    env = json.load(f)
    openai_key = env["openai_key"]
os.environ["OPENAI_API_KEY"] = openai_key

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
    if provider == "openai":
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
    if provider == "openai":
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

async def batch_embed_query_async(query_path: str, output_path: str, provider: str, model_name: str = "text-embedding-3-large", device: str = None, normalize: bool = True):
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
            embeds = embed_texts_openai(query_list, model_name) 
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
    

async def search_captions(vid_path, question_uid, query, embeddings_path, topk = 30):
    # Directly compute the embedding without queueing/waiting
    # Check if open-source embeddings exist, use those instead
    import os
    sbert_embeddings_path = embeddings_path.replace('frame_captions_enriched_embeddings.jsonl',
                                                      'frame_captions_enriched_embeddings_sbert.jsonl')
    if os.path.exists(sbert_embeddings_path):
        embeddings_path = sbert_embeddings_path
        provider = "sbert"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using open-source embeddings: {sbert_embeddings_path}")
    else:
        provider = "openai"
        model_name = "text-embedding-3-large"
        print(f"Using OpenAI embeddings: {embeddings_path}")

    # Use open-source embedding model instead of OpenAI
    query_emb = await embed_query_async(query, provider=provider, model_name=model_name, normalize=True)
    print(f"search for {query} embedded.")
    records, matrix = load_jsonl_embeddings(embeddings_path)
    idx, scores = cosine_topk(query_emb, matrix, k=topk)
    results = []
    for rank, (i, score) in enumerate(zip(idx, scores), start=1):
        rec = dict(records[int(i)])
        id_cap_score = {}
        id_cap_score[rec["id"]] = rec["id"]
        id_cap_score["text"] = rec["text"]
        id_cap_score["similarity score"] = float(score)
        results.append(id_cap_score)
    message = f"Caption search results: {results}"
    log(message, f"logs/log_video_{vid_path}_{question_uid}")
    return results
    
def main():
    parser = argparse.ArgumentParser(description="Search most similar captions with timestamps.")
    parser.add_argument("embeddings", help="Path to *_embeddings.jsonl produced by embed_frame_captions.py")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    parser.add_argument("--provider", choices=["sbert", "openai"], default="openai")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model name for query embedding")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (sbert only)")
    parser.add_argument("--env", default=None, help="Path to env.json with 'openai_key'")
    parser.add_argument("--print-json", action="store_true", help="Output results as JSONL to stdout")

    args = parser.parse_args()

    _maybe_load_env_keys(args.env)
    records, matrix = load_jsonl_embeddings(args.embeddings)
    if args.provider == "openai":
        args.model = "text-embedding-3-small"
    qvec = embed_query(args.query, provider=args.provider, model_name=args.model, device=args.device, normalize=True)
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