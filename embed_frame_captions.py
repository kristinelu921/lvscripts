import argparse
import json
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
import torch
import numpy as np
from tqdm import tqdm
import asyncio
import time

FRAME_CAPTION_PATTERN = re.compile(r"^(?P<frame_path>frames/\S+)\s+seconds:\s+(?P<caption>.+)$")

with open("env.json", "r") as f:
    env = json.load(f)
    openai_key = env["openai_key"]
    together_key = env["together_key"]
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["TOGETHER_API_KEY"] = together_key

def _parse_frame_caption(raw: str) -> Optional[Dict[str, Union[str, int]]]:
    """Parse strings like: "frames/frame_0001 seconds: <caption>".

    Returns dict with frame_path, caption, frame_number, frame_index, frame_second.
    """
    match = FRAME_CAPTION_PATTERN.match(raw.strip())
    if not match:
        return None
    frame_path = match.group("frame_path")
    caption = match.group("caption").strip()
    frame_num_match = re.search(r"frame_(\d+)", frame_path)
    frame_number = int(frame_num_match.group(1)) if frame_num_match else None
    frame_index = frame_number - 1 if frame_number is not None else None
    # Frames extracted at 1 fps per pipeline instructions. Use seconds == frame_number or index.
    frame_second = frame_number if frame_number is not None else frame_index
    return {
        "frame_path": frame_path,
        "caption": caption,
        "frame_number": frame_number,
        "frame_index": frame_index,
        "frame_second": frame_second,
    }


def _coerce_to_text(value: Union[str, List[str], Dict]) -> Optional[str]:
    """Best-effort conversion of a JSON value into a caption string.

    - If string, return as-is
    - If list of strings, join with a space
    - If dict, try common text fields
    - Otherwise, return None
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Try to parse specialized "frames/... seconds: ..." format first
        parsed = _parse_frame_caption(value)
        if parsed is not None:
            return parsed["caption"]  # type: ignore[index]
        return value.strip() if value.strip() else None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        combined = " ".join(x.strip() for x in value if x and x.strip())
        return combined if combined else None
    if isinstance(value, dict):
        for key in ("caption", "text", "description", "sentence"):
            if key in value and isinstance(value[key], str):
                text = value[key].strip()
                if text:
                    return text
    return None


def _guess_id(item_key: Optional[str], item_value: Union[dict, str, List[str], None], fallback_index: int) -> str:
    """Derive a stable identifier for the caption entry.

    Priority:
    - Explicit id-like fields on dict values
    - The JSON key (for object-mapped inputs)
    - Fallback to a numeric index
    """
    if isinstance(item_value, dict):
        for key in ("id", "frame_id", "frame", "frame_path", "image", "uid", "name"):
            if key in item_value and isinstance(item_value[key], str) and item_value[key].strip():
                return item_value[key]
            if key in item_value and isinstance(item_value[key], (int, float)):
                return str(item_value[key])
    if item_key is not None:
        return str(item_key)
    return str(fallback_index)


def iter_records(json_root: Union[Dict, List]) -> Iterator[Dict[str, Union[str, int]]]:
    """Yield record dicts with at least {id, text}; may include frame metadata.

    Supported shapes:
    - { "some_id": "caption", ... }
    - { "some_id": ["cap1", "cap2"], ... }
    - [ { "id": "...", "caption": "..." }, ... ]
    - [ { "frame": "...", "text": "..." }, ... ]
    - { "items": [ ... ] } or { "captions": [ ... ] }
    """
    root = json_root

    # If wrapped list under a common key
    if isinstance(root, dict):
        for list_key in ("items", "captions", "data", "frames"):
            if list_key in root and isinstance(root[list_key], list):
                root = root[list_key]
                break

    if isinstance(root, dict):
        for key, value in root.items():
            if isinstance(value, str):
                parsed = _parse_frame_caption(value)
                if parsed is not None:
                    yield {
                        "id": parsed["frame_path"],
                        "text": parsed["caption"],
                        "frame_path": parsed["frame_path"],
                        "frame_number": parsed["frame_number"],
                        "frame_index": parsed["frame_index"],
                        "frame_second": parsed["frame_second"],
                    }
                    continue
            text = _coerce_to_text(value)
            if text:
                yield {"id": _guess_id(key, value, 0), "text": text}
    elif isinstance(root, list):
        for idx, value in enumerate(root):
            if isinstance(value, dict):
                text = _coerce_to_text(value)
                if text:
                    rec: Dict[str, Union[str, int]] = {
                        "id": _guess_id(None, value, idx),
                        "text": text,
                    }
                    # bubble up any known metadata fields
                    for mkey in ("frame_path", "frame", "frame_id", "frame_number", "frame_index", "frame_second"):
                        if mkey in value and isinstance(value[mkey], (str, int)):
                            rec[mkey] = value[mkey]
                    yield rec
            else:
                if isinstance(value, str):
                    parsed = _parse_frame_caption(value)
                    if parsed is not None:
                        yield {
                            "id": parsed["frame_path"],
                            "text": parsed["caption"],
                            "frame_path": parsed["frame_path"],
                            "frame_number": parsed["frame_number"],
                            "frame_index": parsed["frame_index"],
                            "frame_second": parsed["frame_second"],
                        }
                        continue
                text = _coerce_to_text(value)
                if text:
                    yield {"id": _guess_id(None, None, idx), "text": text}
    else:
        raise ValueError("Unsupported JSON root type. Expected object or array.")


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_load_env_keys(env_path: Optional[str]) -> None:
    if not env_path:
        return
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Env file not found: {env_path}")
    with open(env_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Only set if present, avoid overriding an already-set env var
    if isinstance(data, dict) and "openai_key" in data and data["openai_key"]:
        os.environ.setdefault("OPENAI_API_KEY", str(data["openai_key"]))


def embed_texts_sbert(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype=np.float32)


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
    return mat


def embed_texts(
    texts: List[str],
    provider: str,
    model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True,
    max_chars: Optional[int] = None,
) -> np.ndarray:
    if provider == "openai":
        return embed_texts_openai(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize=normalize,
            max_chars=max_chars,
        )
    # default to sbert
    return embed_texts_sbert(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
    )


def write_jsonl(out_path: str, records: List[Dict[str, Union[str, int]]], embeddings: np.ndarray) -> None:
    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (rec, emb) in enumerate(zip(records, embeddings)):
            out_rec = dict(rec)
            out_rec["embedding"] = emb.tolist()
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")



async def embed_one(cap_path, out_path):
    with open("env.json", "r") as f:
        env = json.load(f)
        openai_key = env["openai_key"]
    
    os.environ["OPENAI_API_KEY"] = openai_key
    model="text-embedding-3-large"

    data = load_json(cap_path)
    print(f"Extracting texts from {cap_path}...")
    records: List[Dict[str, Union[str, int]]] = list(iter_records(data))
    texts: List[str] = [str(r["text"]) for r in records]

    # Get device string for sbert (if using sbert provider)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run synchronous function in executor for async compatibility
    loop = asyncio.get_running_loop()
    start_time = time.time()
    #print("reached start time")
    print("Awaiting embeddings...")
    embeddings = await loop.run_in_executor(
        None,
        embed_texts,
        texts,
        "openai",
        model,
        64,
        device_str,
        True,
        1000000
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"time taken for whole doc: {total_time} seconds")
    print(f"Writing embeddings to {out_path}...")
    await loop.run_in_executor(None, write_jsonl, out_path, records, embeddings)
    print(f"Done with {out_path}")


async def embed_many(vid_folder, batch_size=10):
    curr_folder = vid_folder
    curr_paths = os.listdir(curr_folder)
    
    # Collect all valid tasks
    all_tasks = []
    task_info = []  # Store task metadata for error reporting

    print(curr_paths)
    for num in curr_paths:
        input_path = f'{curr_folder}/{num}/captions/frame_captions_sorted.json'
        output_path = f'{curr_folder}/{num}/captions/frame_captions_sorted_embeddings.jsonl'
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Skipping {num}: {input_path} not found")
            continue

        elif os.path.exists(output_path):
            print(f"Skipping {num}: {output_path} already exists")
            continue
    
        else:
            all_tasks.append(embed_one(input_path, output_path))
            task_info.append(num)
    
    # Process in batches
    total_tasks = len(all_tasks)
    failed_tasks = []
    for i in range(0, total_tasks, batch_size):
        batch_tasks = all_tasks[i:i+batch_size]
        batch_info = task_info[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_tasks + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (videos: {', '.join(batch_info)})")
        
        completed = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for j, result in enumerate(completed):
            if isinstance(result, Exception):
                print(f"Error processing video {batch_info[j]}: {result}")
                failed_tasks.append(batch_info[j])
            else:
                print(f"Successfully processed video {batch_info[j]}")
    
    print("\nAll videos processed")
    print(failed_tasks)
    return failed_tasks


async def main():
    await embed_many('./videos_two')
    

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()