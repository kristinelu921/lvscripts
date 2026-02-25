#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from together import Together
from tqdm import tqdm

FRAME_CAPTION_PATTERN = re.compile(r"^(?P<frame_path>frames/\S+)\s+seconds:\s+(?P<caption>.+)$")

CAPTION_STYLE_PATHS = {
    "frame": ("captions/frame_captions_sorted.json", "captions/frame_captions_sorted_embeddings.jsonl"),
    "clip": ("captions/clip_captions.json", "captions/clip_embeddings.jsonl"),
    "clip-frame": ("captions/clip_frame_captions.json", "clip_frame_embeddings.jsonl"),
}
DEFAULT_EMBEDDING_MODELS = {
    "frame": "BAAI/bge-large-en-v1.5",
    "clip": "Alibaba-NLP/gte-modernbert-base",
    "clip-frame": "nvidia/NV-Embed-v2",
}

# Load Together API key and keep current behavior for scripts that rely on env variable.
with open("env.json", "r") as f:
    env = json.load(f)
    together_key = env["together_key"]
os.environ["TOGETHER_API_KEY"] = together_key


def _parse_frame_caption(raw: str) -> Optional[Dict[str, Union[str, int]]]:
    """Parse strings like: \"frames/frame_0001 seconds: <caption>\"."""
    match = FRAME_CAPTION_PATTERN.match(raw.strip())
    if not match:
        return None
    frame_path = match.group("frame_path")
    caption = match.group("caption").strip()
    frame_num_match = re.search(r"frame_(\d+)", frame_path)
    frame_number = int(frame_num_match.group(1)) if frame_num_match else None
    frame_index = frame_number - 1 if frame_number is not None else None
    frame_second = frame_number if frame_number is not None else frame_index
    return {
        "frame_path": frame_path,
        "caption": caption,
        "frame_number": frame_number,
        "frame_index": frame_index,
        "frame_second": frame_second,
    }


def _coerce_to_text(value: Union[str, List[str], Dict, None]) -> Optional[str]:
    """Best-effort conversion of a JSON value into a caption string."""
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _parse_frame_caption(value)
        if parsed is not None:
            return parsed["caption"]  # type: ignore[index]
        cleaned = value.strip()
        return cleaned if cleaned else None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        joined = " ".join(x.strip() for x in value if x and x.strip())
        return joined if joined else None
    if isinstance(value, dict):
        for key in ("caption", "text", "description", "sentence"):
            if key in value and isinstance(value[key], str):
                text = value[key].strip()
                if text:
                    return text
    return None


def _guess_id(item_key: Optional[str], item_value: Union[dict, str, List[str], None], fallback_index: int) -> str:
    if isinstance(item_value, dict):
        for key in ("id", "frame_id", "frame", "frame_path", "image", "uid", "name", "filename", "clip_filename"):
            if key in item_value and isinstance(item_value[key], str) and item_value[key].strip():
                return item_value[key]
            if key in item_value and isinstance(item_value[key], (int, float)):
                return str(item_value[key])
    if item_key is not None:
        return str(item_key)
    return str(fallback_index)


def iter_records(json_root: Union[Dict, List]) -> Iterator[Dict[str, Union[str, int]]]:
    """Existing frame-style parser for ``frame_captions[_sorted].json``-style inputs."""
    root = json_root

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
                if "captions" in value and isinstance(value["captions"], dict):
                    clip_id = value.get("clip_filename", idx)
                    for caption_type, caption_text in value["captions"].items():
                        if isinstance(caption_text, str) and caption_text.strip():
                            rec: Dict[str, Union[str, int]] = {
                                "id": f"{clip_id}_{caption_type}",
                                "text": caption_text.strip(),
                                "clip_filename": clip_id,
                                "caption_type": caption_type,
                            }
                            for mkey in ("start", "end", "duration"):
                                if mkey in value and isinstance(value[mkey], (int, float)):
                                    rec[mkey] = value[mkey]
                            yield rec
                    continue

                text = _coerce_to_text(value)
                if text:
                    rec: Dict[str, Union[str, int]] = {
                        "id": _guess_id(None, value, idx),
                        "text": text,
                    }
                    for mkey in ("frame_path", "frame", "frame_id", "frame_number", "frame_index", "frame_second"):
                        if mkey in value and isinstance(value[mkey], (str, int, float)):
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


def _iter_records_for_style(json_root: Union[Dict, List], caption_style: str) -> Iterator[Dict[str, Union[str, int]]]:
    if caption_style == "frame":
        yield from iter_records(json_root)
        return

    if caption_style in ("clip", "clip-frame"):
        entries = json_root
        if isinstance(entries, dict):
            for key in ("items", "clips", "data", "captions"):
                if key in entries and isinstance(entries[key], list):
                    entries = entries[key]
                    break
        if not isinstance(entries, list):
            return

        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            text = _coerce_to_text(item)
            if not text:
                continue

            rec: Dict[str, Union[str, int]] = {
                "id": _guess_id(None, item, idx),
                "text": text,
            }

            filename = item.get("filename") or item.get("clip_filename") or item.get("video")
            if filename:
                rec["filename"] = str(filename)

            start = item.get("start")
            end = item.get("end")
            if isinstance(start, (int, float)):
                rec["start"] = start
            if isinstance(end, (int, float)):
                rec["end"] = end

            if "duration" in item and isinstance(item["duration"], (int, float)):
                rec["duration"] = item["duration"]

            if filename:
                rec["id"] = str(Path(str(filename)).stem)
            elif isinstance(start, (int, float)) and isinstance(end, (int, float)):
                rec["id"] = f"clip_{int(start)}_{int(end)}"
            elif rec["id"] and rec["id"].endswith(".mp4"):
                rec["id"] = Path(str(rec["id"])).stem

            yield rec
        return

    raise ValueError(f"Unsupported caption style: {caption_style}")


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
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
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
    except Exception as e:
        raise RuntimeError(
            "The 'openai' package is required for provider=openai. Install with: pip install openai"
        ) from e

    client = OpenAI()

    def _truncate(text: str) -> str:
        return text[:max_chars] if max_chars is not None and len(text) > max_chars else text

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = [_truncate(t) for t in texts[start : start + batch_size]]
        start_time = time.time()
        resp = client.embeddings.create(model=model_name, input=chunk)
        _ = time.time() - start_time
        for item in resp.data:
            vectors.append(np.asarray(item.embedding, dtype=np.float32))

    mat = np.vstack(vectors)
    if normalize:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
    return mat


def embed_texts_together(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
    max_chars: Optional[int] = None,
) -> np.ndarray:
    client = Together()

    def _truncate(text: str) -> str:
        return text[:max_chars] if max_chars is not None and len(text) > max_chars else text

    vectors: List[np.ndarray] = []
    for idx in range(0, len(texts), batch_size):
        chunk = [_truncate(t) for t in texts[idx : idx + batch_size]]
        resp = client.embeddings.create(model=model_name, input=chunk)
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
    if provider == "sbert":
        return embed_texts_sbert(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
        )
    return embed_texts_together(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        max_chars=max_chars,
    )


def write_jsonl(out_path: str, records: List[Dict[str, Union[str, int]]], embeddings: np.ndarray) -> None:
    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, emb in zip(records, embeddings):
            out_rec = dict(rec)
            out_rec["embedding"] = emb.tolist()
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")


def _resolve_caption_paths(video_dir: Union[str, Path], caption_style: str) -> Tuple[str, str]:
    video_dir = Path(video_dir)
    if caption_style not in CAPTION_STYLE_PATHS:
        raise ValueError(f"Unsupported caption style: {caption_style}")
    rel_input, rel_output = CAPTION_STYLE_PATHS[caption_style]
    return str(video_dir / rel_input), str(video_dir / rel_output)


def _default_model(caption_style: str) -> str:
    if caption_style not in DEFAULT_EMBEDDING_MODELS:
        raise ValueError(f"Unsupported caption style: {caption_style}")
    return DEFAULT_EMBEDDING_MODELS[caption_style]


async def embed_one(cap_path: str, out_path: str, caption_style: str, provider: str = "together", model: Optional[str] = None,
                  batch_size: int = 64, device: Optional[str] = None, max_chars: Optional[int] = 1500) -> None:
    data = load_json(cap_path)
    model = model or _default_model(caption_style)
    print(f"Extracting texts from {cap_path}...")

    records: List[Dict[str, Union[str, int]]] = list(_iter_records_for_style(data, caption_style))
    texts: List[str] = [str(r["text"]) for r in records]
    if not texts:
        print(f"  No valid captions found in {cap_path}")
        return

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    loop = asyncio.get_running_loop()
    start_time = time.time()
    print(f"Generating embeddings using {provider} model: {model}...")
    print(f"  {len(texts)} texts to embed")
    embeddings = await loop.run_in_executor(
        None,
        embed_texts,
        texts,
        provider,
        model,
        batch_size,
        device_str,
        True,
        max_chars,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Writing embeddings to {out_path}...")
    await loop.run_in_executor(None, write_jsonl, out_path, records, embeddings)
    print(f"Done with {out_path}")


async def embed_many(
    vid_folder: str,
    batch_size: int = 10,
    caption_style: str = "frame",
    provider: str = "together",
    model: Optional[str] = None,
    force: bool = False,
) -> List[str]:
    curr_folder = Path(vid_folder)
    if not curr_folder.exists() or not curr_folder.is_dir():
        print(f"Invalid input directory: {curr_folder}")
        return []

    curr_paths = sorted([d for d in curr_folder.iterdir() if d.is_dir()], key=lambda x: x.name)

    all_tasks = []
    task_info = []

    for video_dir in curr_paths:
        input_path, output_path = _resolve_caption_paths(video_dir, caption_style)

        if not os.path.exists(input_path):
            print(f"Skipping {video_dir.name}: {input_path} not found")
            continue

        if os.path.exists(output_path) and not force:
            print(f"Skipping {video_dir.name}: {output_path} already exists")
            continue

        all_tasks.append(embed_one(input_path, output_path, caption_style, provider=provider, model=model or _default_model(caption_style), batch_size=batch_size))
        task_info.append(video_dir.name)

    total_tasks = len(all_tasks)
    failed_tasks: List[str] = []

    for i in range(0, total_tasks, batch_size):
        batch_tasks = all_tasks[i : i + batch_size]
        batch_info = task_info[i : i + batch_size]
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
    if failed_tasks:
        print(f"Failed videos: {failed_tasks}")
    return failed_tasks


async def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for caption files")
    parser.add_argument("input", help="Path to captions file or directory containing video folders")
    parser.add_argument("--output", help="Output path for embeddings (required for single file mode)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument(
        "--caption-style",
        choices=["frame", "clip", "clip-frame"],
        default="frame",
        help="Caption JSON layout to embed",
    )
    parser.add_argument(
        "--provider",
        choices=["together", "openai", "sbert"],
        default="together",
        help="Embedding provider",
    )
    parser.add_argument("--model", help="Embedding model (defaults by style)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embeddings files")
    parser.add_argument("--env", help="Optional env file containing openai_key")

    args = parser.parse_args()
    _maybe_load_env_keys(args.env)

    model = args.model or _default_model(args.caption_style)

    if os.path.isfile(args.input):
        if not args.output:
            parser.error("--output is required when processing a single file")
        print(f"Processing single file: {args.input}")
        await embed_one(
            args.input,
            args.output,
            caption_style=args.caption_style,
            provider=args.provider,
            model=model,
            batch_size=64,
            max_chars=1500,
        )
    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        await embed_many(
            args.input,
            batch_size=args.batch_size,
            caption_style=args.caption_style,
            provider=args.provider,
            model=model,
            force=args.force,
        )
    else:
        parser.error(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
