# token_tracker.py
"""Small utility to record token usage per step.
Creates a global `log` list.  Call `record()` after every model call.
Later call `flush_csv()` to write a CSV, or read the in-memory `log`.
"""
from __future__ import annotations
import csv, time, os, json
from pathlib import Path
from typing import Dict, List

log: List[Dict] = []

def record(step: str, model: str, prompt_tokens: int, completion_tokens: int):
    """Append a token-usage entry."""
    log.append({
        "timestamp": time.time(),
        "step": step,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total": prompt_tokens + completion_tokens,
    })


def flush_csv(path: str = "token_usage.csv"):
    """Write the log to CSV (overwrites existing)."""
    if not log:
        return
    keys = log[0].keys()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log)

    # also JSON for easy loading
    with open(Path(path).with_suffix(".json"), "w") as f:
        json.dump(log, f, indent=2)


def num_tokens(text: str, approx: bool = True) -> int:
    """Very light token estimate: tries tiktoken/transformers; falls back to whitespace count."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback word count *1.3 heuristic
        return int(len(text.split()) * 1.3)
