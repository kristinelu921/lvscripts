#!/usr/bin/env python3
"""Lightweight Together client compatibility shim.

This project imports `together.Together` and `together.AsyncTogether` directly.
The execution environment may not have the official SDK installed, so this shim
provides a minimal API-compatible layer for the methods used by the pipeline:

- client.chat.completions.create(...)
- client.embeddings.create(...)
- AsyncTogether equivalents
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List
from urllib.parse import urljoin
import os

import requests


API_BASE_URL = "https://api.together.xyz/v1/"
TOGETHER_REQUEST_TIMEOUT_SECONDS = int(os.getenv("TOGETHER_REQUEST_TIMEOUT_SECONDS", "300"))


class TogetherError(RuntimeError):
    """Raised when a Together API request fails."""


def _read_api_key(api_key: str | None) -> str:
    """Return a normalized API key."""
    if not api_key:
        api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY".lower())
    if not api_key:
        raise TogetherError("Together API key is required.")
    return api_key


@dataclass
class _Message:
    content: Any


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Embedding:
    embedding: List[float]


class _Response:
    """Base response with a `.choices` property."""

    def __init__(self, payload: dict):
        self._payload = payload

        choices = payload.get("choices", [])
        self.choices = [_Choice(_Message(c.get("message", {}).get("content"))) for c in choices]


class _EmbeddingResponse:
    """Base embedding response with a `.data` property."""

    def __init__(self, payload: dict):
        self._payload = payload
        data = payload.get("data", [])
        self.data = [_Embedding(item.get("embedding", [])) for item in data]


class _Completions:
    def __init__(self, api_key: str, session: requests.Session):
        self.api_key = api_key
        self.session = session

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = urljoin(API_BASE_URL, endpoint)
        response = self.session.post(
            url,
            headers=self._build_headers(),
            data=json.dumps(payload),
            timeout=TOGETHER_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise TogetherError(f"Together chat completion failed ({response.status_code}): {response.text}")
        return response.json()

    def create(self, **kwargs) -> _Response:
        payload = dict(kwargs)
        return _Response(self._post("chat/completions", payload))


class _AsyncCompletions(_Completions):
    async def create(self, **kwargs) -> _Response:
        return super().create(**kwargs)


class _Embeddings:
    def __init__(self, api_key: str, session: requests.Session):
        self.api_key = api_key
        self.session = session

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = urljoin(API_BASE_URL, endpoint)
        response = self.session.post(
            url,
            headers=self._build_headers(),
            data=json.dumps(payload),
            timeout=TOGETHER_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise TogetherError(f"Together embeddings failed ({response.status_code}): {response.text}")
        return response.json()

    def create(self, **kwargs) -> _EmbeddingResponse:
        payload = dict(kwargs)
        return _EmbeddingResponse(self._post("embeddings", payload))


class _AsyncEmbeddings(_Embeddings):
    async def create(self, **kwargs) -> _EmbeddingResponse:
        return super().create(**kwargs)


class Together:
    def __init__(self, api_key: str | None = None, *_, **__):
        self.api_key = _read_api_key(api_key)
        self._session = requests.Session()
        self.chat = type("chat", (), {})()
        self.embeddings = type("embeddings", (), {})()
        self.chat.completions = _Completions(self.api_key, self._session)
        self.embeddings = _Embeddings(self.api_key, self._session)


class AsyncTogether:
    def __init__(self, api_key: str | None = None, *_, **__):
        self.api_key = _read_api_key(api_key)
        self._session = requests.Session()
        self.chat = type("chat", (), {})()
        self.embeddings = type("embeddings", (), {})()
        self.chat.completions = _AsyncCompletions(self.api_key, self._session)
        self.embeddings = _AsyncEmbeddings(self.api_key, self._session)
