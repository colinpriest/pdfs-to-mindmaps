# =============================
# FILE: src/llm/client.py
# =============================
from typing import Callable, Type, TypeVar
import os
import random
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import instructor
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

_openai = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
# Wrap with Instructor to get Pydantic parsing out of the box
client = instructor.from_openai(_openai, mode=instructor.Mode.JSON)

# Defaults
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

_T = TypeVar("_T")


def _ensure_api_key() -> None:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it before running the pipeline.")


def _call_with_backoff(func: Callable[[], _T], max_retries: int = 5) -> _T:
    delay = 0.5
    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError:
            if attempt >= max_retries:
                raise
            jitter = random.uniform(0, 0.25)
            time.sleep(min(delay + jitter, 10))
            delay *= 2


def chat_structured(response_model: Type[BaseModel], system: str, user: str, temperature: float = 0.1) -> BaseModel:
    _ensure_api_key()
    return _call_with_backoff(
        lambda: client.chat.completions.create(
            model=CHAT_MODEL,
            response_model=response_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    )


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Simple chat function that returns plain text response."""
    _ensure_api_key()
    resp = _call_with_backoff(
        lambda: _openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    )
    return resp.choices[0].message.content


def embed(texts: list[str]) -> list[list[float]]:
    _ensure_api_key()
    resp = _call_with_backoff(
        lambda: _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    )
    return [d.embedding for d in resp.data]
