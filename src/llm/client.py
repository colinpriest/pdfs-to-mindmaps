# =============================
# FILE: src/llm/client.py
# =============================
from typing import Any, Type
import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

_openai = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# Wrap with Instructor to get Pydantic parsing out of the box
client = instructor.from_openai(_openai, mode=instructor.Mode.JSON)

# Defaults
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

def chat_structured(response_model: Type[BaseModel], system: str, user: str, temperature: float = 0.1) -> BaseModel:
    return client.chat.completions.create(
        model=CHAT_MODEL,
        response_model=response_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Simple chat function that returns plain text response."""
    resp = _openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def embed(texts: list[str]) -> list[list[float]]:
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]