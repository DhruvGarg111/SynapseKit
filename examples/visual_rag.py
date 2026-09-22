"""Run visual RAG over a scanned PDF or PowerPoint deck.

Install the optional visual dependencies and a compatible PyTorch build first:

    pip install "synapsekit[visual]"

Set OPENAI_API_KEY in the environment before running this example. The value
is never stored in this repository.
"""

from __future__ import annotations

import asyncio
import os
import sys

from synapsekit import ColPaliEmbeddings, MultimodalRAG


async def main(path: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "[REDACTED]")
    if api_key == "[REDACTED]":
        raise RuntimeError("Set OPENAI_API_KEY before running this example")

    rag = MultimodalRAG(
        model="gpt-4o-mini",
        api_key=api_key,
        visual_embeddings=ColPaliEmbeddings(model="vidore/colpali-v1.3-hf"),
        retrieval_top_k=3,
    )
    await rag.add_async(path)
    print(await rag.ask("Which page contains the highlighted revenue total?"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/visual_rag.py <file.pdf|file.pptx>")
    asyncio.run(main(sys.argv[1]))
