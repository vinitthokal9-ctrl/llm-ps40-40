
"""PDF ingestion, local vector search, and grounded Gemini responses."""

from __future__ import annotations

import io
import os
from pathlib import Path
from dataclasses import dataclass

import pymupdf as fitz
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# Load .env from the project folder
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 5


class ManualError(Exception):
    """Expected, user-readable manual processing or answering error."""


@dataclass(frozen=True)
class ManualChunk:
    page: int
    text: str


class ManualAssistant:
    """A session-local manual index and grounded question-answering service."""

    def __init__(self, chunks: list[ManualChunk], page_count: int):
        if not chunks:
            raise ManualError(
                "No readable text was found in this PDF. "
                "It may be a scanned image PDF."
            )

        self.chunks = chunks
        self.page_count = page_count
        self.chunk_count = len(chunks)

        try:
            self._model = SentenceTransformer(EMBEDDING_MODEL)

            vectors = self._model.encode(
                [chunk.text for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vectors = np.asarray(vectors, dtype=np.float32)

            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)

        except Exception as exc:
            raise ManualError(
                f"Could not create text embeddings: {exc}"
            ) from exc

    @classmethod
    def from_pdf(cls, pdf_bytes: bytes) -> "ManualAssistant":
        if not pdf_bytes:
            raise ManualError("The uploaded PDF is empty.")

        try:
            with fitz.open(
                stream=io.BytesIO(pdf_bytes),
                filetype="pdf"
            ) as document:

                if document.needs_pass:
                    raise ManualError(
                        "This PDF is password-protected. "
                        "Please upload an unlocked copy."
                    )

                page_count = document.page_count
                chunks: list[ManualChunk] = []

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()

                    chunks.extend(
                        ManualChunk(page=page_number, text=part)
                        for part in cls._split_text(text)
                    )

        except ManualError:
            raise

        except Exception as exc:
            raise ManualError(
                f"That file could not be read as a valid PDF: {exc}"
            ) from exc

        if not chunks:
            raise ManualError(
                "No selectable text was found. This may be a scanned PDF; "
                "OCR is not included, so try a text-based PDF manual."
            )

        return cls(chunks, page_count)

    @staticmethod
    def _split_text(text: str) -> list[str]:
        words = text.split()

        if not words:
            return []

        step = CHUNK_WORDS - CHUNK_OVERLAP

        return [
            " ".join(words[start:start + CHUNK_WORDS])
            for start in range(0, len(words), step)
        ]

    def _retrieve(
        self, question: str
    ) -> list[tuple[ManualChunk, float]]:

        vector = self._model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores, indices = self._index.search(
            np.asarray(vector, dtype=np.float32),
            min(TOP_K, self.chunk_count),
        )

        return [
            (self.chunks[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def answer(
        self, question: str
    ) -> tuple[str, list[dict[str, str | int]]]:

        question = question.strip()

        if not question:
            raise ManualError(
                "Please enter a question about the manual."
            )

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ManualError(
                "Gemini API key is missing. "
                "Add GEMINI_API_KEY to your .env file and restart the app."
            )

        # Retrieve relevant manual sections
        try:
            matches = self._retrieve(question)

        except Exception as exc:
            raise ManualError(
                f"Could not search the manual: {exc}"
            ) from exc

        if not matches:
            return (
                "I could not find relevant information in the uploaded manual.",
                [],
            )

        # Prepare context for Gemini
        context = "\n\n".join(
            f"[Manual page {chunk.page}]\n{chunk.text}"
            for chunk, _score in matches
        )

        prompt = f"""You are a careful troubleshooting assistant for an appliance manual.

Answer using only the manual excerpts below.
Never use outside knowledge to fill gaps.

If the excerpts do not contain enough information, say clearly:
"I couldn't find that information in the uploaded manual."

Use concise numbered steps for instructions.
Quote relevant manufacturer warnings prominently.
Do not invent safety warnings.
Do not claim you performed an action.
Do not mention unsupported product details.

The UI will show source pages separately.
Do not make up page references in your response.

MANUAL EXCERPTS:
{context}

USER QUESTION:
{question}
"""

        # Generate answer using Gemini
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1
                ),
            )

            answer_text = (response.text or "").strip()

            if not answer_text:
                raise ManualError(
                    "Gemini returned an empty answer. "
                    "Please try asking another way."
                )

        except ManualError:
            raise

        except Exception as exc:
            print(
                f"\nGEMINI API ERROR: {type(exc).__name__}: {exc}\n"
            )

            raise ManualError(
                f"Gemini API Error: {type(exc).__name__}: {exc}"
            ) from exc

        # Prepare source page references
        seen: set[int] = set()
        sources: list[dict[str, str | int]] = []

        for chunk, _score in matches:
            if chunk.page not in seen:
                seen.add(chunk.page)

                preview = " ".join(chunk.text.split())

                sources.append({
                    "page": chunk.page,
                    "preview": (
                        preview[:220]
                        + ("…" if len(preview) > 220 else "")
                    ),
                })

        return answer_text, sources
