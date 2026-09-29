"""MyMemory Public Collaborative Translation Provider.

NOTE: Free tier queries to MyMemory are stored in a public collaborative
translation memory. This provider requires explicit user consent before use.
"""
from __future__ import annotations

import html
import logging
import os
import re
from typing import Optional
import httpx

from translation.providers.base import BaseTranslationProvider

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
MAX_CHUNK_CHARS = 450


class MyMemoryError(RuntimeError):
    """Raised when MyMemory API fails or is rate-limited."""
    pass


def split_text_into_chunks(text: str, max_len: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Splits text into chunks <= max_len while preserving newlines and sentence boundaries.
    """
    if len(text) <= max_len:
        return [text]

    paragraphs = text.split("\n")
    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) <= max_len:
            test_len = current_len + len(para) + (1 if current_chunk else 0)
            if test_len <= max_len:
                current_chunk.append(para)
                current_len = test_len
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
        else:
            # Paragraph exceeds max_len, split by sentence endings
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            sentences = re.split(r"([.?!;]+(?:\s+|$))", para)
            combined = []
            for i in range(0, len(sentences), 2):
                s = sentences[i]
                delim = sentences[i + 1] if i + 1 < len(sentences) else ""
                if s or delim:
                    combined.append(s + delim)

            for s in combined:
                if not s:
                    continue
                if len(s) > max_len:
                    # Rare: single sentence longer than max_len, chunk by words
                    words = s.split(" ")
                    w_chunk: list[str] = []
                    w_len = 0
                    for w in words:
                        if w_len + len(w) + 1 <= max_len:
                            w_chunk.append(w)
                            w_len += len(w) + 1
                        else:
                            if w_chunk:
                                chunks.append(" ".join(w_chunk))
                            w_chunk = [w]
                            w_len = len(w)
                    if w_chunk:
                        chunks.append(" ".join(w_chunk))
                else:
                    if current_len + len(s) <= max_len:
                        current_chunk.append(s)
                        current_len += len(s)
                    else:
                        if current_chunk:
                            chunks.append("".join(current_chunk))
                        current_chunk = [s]
                        current_len = len(s)

            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [c for c in chunks if c]


class MyMemoryProvider(BaseTranslationProvider):
    """MyMemory API translation provider."""

    def __init__(
        self,
        timeout: float = 6.0,
        email: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self._timeout = timeout
        self._email = email or os.getenv("TRANSLATION_MYMEMORY_EMAIL")
        self._user_agent = user_agent or DEFAULT_USER_AGENT

    @property
    def name(self) -> str:
        return "mymemory"

    @property
    def display_name(self) -> str:
        return "MyMemory API (Dự phòng)"

    @property
    def is_official(self) -> bool:
        return False

    @property
    def requires_public_memory_consent(self) -> bool:
        return True

    async def _translate_single_chunk(
        self, client: httpx.AsyncClient, chunk: str, source_lang: str, target_lang: str
    ) -> str:
        endpoint = "https://api.mymemory.translated.net/get"
        params = {
            "q": chunk,
            "langpair": f"{source_lang}|{target_lang}",
        }
        if self._email:
            params["de"] = self._email

        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }

        try:
            response = await client.get(endpoint, params=params, headers=headers)
        except httpx.TimeoutException as e:
            raise MyMemoryError(f"MyMemory hết thời gian phản hồi ({self._timeout}s).") from e
        except Exception as e:
            raise MyMemoryError(f"Lỗi kết nối tới MyMemory: {str(e)}") from e

        if response.status_code == 429:
            raise MyMemoryError("MyMemory đã hết hạn mức truy vấn miễn phí trong ngày (HTTP 429).")

        if response.status_code != 200:
            raise MyMemoryError(f"MyMemory trả về HTTP {response.status_code}.")

        data = response.json()
        status = data.get("responseStatus")
        if status != 200:
            details = data.get("responseDetails", "Không rõ nguyên nhân")
            raise MyMemoryError(f"MyMemory trả về trạng thái lỗi {status}: {details}")

        raw_text = data.get("responseData", {}).get("translatedText")
        if not raw_text:
            raise MyMemoryError("MyMemory không trả về nội dung dịch.")

        # MyMemory returns HTML-encoded entities like &#39; or &quot;
        return html.unescape(raw_text)

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text via MyMemory API with automatic sentence chunking.
        """
        chunks = split_text_into_chunks(text, max_len=MAX_CHUNK_CHARS)
        if not chunks:
            return ""

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            translated_chunks = []
            for chunk in chunks:
                trans = await self._translate_single_chunk(client, chunk, source_lang, target_lang)
                translated_chunks.append(trans)

        # Recombine preserving original sentence spacing
        # If original text has newlines between paragraphs, reassemble appropriately
        if len(translated_chunks) == 1:
            return translated_chunks[0]

        # Join chunks with space or newline depending on chunk endings
        result_parts = []
        for i, tc in enumerate(translated_chunks):
            result_parts.append(tc)
            if i < len(translated_chunks) - 1:
                if not tc.endswith("\n") and not tc.endswith(" "):
                    result_parts.append(" ")
        return "".join(result_parts)
