"""Google Web RPC (client=tw-ob) Translation Provider.

NOTE: This is an UNOFFICIAL, undocumented experimental provider.
It does not offer an SLA and may return HTTP 429 when rate-limited.
"""
from __future__ import annotations

import logging
from typing import Optional
import httpx

from translation.providers.base import BaseTranslationProvider

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class GoogleWebRpcError(RuntimeError):
    """Raised when Google Web RPC fails or is rate-limited."""
    pass


class GoogleWebRpcProvider(BaseTranslationProvider):
    """Unofficial Google Web RPC translation provider."""

    def __init__(self, timeout: float = 6.0, user_agent: Optional[str] = None):
        self._timeout = timeout
        self._user_agent = user_agent or DEFAULT_USER_AGENT

    @property
    def name(self) -> str:
        return "google_web_rpc"

    @property
    def display_name(self) -> str:
        return "Google Web RPC (Thử nghiệm)"

    @property
    def is_official(self) -> bool:
        return False

    @property
    def requires_public_memory_consent(self) -> bool:
        return False

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text via Google Web RPC.
        Raises GoogleWebRpcError if status is not 200 (e.g. HTTP 429) or request times out.
        """
        endpoint = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "tw-ob",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "*/*",
        }

        try:
            async with httpx.AsyncClient(headers=headers, timeout=self._timeout) as client:
                response = await client.get(endpoint, params=params)

            if response.status_code == 429:
                raise GoogleWebRpcError(
                    "Google Web RPC tạm thời bị giới hạn tần suất truy vấn (HTTP 429 Too Many Requests)."
                )

            if response.status_code != 200:
                raise GoogleWebRpcError(
                    f"Google Web RPC trả về HTTP {response.status_code}."
                )

            data = response.json()
            # Google RPC response format: [[[translated_part, original_part, ...], ...], ...]
            if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
                raise GoogleWebRpcError("Google Web RPC trả về định dạng dữ liệu không hợp lệ.")

            translated_segments = []
            for item in data[0]:
                if isinstance(item, list) and len(item) > 0 and item[0]:
                    translated_segments.append(str(item[0]))

            result = "".join(translated_segments)
            if not result:
                raise GoogleWebRpcError("Google Web RPC không trả về nội dung dịch.")

            return result

        except GoogleWebRpcError:
            raise
        except httpx.TimeoutException as e:
            raise GoogleWebRpcError(f"Google Web RPC hết thời gian phản hồi ({self._timeout}s).") from e
        except Exception as e:
            raise GoogleWebRpcError(f"Lỗi kết nối tới Google Web RPC: {str(e)}") from e
