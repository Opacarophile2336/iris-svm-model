"""Translation Service for IrisAI Studio.

Orchestrates pluggable translation providers with privacy-first consent controls,
strict input validation, zero SQL persistence, and no sensitive text logging.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

from translation.providers.google_web import GoogleWebRpcProvider, GoogleWebRpcError
from translation.providers.mymemory import MyMemoryProvider, MyMemoryError

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "vi"}
DEFAULT_MAX_CHARS = 5000


class TranslationService:
    """Manages translation requests and provider fallbacks."""

    def __init__(
        self,
        google_provider: Optional[GoogleWebRpcProvider] = None,
        mymemory_provider: Optional[MyMemoryProvider] = None,
        max_chars: Optional[int] = None,
    ):
        self.google_provider = google_provider or GoogleWebRpcProvider()
        self.mymemory_provider = mymemory_provider or MyMemoryProvider()
        self.max_chars = max_chars or int(os.getenv("TRANSLATION_MAX_CHARS", DEFAULT_MAX_CHARS))

    async def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "vi",
        allow_mymemory: bool = False,
    ) -> Dict[str, Any]:
        """
        Translates text from source_lang to target_lang.

        Rules:
        - Validates input boundaries (non-empty, max_chars, supported langs).
        - Defaults to Google Web RPC (experimental).
        - If Google fails, NEVER calls MyMemory unless allow_mymemory is explicitly True.
        - Does NOT persist text to database, cache, or logs.
        """
        # 1. Validation
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Văn bản cần dịch không được để trống.",
            )

        cleaned_text = text.strip()
        if len(cleaned_text) > self.max_chars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Văn bản vượt quá giới hạn tối đa {self.max_chars} ký tự.",
            )

        s_lang = source_lang.strip().lower()
        t_lang = target_lang.strip().lower()

        if s_lang not in SUPPORTED_LANGUAGES or t_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hệ thống hiện chỉ hỗ trợ dịch song ngữ giữa tiếng Anh ('en') và tiếng Việt ('vi').",
            )

        if s_lang == t_lang:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ngôn ngữ nguồn và ngôn ngữ đích phải khác nhau.",
            )

        # 2. Primary Provider: Google Web RPC
        google_error: Optional[str] = None
        try:
            translated = await self.google_provider.translate(cleaned_text, s_lang, t_lang)
            logger.info(
                f"[Translate] Success via {self.google_provider.name} "
                f"({s_lang}->{t_lang}, len={len(cleaned_text)})"
            )
            return {
                "translated_text": translated,
                "source_lang": s_lang,
                "target_lang": t_lang,
                "provider": self.google_provider.name,
                "provider_display": self.google_provider.display_name,
                "character_count": len(cleaned_text),
                "fallback_used": False,
                "requires_consent": False,
            }
        except (GoogleWebRpcError, Exception) as e:
            google_error = str(e)
            logger.warning(
                f"[Translate] Google Web RPC failed ({s_lang}->{t_lang}, "
                f"len={len(cleaned_text)}): {type(e).__name__} - {google_error}"
            )

        # 3. Handle Fallback with User Consent Enforcement
        if not allow_mymemory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Dịch vụ Google Web RPC thử nghiệm tạm thời không phản hồi ({google_error}). "
                    "Bạn có thể kích hoạt tùy chọn dự phòng MyMemory (Lưu ý: bản dịch sẽ được đóng góp "
                    "vào bộ nhớ dịch thuật cộng đồng công khai) hoặc thử lại sau."
                ),
            )

        # User has explicitly consented to MyMemory fallback
        logger.info(
            f"[Translate] Attempting MyMemory fallback with user consent "
            f"({s_lang}->{t_lang}, len={len(cleaned_text)})"
        )
        try:
            translated = await self.mymemory_provider.translate(cleaned_text, s_lang, t_lang)
            logger.info(
                f"[Translate] Success via {self.mymemory_provider.name} "
                f"({s_lang}->{t_lang}, len={len(cleaned_text)})"
            )
            return {
                "translated_text": translated,
                "source_lang": s_lang,
                "target_lang": t_lang,
                "provider": self.mymemory_provider.name,
                "provider_display": self.mymemory_provider.display_name,
                "character_count": len(cleaned_text),
                "fallback_used": True,
                "requires_consent": True,
                "notice": (
                    "Bản dịch được thực hiện qua MyMemory do bạn đã đồng ý kích hoạt dự phòng cộng đồng."
                ),
            }
        except (MyMemoryError, Exception) as e:
            logger.error(
                f"[Translate] MyMemory fallback also failed ({s_lang}->{t_lang}, "
                f"len={len(cleaned_text)}): {type(e).__name__} - {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Cả hai dịch vụ dịch thuật đều không khả dụng. "
                    f"Google lỗi: {google_error}. MyMemory lỗi: {str(e)}."
                ),
            )


# Global singleton instance
translation_service = TranslationService()
