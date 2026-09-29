"""Abstract base class for translation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTranslationProvider(ABC):
    """Abstract interface for all pluggable translation providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine identifier of the provider (e.g. 'google_web_rpc', 'mymemory')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name of the provider."""
        pass

    @property
    def is_official(self) -> bool:
        """Whether this provider is an officially supported API with SLA."""
        return False

    @property
    def requires_public_memory_consent(self) -> bool:
        """Whether this provider contributes text to a public collaborative memory."""
        return False

    @abstractmethod
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source_lang to target_lang.

        Args:
            text: Non-empty string to translate.
            source_lang: 2-letter ISO language code (e.g. 'en', 'vi').
            target_lang: 2-letter ISO language code (e.g. 'vi', 'en').

        Returns:
            Translated string preserving structure, punctuation, and casing.

        Raises:
            Exception: If translation fails, times out, or encounters rate-limiting.
        """
        pass
