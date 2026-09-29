"""Comprehensive Test Suite for IrisAI Studio Translation Studio.

Covers:
1. Unit tests: Text chunking and sentence boundary preservation.
2. Unit tests: Input validation (empty, whitespace, length limit, invalid languages).
3. Unit/Mock tests: Privacy-first provider dispatch and consent enforcement.
4. API & Auth tests: JWT bearer token requirement and endpoint responses.
5. Real integration tests: Two-way English <-> Vietnamese translations with actual external services.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.service import create_access_token
from main import app
from translation.providers.google_web import GoogleWebRpcProvider, GoogleWebRpcError
from translation.providers.mymemory import MyMemoryProvider, MyMemoryError, split_text_into_chunks
from translation.service import TranslationService

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def auth_headers():
    token = create_access_token("test_user_translate", "USER")
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# 1. UNIT TESTS: Text Chunking
# ==============================================================================

class TestTextChunking:
    """Verifies the sentence-aware text chunking algorithm for providers with length limits."""

    def test_short_text_single_chunk(self):
        text = "Iris versicolor is a species of iris."
        chunks = split_text_into_chunks(text, max_len=450)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_splitting_preserves_sentences(self):
        text = (
            "Iris setosa has small petals.\n"
            "Iris versicolor has intermediate characteristics.\n"
            "Iris virginica is the largest of the three species."
        )
        chunks = split_text_into_chunks(text, max_len=60)
        assert len(chunks) >= 2
        # Verify all original sentences are present across chunks
        joined = " ".join(chunks)
        assert "Iris setosa" in joined
        assert "Iris versicolor" in joined
        assert "Iris virginica" in joined

    def test_long_single_sentence_splits_on_words(self):
        words = ["word" + str(i) for i in range(50)]
        long_sentence = " ".join(words)
        chunks = split_text_into_chunks(long_sentence, max_len=100)
        for c in chunks:
            assert len(c) <= 100
        # All words preserved
        reconstructed = " ".join(chunks)
        for w in ["word0", "word25", "word49"]:
            assert w in reconstructed


# ==============================================================================
# 2. UNIT TESTS: Input Validation
# ==============================================================================

class TestInputValidation:
    """Verifies input validation without making network requests."""

    @pytest.mark.asyncio
    async def test_empty_text_raises_400(self):
        service = TranslationService()
        with pytest.raises(HTTPException) as exc_info:
            await service.translate("", "en", "vi")
        assert exc_info.value.status_code == 400
        assert "không được để trống" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_400(self):
        service = TranslationService()
        with pytest.raises(HTTPException) as exc_info:
            await service.translate("   \n\t  ", "en", "vi")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_text_exceeding_max_chars_raises_400(self):
        service = TranslationService(max_chars=100)
        long_text = "a" * 105
        with pytest.raises(HTTPException) as exc_info:
            await service.translate(long_text, "en", "vi")
        assert exc_info.value.status_code == 400
        assert "giới hạn tối đa" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unsupported_language_raises_400(self):
        service = TranslationService()
        with pytest.raises(HTTPException) as exc_info:
            await service.translate("Hello", "fr", "vi")
        assert exc_info.value.status_code == 400
        assert "chỉ hỗ trợ" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_same_source_and_target_raises_400(self):
        service = TranslationService()
        with pytest.raises(HTTPException) as exc_info:
            await service.translate("Hello", "en", "en")
        assert exc_info.value.status_code == 400
        assert "phải khác nhau" in exc_info.value.detail


# ==============================================================================
# 3. MOCK TESTS: Privacy Enforcement & Consent-based Fallback
# ==============================================================================

class TestProviderOrchestrationAndPrivacy:
    """Verifies that MyMemory is NEVER called without explicit user consent."""

    @pytest.mark.asyncio
    async def test_google_success_returns_google_result(self):
        mock_google = AsyncMock()
        mock_google.name = "google_web_rpc"
        mock_google.display_name = "Google Web RPC (Thử nghiệm)"
        mock_google.translate.return_value = "Hoa diên vĩ rất đẹp."

        mock_mymemory = AsyncMock()

        service = TranslationService(google_provider=mock_google, mymemory_provider=mock_mymemory)
        res = await service.translate("The iris flower is beautiful.", "en", "vi", allow_mymemory=False)

        assert res["translated_text"] == "Hoa diên vĩ rất đẹp."
        assert res["provider"] == "google_web_rpc"
        assert res["fallback_used"] is False
        mock_mymemory.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_google_failure_without_consent_blocks_mymemory(self):
        """When Google fails and allow_mymemory=False, MUST NOT call MyMemory."""
        mock_google = AsyncMock()
        mock_google.name = "google_web_rpc"
        mock_google.translate.side_effect = GoogleWebRpcError("HTTP 429 Rate limited")

        mock_mymemory = AsyncMock()

        service = TranslationService(google_provider=mock_google, mymemory_provider=mock_mymemory)

        with pytest.raises(HTTPException) as exc_info:
            await service.translate("Test text", "en", "vi", allow_mymemory=False)

        assert exc_info.value.status_code == 503
        assert "Google Web RPC" in exc_info.value.detail
        # MyMemory was NEVER contacted
        mock_mymemory.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_google_failure_with_consent_triggers_mymemory(self):
        """When Google fails and allow_mymemory=True, calls MyMemory with fallback flag."""
        mock_google = AsyncMock()
        mock_google.name = "google_web_rpc"
        mock_google.translate.side_effect = GoogleWebRpcError("HTTP 429 Rate limited")

        mock_mymemory = AsyncMock()
        mock_mymemory.name = "mymemory"
        mock_mymemory.display_name = "MyMemory API (Dự phòng)"
        mock_mymemory.translate.return_value = "Hoa diên vĩ là một loài thực vật."

        service = TranslationService(google_provider=mock_google, mymemory_provider=mock_mymemory)
        res = await service.translate("Iris is a plant species.", "en", "vi", allow_mymemory=True)

        assert res["translated_text"] == "Hoa diên vĩ là một loài thực vật."
        assert res["provider"] == "mymemory"
        assert res["fallback_used"] is True
        assert res["requires_consent"] is True
        mock_mymemory.translate.assert_called_once()

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises_503(self):
        mock_google = AsyncMock()
        mock_google.name = "google_web_rpc"
        mock_google.translate.side_effect = GoogleWebRpcError("Network error")

        mock_mymemory = AsyncMock()
        mock_mymemory.name = "mymemory"
        mock_mymemory.translate.side_effect = MyMemoryError("Service down")

        service = TranslationService(google_provider=mock_google, mymemory_provider=mock_mymemory)

        with pytest.raises(HTTPException) as exc_info:
            await service.translate("Test text", "en", "vi", allow_mymemory=True)

        assert exc_info.value.status_code == 503
        assert "Cả hai dịch vụ dịch thuật đều không khả dụng" in exc_info.value.detail


# ==============================================================================
# 4. API & AUTHENTICATION TESTS
# ==============================================================================

class TestTranslationAPI:
    """Verifies FastAPI router behavior, authentication, and HTTP status codes."""

    def test_unauthenticated_request_blocked(self):
        resp = client.post(
            "/translation/translate",
            json={"text": "Hello world", "source_lang": "en", "target_lang": "vi"},
        )
        assert resp.status_code in (401, 403)

    def test_invalid_token_blocked(self):
        resp = client.post(
            "/translation/translate",
            headers={"Authorization": "Bearer invalid_token_xyz"},
            json={"text": "Hello world", "source_lang": "en", "target_lang": "vi"},
        )
        assert resp.status_code == 401

    def test_validation_empty_text_returns_400(self, auth_headers):
        resp = client.post(
            "/translation/translate",
            headers=auth_headers,
            json={"text": "   ", "source_lang": "en", "target_lang": "vi"},
        )
        assert resp.status_code == 400
        assert "không được để trống" in resp.json().get("detail", "")

    def test_validation_invalid_language_returns_400(self, auth_headers):
        resp = client.post(
            "/translation/translate",
            headers=auth_headers,
            json={"text": "Hello", "source_lang": "ja", "target_lang": "vi"},
        )
        assert resp.status_code == 400

    def test_spa_navigation_serves_html_for_translate(self):
        """GET /translate must return 200 index.html via SPA fallback."""
        resp = client.get("/translate")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ==============================================================================
# 5. REAL INTEGRATION TESTS: Live Network Calls
# ==============================================================================

class TestLiveRealTranslation:
    """
    Live integration tests calling external translation services.
    Distinguished clearly from mock tests.
    """

    @pytest.mark.asyncio
    async def test_live_mymemory_en_to_vi_real_translation(self):
        """Live test: MyMemory API translating English botanical sentence to Vietnamese."""
        provider = MyMemoryProvider(timeout=8.0)
        source = "The Iris flower has three recognized species in the dataset."
        result = await provider.translate(source, "en", "vi")

        assert isinstance(result, str)
        assert len(result) > 5
        # Verify semantic content (e.g. 'Iris' and 'loài' or 'hoa')
        lower_res = result.lower()
        assert "iris" in lower_res or "hoa" in lower_res
        assert any(k in lower_res for k in ["loài", "loai", "ba"])

    @pytest.mark.asyncio
    async def test_live_mymemory_vi_to_en_real_translation(self):
        """Live test: MyMemory API translating Vietnamese sentence with diacritics to English."""
        provider = MyMemoryProvider(timeout=8.0)
        source = "Mô hình học máy phân loại hoa diên vĩ với độ chính xác cao."
        result = await provider.translate(source, "vi", "en")

        assert isinstance(result, str)
        assert len(result) > 5
        lower_res = result.lower()
        # Verify semantic content
        assert any(k in lower_res for k in ["machine learning", "model", "iris", "accura"])

    @pytest.mark.asyncio
    async def test_live_google_web_rpc_or_graceful_rate_limit(self):
        """
        Live test: Google Web RPC.
        Verifies that it either succeeds (HTTP 200) or raises a clean GoogleWebRpcError
        (HTTP 429 rate limit) without crashing the service or leaking unhandled exceptions.
        """
        provider = GoogleWebRpcProvider(timeout=6.0)
        try:
            result = await provider.translate("Hello world", "en", "vi")
            assert isinstance(result, str)
            assert len(result) > 0
        except GoogleWebRpcError as e:
            # Expected if IP is temporarily rate-limited
            assert "HTTP 429" in str(e) or "Google Web RPC" in str(e)
