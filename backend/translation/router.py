"""FastAPI Router for Translation Studio."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends

from auth.dependencies import require_user
from translation.service import translation_service

router = APIRouter()


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Văn bản cần dịch")
    source_lang: str = Field("en", description="Mã ngôn ngữ nguồn ('en' hoặc 'vi')")
    target_lang: str = Field("vi", description="Mã ngôn ngữ đích ('vi' hoặc 'en')")
    allow_mymemory: bool = Field(
        False,
        description="Đồng ý kích hoạt dự phòng MyMemory (lưu vào bộ nhớ công cộng) nếu Google lỗi",
    )


class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str
    provider_display: str
    character_count: int
    fallback_used: bool = False
    requires_consent: bool = False
    notice: Optional[str] = None


@router.post("/translate", response_model=TranslationResponse)
async def translate_text_endpoint(
    req: TranslationRequest,
    current_user: dict = Depends(require_user),
):
    """
    Translates text between English and Vietnamese.
    Requires user authentication (JWT token).
    """
    result = await translation_service.translate(
        text=req.text,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        allow_mymemory=req.allow_mymemory,
    )
    return result
