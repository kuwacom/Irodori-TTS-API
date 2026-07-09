"""話者登録・参照に関する Pydantic モデル"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SpeakerMetadataIn(BaseModel):
    """POST /v1/speakers の metadata フィールド"""

    name: str = Field(default="", description="話者名")
    description: str = Field(default="", description="説明")
    caption: str | None = Field(
        default=None,
        description="合成時のデフォルトキャプション（スタイル指示）",
    )
    maxRefSeconds: float = Field(default=30.0, description="参照音声の最大秒数")
    normalizeDb: float = Field(default=-16.0, description="正規化目標 dB")
    ensureMax: bool = Field(
        default=True,
        description="正規化後にピーククリップ",
    )


class SpeakerCreateResponse(BaseModel):
    """POST /v1/speakers のレスポンス"""

    speakerId: str
    name: str = ""
    status: str
    sha256: str = ""
    createdAt: str = ""


class SpeakerSummary(BaseModel):
    """GET /v1/speakers 一覧の各要素"""

    speakerId: str
    name: str = ""
    createdAt: str
    lastUsedAt: str | None = None


class SpeakerDetail(BaseModel):
    """GET /v1/speakers/{speakerId} のレスポンス"""

    speakerId: str
    name: str = ""
    description: str = ""
    caption: str | None = None
    sha256: str = ""
    maxRefSeconds: float = 30.0
    normalizeDb: float = -16.0
    ensureMax: bool = True
    codecRepo: str = ""
    createdAt: str
    updatedAt: str
    lastUsedAt: str | None = None


class SpeakerDeleteResponse(BaseModel):
    """DELETE /v1/speakers/{speakerId} のレスポンス"""

    speakerId: str
    deleted: bool = True


def speaker_summary_from_dict(speaker_id: str, data: dict[str, Any]) -> SpeakerSummary:
    """内部 dict から一覧要素を構築する"""
    return SpeakerSummary(
        speakerId=speaker_id,
        name=data.get("name", ""),
        createdAt=data.get("createdAt", ""),
        lastUsedAt=data.get("lastUsedAt"),
    )


def speaker_detail_from_dict(speaker_id: str, data: dict[str, Any]) -> SpeakerDetail:
    """内部 dict から詳細要素を構築する"""
    return SpeakerDetail(
        speakerId=speaker_id,
        name=data.get("name", ""),
        description=data.get("description", ""),
        caption=data.get("caption"),
        sha256=data.get("sha256", ""),
        maxRefSeconds=data.get("maxRefSeconds", 30.0),
        normalizeDb=data.get("normalizeDb", -16.0),
        ensureMax=data.get("ensureMax", True),
        codecRepo=data.get("codecRepo", ""),
        createdAt=data.get("createdAt", ""),
        updatedAt=data.get("updatedAt", ""),
        lastUsedAt=data.get("lastUsedAt"),
    )
