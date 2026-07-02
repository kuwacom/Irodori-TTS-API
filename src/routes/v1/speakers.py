"""話者 CRUD エンドポイント"""

from __future__ import annotations

import asyncio
import io

import torchaudio
from fastapi import APIRouter, File, Form, UploadFile

from src.configs.env import env
from src.lib.api_error import ErrorCode, api_error
from src.schemas.speaker import (
    SpeakerCreateResponse,
    SpeakerDeleteResponse,
    SpeakerDetail,
    SpeakerMetadataIn,
    SpeakerSummary,
    speaker_detail_from_dict,
    speaker_summary_from_dict,
)
from src.services import speaker_store
from src.services.logger import logger
from src.services.tts_runtime import get_runtime

speakers_router = APIRouter()

_file_default = File(...)
_form_default = Form(...)


def _encode_audio(
    audio_bytes: bytes,
    max_ref_seconds: float,
    normalize_db: float,
    ensure_max: bool,
):
    """スレッドプールで音声読み込み → codec encode を実行する"""
    audio_buf = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(audio_buf)

    # encode_waveform 向けに (B, C, T) に整形
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    # trim: maxRefSeconds 秒以内にトリム
    max_samples = int(max_ref_seconds * sample_rate)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    runtime = get_runtime()
    codec = runtime.codec
    latent = codec.encode_waveform(
        waveform,
        sample_rate,
        normalize_db=normalize_db,
        ensure_max=ensure_max,
    )
    return latent


@speakers_router.post("", response_model=SpeakerCreateResponse)
async def create_speaker(
    audio: UploadFile = _file_default,
    metadata: str = _form_default,
) -> SpeakerCreateResponse:
    """話者を登録する"""
    # メタデータのパース
    parsed = SpeakerMetadataIn.model_validate_json(metadata)

    audio_bytes = await audio.read()
    if len(audio_bytes) > env.max_request_body_size:
        raise api_error(ErrorCode.VALIDATION_ERROR, message="Audio file too large")

    audio_sha256 = speaker_store.compute_sha256(audio_bytes)

    # 重複チェック
    existing_id = speaker_store.find_by_sha256(
        audio_sha256, parsed.normalizeDb, parsed.ensureMax
    )
    if existing_id is not None:
        existing = speaker_store.get_speaker(existing_id)
        return SpeakerCreateResponse(
            speakerId=existing_id,
            name=existing.get("name", "") if existing else "",
            status="already exists",
            sha256=audio_sha256,
        )

    # 重い処理をスレッドプールで実行しイベントループを解放する
    try:
        latent = await asyncio.to_thread(
            _encode_audio,
            audio_bytes,
            parsed.maxRefSeconds,
            parsed.normalizeDb,
            parsed.ensureMax,
        )
    except Exception:
        logger.exception("Failed to encode waveform")
        raise api_error(
            ErrorCode.VALIDATION_ERROR,
            message="Unsupported audio format or encode failure",
        ) from None

    # 話者登録
    speaker_id = speaker_store.register_speaker(
        audio_sha256=audio_sha256,
        name=parsed.name,
        description=parsed.description,
        max_ref_seconds=parsed.maxRefSeconds,
        normalize_db=parsed.normalizeDb,
        ensure_max=parsed.ensureMax,
        codec_repo=env.codec_repo,
        latent_tensor=latent,
    )

    created = speaker_store.get_speaker(speaker_id)
    return SpeakerCreateResponse(
        speakerId=speaker_id,
        name=parsed.name,
        status="created",
        sha256=audio_sha256,
        createdAt=created.get("createdAt", "") if created else "",
    )


@speakers_router.get("", response_model=list[SpeakerSummary])
async def list_speakers() -> list[SpeakerSummary]:
    """話者一覧を返す"""
    speakers = speaker_store.all_speakers()
    return [speaker_summary_from_dict(sid, data) for sid, data in speakers.items()]


@speakers_router.get("/{speaker_id}", response_model=SpeakerDetail)
async def get_speaker_detail(speaker_id: str) -> SpeakerDetail:
    """話者詳細を返す"""
    data = speaker_store.get_speaker(speaker_id)
    if data is None:
        raise api_error(
            ErrorCode.NOT_FOUND,
            message=f"Speaker not found: {speaker_id}",
        )
    return speaker_detail_from_dict(speaker_id, data)


@speakers_router.delete("/{speaker_id}", response_model=SpeakerDeleteResponse)
async def delete_speaker(speaker_id: str) -> SpeakerDeleteResponse:
    """話者を削除する"""
    deleted = speaker_store.delete_speaker(speaker_id)
    if not deleted:
        raise api_error(
            ErrorCode.NOT_FOUND,
            message=f"Speaker not found: {speaker_id}",
        )
    return SpeakerDeleteResponse(speakerId=speaker_id)
