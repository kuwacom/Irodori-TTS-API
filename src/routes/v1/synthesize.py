"""音声合成エンドポイント"""

from __future__ import annotations

import asyncio
import base64
import io

import soundfile
from fastapi import APIRouter, Request, Response
from irodori_tts.inference_runtime import SamplingRequest

from src.configs.env import env
from src.lib.api_error import ErrorCode, api_error
from src.middleware.logger import REQUEST_ID_KEY
from src.schemas.synthesize import (
    AudioItem,
    ConditioningInfo,
    ModelInfo,
    SynthesizeRequest,
    SynthesizeResponse,
)
from src.services import speaker_store
from src.services.logger import logger
from src.services.tts_runtime import get_runtime

synthesize_router = APIRouter()


def _determine_conditioning_mode(speaker_id: str | None, caption: str | None) -> str:
    """speakerId / caption の有無からモード名を決定する"""
    has_speaker = speaker_id is not None
    has_caption = caption is not None
    if has_speaker and has_caption:
        return "speaker_and_caption"
    if has_speaker:
        return "speaker"
    if has_caption:
        return "voice_design"
    return "text_only"


def _save_wav_bytes(audio_tensor, sample_rate, format_name):
    """テンソルをWAVバイト列に変換する

    torchaudio.save は BytesIO を渡すと torchcodec が
    AVFormatContext を確保できずクラッシュするため
    soundfile.write で直接書き込む
    """
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    # (C, T) → (T, C) に転位してnumpy化
    waveform_np = audio_tensor.numpy().T
    fmt = format_name.upper()
    subtype = "FLOAT" if format_name == "wav" else "PCM_16"
    buf = io.BytesIO()
    soundfile.write(buf, waveform_np, sample_rate, format=fmt, subtype=subtype)
    return buf.getvalue()


def _run_synthesize(runtime, sampling_req):
    """スレッドプール上で同期的に推論を実行する"""
    return runtime.synthesize(sampling_req)


@synthesize_router.post("", response_model=None)
async def synthesize(
    req: SynthesizeRequest,
    request: Request,
):
    """音声合成を実行する"""
    # ミドルウェアで生成したリクエストIDを取得し、レスポンスの id に一貫性を持たせる
    request_id = getattr(request.state, REQUEST_ID_KEY, "")

    # 制限チェック
    if (
        req.duration.seconds is not None
        and req.duration.seconds > env.max_generate_seconds
    ):
        raise api_error(
            ErrorCode.VALIDATION_ERROR,
            message=f"seconds exceeds max {env.max_generate_seconds}",
        )
    if req.sampling.numCandidates > env.max_num_candidates:
        raise api_error(
            ErrorCode.VALIDATION_ERROR,
            message=(f"numCandidates exceeds max {env.max_num_candidates}"),
        )

    # 話者 lookup
    ref_latent_path: str | None = None
    speaker_data: dict | None = None
    if req.speakerId is not None:
        speaker_data = speaker_store.get_speaker(req.speakerId)
        if speaker_data is None:
            raise api_error(
                ErrorCode.NOT_FOUND,
                message=f"Speaker not found: {req.speakerId}",
            )
        ref_latent_path = speaker_store.latent_path_for(req.speakerId)

    # 起動時に初期化済みのランタイムを取得
    runtime = get_runtime()
    checkpoint = env.default_model

    # SamplingRequest 構築
    no_ref = req.speakerId is None
    norm_db = speaker_data.get("normalizeDb", -16.0) if speaker_data else None
    ensure_max = speaker_data.get("ensureMax", True) if speaker_data else True
    max_ref = (
        speaker_data.get("maxRefSeconds", env.max_ref_seconds)
        if speaker_data
        else env.max_ref_seconds
    )

    sampling_req = SamplingRequest(
        text=req.text,
        caption=req.caption,
        ref_latent=ref_latent_path,
        no_ref=no_ref,
        ref_normalize_db=norm_db,
        ref_ensure_max=ensure_max,
        max_ref_seconds=max_ref,
        max_seconds=env.max_generate_seconds,
        num_candidates=req.sampling.numCandidates,
        # nullならモデル内蔵のduration predictorがテキスト長から自動予測
        seconds=req.duration.seconds,
        seed=req.sampling.seed,
        sampling_preset=req.sampling.preset,
        num_steps=req.sampling.numSteps,
        cfg_scale_text=req.guidance.cfgScaleText,
        cfg_scale_caption=req.guidance.cfgScaleCaption,
        cfg_scale_speaker=req.guidance.cfgScaleSpeaker,
        cfg_guidance_mode=req.guidance.mode,
        cfg_scale=req.guidance.cfgScale,
        cfg_min_t=req.guidance.cfgMinT,
        cfg_max_t=req.guidance.cfgMaxT,
        truncation_factor=req.truncation.factor,
        rescale_k=req.truncation.rescaleK,
        rescale_sigma=req.truncation.rescaleSigma,
        context_kv_cache=req.kvCache.contextKvCache,
        speaker_kv_scale=req.kvCache.speakerKvScale,
        speaker_kv_min_t=req.kvCache.speakerKvMinT,
        speaker_kv_max_layers=req.kvCache.speakerKvMaxLayers,
        speaker_uncond_mode=req.kvCache.speakerUncondMode,
        t_schedule_mode=req.sampling.tScheduleMode,
        sway_coeff=req.sampling.swayCoeff,
        trim_tail=req.tailTrim.trimTail,
        tail_window_size=req.tailTrim.tailWindowSize,
        tail_std_threshold=req.tailTrim.tailStdThreshold,
        tail_mean_threshold=req.tailTrim.tailMeanThreshold,
        lora_adapter=req.model.loraAdapter,
        decode_mode=req.decode.mode,
        duration_scale=req.duration.durationScale,
        min_seconds=req.duration.minSeconds,
        max_text_len=req.tokenLimits.maxTextLen,
        max_caption_len=req.tokenLimits.maxCaptionLen,
    )

    # 推論をスレッドプールで実行しイベントループを解放する
    try:
        result = await asyncio.to_thread(_run_synthesize, runtime, sampling_req)
    except Exception:
        logger.exception("Synthesis failed")
        raise api_error(
            ErrorCode.INTERNAL_SERVER_ERROR,
            message="Synthesis failed",
        ) from None

    # lastUsedAt 更新
    if req.speakerId is not None:
        speaker_store.update_last_used(req.speakerId)

    # buffer モード: 音声を直接 Response で返す
    if req.output.mode == "buffer":
        audio_tensor = result.audios[0].detach().cpu()
        wav_bytes = _save_wav_bytes(audio_tensor, result.sample_rate, req.output.format)
        return Response(
            content=wav_bytes,
            media_type=(
                "audio/wav"
                if req.output.format == "wav"
                else f"audio/{req.output.format}"
            ),
            headers={
                "X-Sample-Rate": str(result.sample_rate),
                "X-Seed": str(result.used_seed),
                "Content-Disposition": "inline",
            },
        )

    # inline モード: 音声を base64 でエンコードして返す
    audios: list[AudioItem] = []
    for i, audio_tensor in enumerate(result.audios):
        audio_cpu = audio_tensor.detach().cpu()
        wav_bytes = _save_wav_bytes(audio_cpu, result.sample_rate, "wav")
        duration = audio_cpu.shape[-1] / result.sample_rate

        audios.append(
            AudioItem(
                index=i,
                contents=base64.b64encode(wav_bytes).decode("ascii"),
                mimeType="audio/wav",
                sampleRate=result.sample_rate,
                duration=round(duration, 2),
            )
        )

    # タイミング情報を整形
    timings: dict = {"totalToDecodeMs": round(result.total_to_decode * 1000, 1)}
    for name, elapsed in result.stage_timings:
        timings[f"{name}Ms"] = round(elapsed * 1000, 1)

    mode = _determine_conditioning_mode(req.speakerId, req.caption)

    return SynthesizeResponse(
        id=request_id,
        status="succeeded",
        model=ModelInfo(checkpoint=checkpoint),
        conditioning=ConditioningInfo(
            speakerId=req.speakerId,
            caption=req.caption,
            mode=mode,
        ),
        audios=audios,
        seed=result.used_seed,
        timings=timings,
        messages=result.messages,
    )
