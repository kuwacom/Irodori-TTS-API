"""音声合成エンドポイント"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import soundfile
from fastapi import APIRouter, Request, Response
from irodori_tts.inference_runtime import (
    LongTextSamplingRequest,
    SamplingRequest,
)

from src.configs.env import env
from src.lib.api_error import ErrorCode, api_error
from src.middleware.logger import REQUEST_ID_KEY
from src.schemas.synthesize import (
    AudioItem,
    ConditioningInfo,
    ModelInfo,
    OutputParams,
    SamplingParams,
    SegmentInfo,
    SynthesizeQueryParams,
    SynthesizeRequest,
    SynthesizeResponse,
)
from src.services import speaker_store
from src.services.logger import logger
from src.services.tts_runtime import get_runtime

synthesize_router = APIRouter()


def _determine_conditioning_mode(
    speaker_id: str | None, caption: str | None
) -> str:
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
    # (C, T) -> (T, C) に転位してnumpy化
    waveform_np = audio_tensor.numpy().T
    fmt = format_name.upper()
    subtype = "FLOAT" if format_name == "wav" else "PCM_16"
    buf = io.BytesIO()
    soundfile.write(buf, waveform_np, sample_rate, format=fmt, subtype=subtype)
    return buf.getvalue()


def _run_synthesize(runtime, sampling_req):
    """スレッドプール上で同期的に推論を実行する"""
    return runtime.synthesize(sampling_req)


def _run_synthesize_long(runtime, long_req):
    """スレッドプール上で長文分割推論を実行する"""
    return runtime.synthesize_long(long_req)


def _build_common_kwargs(req: SynthesizeRequest) -> tuple[dict, str | None]:
    """SamplingRequest / LongTextSamplingRequest 共通の kwargs を構築する

    両 dataclass で共有可能なフィールドを dict として返し、
   呼び出し側で残りのフィールド（seconds / max_segment_seconds 等）を個別に追加する
   """
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

    # リクエストの caption があれば優先、無ければ speaker の caption を使う
    effective_caption = req.caption
    if effective_caption is None and speaker_data is not None:
        effective_caption = speaker_data.get("caption")

    no_ref = req.speakerId is None
    norm_db = speaker_data.get("normalizeDb", -16.0) if speaker_data else None
    ensure_max = speaker_data.get("ensureMax", True) if speaker_data else True
    max_ref = (
        speaker_data.get("maxRefSeconds", env.max_ref_seconds)
        if speaker_data
        else env.max_ref_seconds
    )

    common: dict[str, Any] = dict(
        text=req.text,
        caption=effective_caption,
        ref_latent=ref_latent_path,
        no_ref=no_ref,
        ref_normalize_db=norm_db,
        ref_ensure_max=ensure_max,
        max_ref_seconds=max_ref,
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
        max_text_len=req.tokenLimits.maxTextLen,
        max_caption_len=req.tokenLimits.maxCaptionLen,
    )

    return common, effective_caption


async def _execute_synthesis(
    req: SynthesizeRequest,
    request_id: str,
) -> Response | SynthesizeResponse:
    """POST / GET 共通の音声合成処理"""
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

    # 共通パラメータを構築
    common_kwargs, effective_caption = _build_common_kwargs(req)

    runtime = get_runtime()
    checkpoint = env.default_model

    is_long = req.longText is not None

    if is_long:
        # maxBatchSegments をサーバ設定の上限でクランプする
        effective_max_batch = min(
            req.longText.maxBatchSegments, env.max_batch_segments
        )
        long_req = LongTextSamplingRequest(
            **common_kwargs,
            duration_scale=req.duration.durationScale,
            max_segment_seconds=req.longText.maxSegmentSeconds,
            max_segment_chars=req.longText.maxSegmentChars,
            chars_per_second=req.longText.charsPerSecond,
            min_segment_chars=req.longText.minSegmentChars,
            segment_gap_seconds=req.longText.segmentGapSeconds,
            segment_trim_silence_db=req.longText.segmentTrimSilenceDb,
            max_batch_segments=effective_max_batch,
        )

        # 推論をスレッドプールで実行しイベントループを解放する
        try:
            result = await asyncio.to_thread(
                _run_synthesize_long, runtime, long_req
            )
        except Exception:
            logger.exception("Long text synthesis failed")
            raise api_error(
                ErrorCode.INTERNAL_SERVER_ERROR,
                message="Synthesis failed",
            ) from None
    else:
        sampling_req = SamplingRequest(
            **common_kwargs,
            num_candidates=req.sampling.numCandidates,
            # nullならモデル内蔵のduration predictorがテキスト長から自動予測
            seconds=req.duration.seconds,
            max_seconds=env.max_generate_seconds,
            duration_scale=req.duration.durationScale,
            min_seconds=req.duration.minSeconds,
        )

        # 推論をスレッドプールで実行しイベントループを解放する
        try:
            result = await asyncio.to_thread(
                _run_synthesize, runtime, sampling_req
            )
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
        wav_bytes = _save_wav_bytes(
            audio_tensor, result.sample_rate, req.output.format
        )
        headers: dict[str, str] = {
            "X-Sample-Rate": str(result.sample_rate),
            "X-Seed": str(result.used_seed),
            "Content-Disposition": "inline",
        }
        if is_long and hasattr(result, "segments") and result.segments:
            headers["X-Segments"] = str(len(result.segments))
        return Response(
            content=wav_bytes,
            media_type=(
                "audio/wav"
                if req.output.format == "wav"
                else f"audio/{req.output.format}"
            ),
            headers=headers,
        )

    # inline モード: 音声を base64 エンコードして返す
    audios: list[AudioItem] = []
    for i, audio_tensor in enumerate(result.audios):
        audio_cpu = audio_tensor.detach().cpu()
        wav_bytes = _save_wav_bytes(audio_cpu, result.sample_rate, "wav")
        dur = audio_cpu.shape[-1] / result.sample_rate
        audios.append(
            AudioItem(
                index=i,
                contents=base64.b64encode(wav_bytes).decode("ascii"),
                mimeType="audio/wav",
                sampleRate=result.sample_rate,
                duration=round(dur, 2),
            )
        )

    # タイミング情報を整形
    timings: dict[str, Any] = {
        "totalToDecodeMs": round(result.total_to_decode * 1000, 1)
    }
    for name, elapsed in result.stage_timings:
        timings[f"{name}Ms"] = round(elapsed * 1000, 1)

    mode = _determine_conditioning_mode(req.speakerId, effective_caption)

    segments: list[SegmentInfo] | None = None
    if is_long and hasattr(result, "segments") and result.segments:
        segments = [
            SegmentInfo(
                index=i,
                text=s.text,
                estimatedSeconds=s.estimatedSeconds,
            )
            for i, s in enumerate(result.segments)
        ]

    return SynthesizeResponse(
        id=request_id,
        status="succeeded",
        model=ModelInfo(checkpoint=checkpoint),
        conditioning=ConditioningInfo(
            speakerId=req.speakerId,
            caption=effective_caption,
            mode=mode,
        ),
        audios=audios,
        seed=result.used_seed,
        timings=timings,
        messages=result.messages,
        segments=segments,
    )


@synthesize_router.post("", response_model=None)
async def synthesize(
    req: SynthesizeRequest,
    request: Request,
):
    """音声合成を実行する"""
    # ミドルウェアで生成したリクエストIDを取得し、レスポンスの id に一貫性を持たせる
    request_id = getattr(request.state, REQUEST_ID_KEY, "")
    return await _execute_synthesis(req, request_id)


@synthesize_router.get("", response_model=None)
async def synthesize_get(
    query: SynthesizeQueryParams,
    request: Request,
):
    """音声合成を実行する (GET)

    TikTok TTS API ライクなクエリベースの簡易呼び出し。
    主要パラメータのみを受け付け、それ以外はサーバ既定値を使う。
    完全制御が必要な場合は POST /v1/synthesize を使用すること。
    """
    request_id = getattr(request.state, REQUEST_ID_KEY, "")
    req = SynthesizeRequest(
        text=query.text,
        speakerId=query.speakerId,
        caption=query.caption,
        sampling=SamplingParams(seed=query.seed),
        output=OutputParams(format=query.format, mode=query.method),
    )
    return await _execute_synthesis(req, request_id)
