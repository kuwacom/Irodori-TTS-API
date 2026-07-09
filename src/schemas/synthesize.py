"""音声合成リクエスト・レスポンスの Pydantic モデル"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    """モデル指定パラメータ"""

    name: str | None = Field(
        default=None,
        description="モデル名（None=サーバ設定値）",
    )
    loraAdapter: str | None = Field(
        default=None,
        description="LoRAアダプタ名（None=使用しない）",
    )


class SamplingParams(BaseModel):
    """サンプリングパラメータ"""

    preset: str = Field(
        default="custom",
        description="preset name",
    )
    numSteps: int = Field(default=40, ge=1, description="拡散ステップ数")
    numCandidates: int = Field(default=1, ge=1, le=4, description="候補数")
    seed: int | None = Field(default=None, description="乱数シード")
    tScheduleMode: str = Field(
        default="linear",
        description="tスケジュールモード (linear / sway)",
    )
    swayCoeff: float = Field(default=-1.0, description="sway coefficient")


class DurationParams(BaseModel):
    """音声長制御パラメータ"""

    seconds: float | None = Field(
        default=None,
        gt=0,
        description="生成秒数。nullならモデル内蔵のduration predictorが自動予測",
    )
    durationScale: float = Field(
        default=1.0,
        gt=0,
        description="duration predictorの出力倍率",
    )
    minSeconds: float = Field(
        default=0.5,
        gt=0,
        description="生成秒数の下限",
    )


class GuidanceParams(BaseModel):
    """ガイダンスパラメータ"""

    mode: str = Field(
        default="independent",
        description="cfg ガイダンスモード (independent / joint / alternating)",
    )
    cfgScale: float | None = Field(
        default=None,
        ge=0,
        description="全CFGスケールを一括指定（None=個別設定を使用）",
    )
    cfgScaleText: float = Field(
        default=3.0,
        ge=0,
        description="テキスト CFG",
    )
    cfgScaleCaption: float = Field(
        default=3.0,
        ge=0,
        description="キャプション CFG",
    )
    cfgScaleSpeaker: float = Field(
        default=5.0,
        ge=0,
        description="スピーカー CFG",
    )
    cfgMinT: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="CFGが有効になるtの下限",
    )
    cfgMaxT: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="CFGが有効になるtの上限",
    )


class TruncationParams(BaseModel):
    """拡散サンプリングの数値的補正（truncation/rescale）"""

    factor: float | None = Field(
        default=None,
        description="truncation factor（None=無効）",
    )
    rescaleK: float | None = Field(
        default=None,
        description="rescale k（None=無効）",
    )
    rescaleSigma: float | None = Field(
        default=None,
        description="rescale sigma（None=無効）",
    )


class KvCacheParams(BaseModel):
    """KVキャッシュ関連の挙動"""

    contextKvCache: bool = Field(
        default=True,
        description="context KVキャッシュを使用",
    )
    speakerKvScale: float | None = Field(
        default=None,
        description="speaker KVスケール（None=無効）",
    )
    speakerKvMinT: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="speaker KVが有効になるtの下限",
    )
    speakerKvMaxLayers: int | None = Field(
        default=None,
        description="speaker KVが適用されるレイヤー数上限",
    )
    speakerUncondMode: str = Field(
        default="mask",
        description="speaker無条件時のモード (mask / zero)",
    )


class TailTrimParams(BaseModel):
    """末尾の無音トリム"""

    trimTail: bool = Field(
        default=True,
        description="末尾の無音をトリム",
    )
    tailWindowSize: int = Field(
        default=20,
        ge=1,
        description="末尾トリムの窓サイズ",
    )
    tailStdThreshold: float = Field(
        default=0.05,
        description="末尾トリムの標準偏差しきい値",
    )
    tailMeanThreshold: float = Field(
        default=0.1,
        description="末尾トリムの平均しきい値",
    )


class DecodeParams(BaseModel):
    """デコード方式"""

    mode: str = Field(
        default="sequential",
        description="デコード方式 (sequential / batch)",
    )


class TokenLimitsParams(BaseModel):
    """トークン長制限"""

    maxTextLen: int | None = Field(
        default=None,
        description="テキスト最大トークン長（None=モデル上限）",
    )
    maxCaptionLen: int | None = Field(
        default=None,
        description="キャプション最大トークン長（None=モデル上限）",
    )


class OutputParams(BaseModel):
    """出力パラメータ"""

    format: str = Field(default="wav", description="出力形式")
    mode: str = Field(
        default="buffer",
        description="inline: JSON応答にbase64埋め込み / buffer: 音声を直接返す",
    )


class LongTextParams(BaseModel):
    """長文分割読み上げの設定

    指定時はテキストを自動分割し、各セグメントを個別に推論して結合する。
    未指定時は従来通り単一推論で処理する。
    """

    maxSegmentSeconds: float = Field(
        default=30.0, gt=0,
        description="1セグメントあたりの最大推定秒数",
    )
    maxSegmentChars: int = Field(
        default=180, ge=1,
        description="1セグメントの最大文字数",
    )
    charsPerSecond: float = Field(
        default=10.0, gt=0,
        description="1秒あたりの発話文字数推定値",
    )
    minSegmentChars: int = Field(
        default=4, ge=1,
        description=(
            "セグメントの最小文字数（これ以下は前セグメントへ結合）"
        ),
    )
    segmentGapSeconds: float = Field(
        default=0.2, ge=0,
        description="セグメント間の無音区間（秒）",
    )
    segmentTrimSilenceDb: float = Field(
        default=-40.0,
        description="セグメント前後無音トリム閾値（dB）",
    )
    maxBatchSegments: int = Field(
        default=8, ge=1,
        description=(
            "1バッチで同時処理するセグメント最大数"
            "（サーバ設定の MAX_BATCH_SEGMENTS が上限）"
        ),
    )


class SynthesizeRequest(BaseModel):
    """POST /v1/synthesize のリクエストボディ"""

    text: str = Field(..., min_length=1, description="読み上げ本文")
    speakerId: str | None = Field(default=None, description="話者ID")
    caption: str | None = Field(default=None, description="スタイル指示")
    model: ModelParams = Field(default_factory=ModelParams)
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    duration: DurationParams = Field(default_factory=DurationParams)
    guidance: GuidanceParams = Field(default_factory=GuidanceParams)
    truncation: TruncationParams = Field(default_factory=TruncationParams)
    kvCache: KvCacheParams = Field(default_factory=KvCacheParams)
    tailTrim: TailTrimParams = Field(default_factory=TailTrimParams)
    decode: DecodeParams = Field(default_factory=DecodeParams)
    tokenLimits: TokenLimitsParams = Field(default_factory=TokenLimitsParams)
    output: OutputParams = Field(default_factory=OutputParams)
    longText: LongTextParams | None = Field(
        default=None,
        description=(
            "長文分割設定。指定時は synthesize_long へ切り替わり、"
            "テキストを自動分割して複数セグメントを生成・結合する"
        ),
    )


class AudioItem(BaseModel):
    """合成結果の各音声"""

    index: int
    contents: str = Field(description="base64 エンコード音声")
    mimeType: str = "audio/wav"
    sampleRate: int
    duration: float


class ModelInfo(BaseModel):
    """使用モデル情報"""

    checkpoint: str


class ConditioningInfo(BaseModel):
    """条件付け情報"""

    speakerId: str | None = None
    caption: str | None = None
    mode: str


class SegmentInfo(BaseModel):
    """長文分割時の各セグメント情報"""

    index: int
    text: str
    estimatedSeconds: float


class SynthesizeResponse(BaseModel):
    """POST /v1/synthesize のレスポンス"""

    id: str
    status: str = "succeeded"
    model: ModelInfo
    conditioning: ConditioningInfo
    audios: list[AudioItem]
    seed: int
    timings: dict[str, Any] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    segments: list[SegmentInfo] | None = None
