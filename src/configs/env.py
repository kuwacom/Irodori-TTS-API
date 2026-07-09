import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    """環境変数から読むアプリケーション設定"""

    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    reload: bool = Field(default=False, alias="RELOAD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_policy_origin: str = Field(default="*", alias="CORS_POLICY_ORIGIN")

    # Irodori TTS モデル設定
    default_model: str = Field(
        default="Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        alias="DEFAULT_MODEL",
    )
    codec_repo: str = Field(
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        alias="CODEC_REPO",
    )
    model_device: str = Field(default="cuda", alias="MODEL_DEVICE")
    codec_device: str = Field(default="cpu", alias="CODEC_DEVICE")
    model_precision: str = Field(default="fp32", alias="MODEL_PRECISION")
    models_dir: str = Field(
        default="models",
        alias="MODELS_DIR",
        description="Hugging Face Hub のキャッシュディレクトリ",
    )

    # GPU 上の同時推論スロット数
    # InferenceRuntime 側で Semaphore および CUDA Stream プールとして管理され、
    # 複数リクエストが異なる Stream 上で並列に推論を実行できるようになる
    max_parallelism: int = Field(default=1, alias="MAX_PARALLELISM")

    # SilentCipher ウォーターマーク有無。
    # lib 側もデフォルト OFF だが明示的に制御可能とするため環境変数化する
    enable_watermark: bool = Field(default=False, alias="ENABLE_WATERMARK")

    # PyTorch が認識するGPUを制限する（マルチGPU環境でCC非対応GPUを除外する等）
    cuda_visible_devices: str = Field(
        default="",
        alias="CUDA_VISIBLE_DEVICES",
        description="空文字ならOS環境変数やデフォルトに従う",
    )

    # 制限値
    max_ref_seconds: float = Field(default=30.0, alias="MAX_REF_SECONDS")
    max_generate_seconds: float = Field(default=30.0, alias="MAX_GENERATE_SECONDS")
    max_num_candidates: int = Field(default=4, alias="MAX_NUM_CANDIDATES")
    max_request_body_size: int = Field(
        default=32 * 1024 * 1024,
        alias="MAX_REQUEST_BODY_SIZE",
    )

    # データディレクトリ
    data_dir: str = Field(default="data", alias="DATA_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_policy_origin")
    @classmethod
    def validate_cors_policy_origin(cls, value: str) -> str:
        """空文字による CORS 設定漏れを起動時に検出する"""
        if not value.strip():
            raise ValueError("CORS_POLICY_ORIGIN must not be empty")
        return value

    def model_post_init(self, __context: object) -> None:
        """Env 初期化直後にOS環境変数を反映する

        CUDA_VISIBLE_DEVICES, HF_HUB_CACHE 等は
        PyTorch / HuggingFace がプロセス初期化時に
        os.environ から読むため .env だけでは効果がなく
        明示的に os.environ へ書き込む必要がある
        """
        # CUDA_VISIBLE_DEVICES: 空文字でなければOS環境変数に反映
        if self.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        resolved = Path(self.models_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(resolved))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(resolved))
        os.environ.setdefault("TORCH_HOME", str(resolved / "torch"))

    @property
    def cors_policy_origins(self) -> list[str]:
        """環境変数形式を FastAPI の配列形式へ変換する"""
        if self.cors_policy_origin == "*":
            return ["*"]

        return [
            origin.strip()
            for origin in self.cors_policy_origin.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_env() -> Env:
    """設定インスタンスをプロセス内で共有する"""
    return Env()


env = get_env()
