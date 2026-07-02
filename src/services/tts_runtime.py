"""InferenceRuntime のグローバル管理"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import RuntimeKey, get_cached_runtime

from src.configs.env import env
from src.services.logger import logger

if TYPE_CHECKING:
    from irodori_tts.inference_runtime import InferenceRuntime

_runtime: InferenceRuntime | None = None
_runtime_key: RuntimeKey | None = None


def _resolve_checkpoint(checkpoint: str) -> str:
    """Hub ID なら hf_hub_download でローカルパスを解決する

    ローカルパス（.pt / .safetensors で終わる、または / \\ を含まない短い文字列以外）
    はそのまま返す
    """
    # すでにローカルファイルが存在すればそのまま
    if Path(checkpoint).is_file():
        return checkpoint

    # Hub ID (org/repo 形式) ならダウンロードして解決
    for filename in ("model.safetensors", "model.pt"):
        try:
            resolved = hf_hub_download(repo_id=checkpoint, filename=filename)
            logger.info(f"Downloaded {filename} from hf://{checkpoint} -> {resolved}")
            return str(resolved)
        except Exception:
            continue

    msg = f"Cannot resolve checkpoint: {checkpoint}"
    raise FileNotFoundError(msg)


def _build_key(checkpoint: str | None = None) -> RuntimeKey:
    """現在の環境変数から RuntimeKey を構築する"""
    return RuntimeKey(
        checkpoint=checkpoint or env.default_model,
        model_device=env.model_device,
        codec_repo=env.codec_repo,
        model_precision=env.model_precision,
        codec_device=env.codec_device,
    )


def init_runtime(checkpoint: str | None = None) -> InferenceRuntime:
    """InferenceRuntime を構築してグローバルに保持する（起動時に1度だけ呼ぶ）"""
    global _runtime, _runtime_key

    raw_checkpoint = checkpoint or env.default_model
    resolved_checkpoint = _resolve_checkpoint(raw_checkpoint)

    key = _build_key(resolved_checkpoint)
    runtime, loaded = get_cached_runtime(key)
    if loaded:
        logger.info(f"Loaded InferenceRuntime for checkpoint={key.checkpoint}")
    _runtime = runtime
    _runtime_key = key
    return runtime


def get_runtime() -> InferenceRuntime:
    """初期化済みの InferenceRuntime を返す"""
    if _runtime is None:
        msg = "InferenceRuntime is not initialized. Call init_runtime() at startup."
        raise RuntimeError(msg)
    return _runtime
