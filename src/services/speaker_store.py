"""話者情報の on-memory CRUD と speakers.json の原子書き込み"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src.configs.env import env
from src.services.logger import logger

# 起動時に一括読み込みする on-memory dict
_SPEAKERS: dict[str, dict[str, Any]] = {}


def _data_dir() -> Path:
    return Path(env.data_dir)


def _speakers_json_path() -> Path:
    return _data_dir() / "speakers.json"


def _latents_dir() -> Path:
    return _data_dir() / "latents"


def _latent_path(speaker_id: str) -> Path:
    return _latents_dir() / f"{speaker_id}.pt"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """tmp -> rename で原子書き込みする"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name,
        suffix=".tmp",
    )
    try:
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, str(path))
    except BaseException:
        # 書き込み失敗時は tmp を残さない
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save() -> None:
    """メモリ上の _SPEAKERS を disk へ原子書き込みする"""
    _atomic_write_json(_speakers_json_path(), _SPEAKERS)


def load_speakers() -> None:
    """起動時に speakers.json を読み込む"""
    global _SPEAKERS
    path = _speakers_json_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            _SPEAKERS = json.load(f)
        logger.info(f"Loaded {len(_SPEAKERS)} speakers from {path}")
    else:
        _SPEAKERS = {}
        logger.info("No speakers.json found, starting empty")


def all_speakers() -> dict[str, dict[str, Any]]:
    """全話者を返す"""
    return _SPEAKERS


def get_speaker(speaker_id: str) -> dict[str, Any] | None:
    """ID で話者を検索する"""
    return _SPEAKERS.get(speaker_id)


def find_by_sha256(
    audio_sha256: str,
    normalize_db: float,
    ensure_max: bool,
) -> str | None:
    """同一音声SHA256 + 前処理条件の既存話者を探す"""
    for sid, data in _SPEAKERS.items():
        if (
            data.get("sha256") == audio_sha256
            and data.get("normalizeDb") == normalize_db
            and data.get("ensureMax") == ensure_max
        ):
            return sid
    return None


def register_speaker(
    *,
    audio_sha256: str,
    name: str,
    description: str,
    max_ref_seconds: float,
    normalize_db: float,
    ensure_max: bool,
    codec_repo: str,
    latent_tensor: Any,
) -> str:
    """新規話者を登録する"""
    speaker_id = str(uuid.uuid4())
    now = _now_iso()

    # latent をファイルへ保存
    _latents_dir().mkdir(parents=True, exist_ok=True)
    latent_path = _latent_path(speaker_id)
    torch.save(latent_tensor.cpu(), str(latent_path))

    _SPEAKERS[speaker_id] = {
        "name": name,
        "description": description,
        "sha256": audio_sha256,
        "maxRefSeconds": max_ref_seconds,
        "normalizeDb": normalize_db,
        "ensureMax": ensure_max,
        "codecRepo": codec_repo,
        "createdAt": now,
        "updatedAt": now,
        "lastUsedAt": None,
    }
    _save()

    logger.info(f"Registered speaker {speaker_id} ({name})")
    return speaker_id


def delete_speaker(speaker_id: str) -> bool:
    """話者を削除する"""
    if speaker_id not in _SPEAKERS:
        return False

    del _SPEAKERS[speaker_id]
    _save()

    latent_path = _latent_path(speaker_id)
    try:
        if latent_path.exists():
            latent_path.unlink()
    except OSError:
        logger.warning(f"Failed to delete latent: {latent_path}")

    logger.info(f"Deleted speaker {speaker_id}")
    return True


def update_last_used(speaker_id: str) -> None:
    """話者の lastUsedAt を更新する"""
    if speaker_id not in _SPEAKERS:
        return
    _SPEAKERS[speaker_id]["lastUsedAt"] = _now_iso()
    _SPEAKERS[speaker_id]["updatedAt"] = _now_iso()
    _save()


def compute_sha256(audio_bytes: bytes) -> str:
    """音声バイト列から SHA256 を計算する"""
    return hashlib.sha256(audio_bytes).hexdigest()


def latent_path_for(speaker_id: str) -> str:
    """話者の latent ファイルパスを返す"""
    return str(_latent_path(speaker_id))
