# Irodori TTS API


[Irodori-TTS-Optimiz](https://github.com/kuwacom/Irodori-TTS-Optimiz) を依存バックエンドにした RESTful 音声合成 APIサーバー


## 必要条件

- Python 3.12+
- NVIDIA GPU（Compute Capability 7.5以上推奨）
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー
- CMake 3.5+（sentencepieceビルド用）

## セットアップ

```bash
git clone https://github.com/kuwacom/Irodori-TTS-API.git
cd Irodori-TTS-API

cp .env.example .env
# .env を編集（モデル・デバイス等）

uv sync
```

初回の `uv sync` では PyTorch CUDA版とIrodori-TTSモデル（約1.2GB）をダウンロードするため時間がかかります。

## 起動

```bash
uv run task start
```

開発モード（ホットリロード）:

```bash
uv run task dev
```

## コマンド一覧

`uv run task <name>` で実行できます。

| コマンド | 説明 |
|---|---|
| `start` | サーバー起動 |
| `dev` | 開発サーバー起動 |
| `test` | テスト実行 |
| `lint` | Ruff lintチェック |
| `format` | Ruffフォーマット |
| `fix` | Ruff自動修正 |
| `check` | lint + テスト |

## 環境変数

`.env` ファイルで設定します。`.env.example` をコピーして編集してください。

### アプリケーション

| 変数 | デフォルト | 説明 |
|---|---|---|
| `HOST` | `127.0.0.1` | バインドアドレス |
| `PORT` | `8000` | ポート番号 |
| `RELOAD` | `false` | ホットリロード（開発用） |
| `LOG_LEVEL` | `INFO` | ログレベル |
| `CORS_POLICY_ORIGIN` | `*` | CORS許可オリジン（カンマ区切り） |

### モデル

| 変数 | デフォルト | 説明 |
|---|---|---|
| `DEFAULT_MODEL` | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | デフォルトのTTSモデル |
| `CODEC_REPO` | `Aratako/Semantic-DACVAE-Japanese-32dim` | DACVAEコーデックリポジトリ |
| `MODEL_DEVICE` | `cuda` | モデル配置デバイス |
| `CODEC_DEVICE` | `cpu` | コーデック配置デバイス |
| `MODEL_PRECISION` | `fp32` | モデル精度（`fp32` / `bf16`） |
| `CUDA_VISIBLE_DEVICES` | (空) | PyTorchが認識するGPUを制限 |

### 推論制御

| 変数 | デフォルト | 説明 |
|---|---|---|
| `MAX_PARALLELISM` | `1` | GPU上の同時推論スロット数（1=直列、2以上=並列推論）。VRAM容量に応じて調整 |
| `ENABLE_WATERMARK` | `false` | SilentCipherウォーターマーク（trueで有効化） |
| `MAX_BATCH_SEGMENTS` | `8` | 長文分割推論で1バッチあたりに同時処理するセグメント最大数の上限。リクエスト側の `maxBatchSegments` はこの値を超えられない |

**VRAM容量と精度・並列度の目安**

`MAX_PARALLELISM` はGPUのVRAM容量と`MODEL_PRECISION`の組み合わせで調整が必要です。
以下は目安であり、実際のテキスト長や生成秒数によって変動します。

| VRAM | 精度 | 推奨 `MAX_PARALLELISM` |
|---|---|---|
| 12GB | `fp32` | 1（並列は厳しい） |
| 12GB | `bf16` | 2（ギリギリ） |
| 24GB | `fp32` | 4〜5 |

### ディレクトリ

| 変数 | デフォルト | 説明 |
|---|---|---|
| `MODELS_DIR` | `models` | HuggingFace Hubキャッシュ |
| `DATA_DIR` | `data` | 話者データ |

### 制限値

| 変数 | デフォルト | 説明 |
|---|---|---|
| `MAX_REF_SECONDS` | `30.0` | 参照音声の最大秒数 |
| `MAX_GENERATE_SECONDS` | `30.0` | 生成音声の最大秒数 |
| `MAX_NUM_CANDIDATES` | `4` | 候補数の上限 |
| `MAX_REQUEST_BODY_SIZE` | `33554432` | リクエストボディ最大サイズ（バイト） |

### GPUに関する注意点

**マルチGPU構成**

モデルとコーデックを別々のGPUに配置できます。

```env
MODEL_DEVICE=cuda:0
CODEC_DEVICE=cuda:1
```

**非対応GPUの除外**

Compute Capability 7.5未満のGPU（GTX 10xx等）が接続されている環境では、
cuDNNの初期化に失敗するため `CUDA_VISIBLE_DEVICES` でPyTorchの視界から除外する必要があります。

```env
CUDA_VISIBLE_DEVICES=0
MODEL_DEVICE=cuda:0
CODEC_DEVICE=cuda:0
```

`.env` に書いた `CUDA_VISIBLE_DEVICES` は起動時にOS環境変数へ反映されるため、
PyTorchより前に確実に効きます。

**CPU動作**

GPUがない・非対応GPUのみの環境ではCPU動作も可能ですが、推論速度は大幅に低下します。

```env
MODEL_DEVICE=cpu
CODEC_DEVICE=cpu
```

## エンドポイント

### 話者管理

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/v1/speakers` | 話者登録 |
| `GET` | `/v1/speakers` | 話者一覧 |
| `GET` | `/v1/speakers/{speakerId}` | 話者詳細 |
| `DELETE` | `/v1/speakers/{speakerId}` | 話者削除 |

### 音声合成

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/v1/synthesize` | 音声合成 |
| `GET` | `/v1/synthesize` | 音声合成（クエリベース・簡易） |

---

### POST /v1/speakers

話者を登録する。音声ファイルからlatentを抽出し、以降の合成で参照できるようにする。

**Content-Type:** `multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `audio` | file | ○ | 音声ファイル（wav/mp3/flac） |
| `metadata` | string(JSON) | ○ | 話者メタデータ |

**metadata:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `name` | string | `""` | 話者名 |
| `description` | string | `""` | 説明 |
| `caption` | string/null | `null` | 合成時のデフォルトキャプション（スタイル指示） |
| `maxRefSeconds` | number | `30.0` | 参照音声の最大秒数 |
| `normalizeDb` | number | `-16.0` | 正規化目標 dB |
| `ensureMax` | boolean | `true` | 正規化後にピーククリップ |

```bash
curl -X POST http://localhost:8000/v1/speakers \
  -F "audio=@voice.wav" \
  -F 'metadata={"name":"female","description":"ナレーション向け"}'
```

```json
{
  "speakerId": "a1b2c3d4-...",
  "name": "female",
  "status": "created",
  "sha256": "a6f2...",
  "createdAt": "2026-06-25T10:00:00Z"
}
```

同一音声・同一条件の話者が既に存在する場合は `status: "already exists"` で既存IDを返します。

---

### GET /v1/speakers

登録済み話者の一覧を返す。

```json
[
  {
    "speakerId": "a1b2c3d4-...",
    "name": "female",
    "createdAt": "2026-06-25T10:00:00Z",
    "lastUsedAt": null
  }
]
```

---

### GET /v1/speakers/{speakerId}

話者の詳細情報を返す。

```json
{
  "speakerId": "a1b2c3d4-...",
  "name": "female",
  "description": "ナレーション向け",
  "caption": null,
  "sha256": "a6f2...",
  "maxRefSeconds": 30.0,
  "normalizeDb": -16.0,
  "ensureMax": true,
  "codecRepo": "Aratako/Semantic-DACVAE-Japanese-32dim",
  "createdAt": "2026-06-25T10:00:00Z",
  "updatedAt": "2026-06-25T10:00:00Z",
  "lastUsedAt": null
}
```

---

### DELETE /v1/speakers/{speakerId}

話者を削除する。latentファイルも同時に削除される。

```json
{
  "speakerId": "a1b2c3d4-...",
  "deleted": true
}
```

---

### POST /v1/synthesize

音声合成を実行する。以下の4パターンをサポート:

| パターン | speakerId | caption | 動作 |
|---|---|---|---|
| Reference | ○ | -- | 参照音声から話者を再現 |
| VoiceDesign | -- | ○ | キャプションでスタイルを指定 |
| Hybrid | ○ | ○ | 参照音声 + キャプション |
| Text-only | -- | -- | テキストのみ |

**Content-Type:** `application/json`

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `text` | string | ○ | -- | 読み上げ本文 |
| `speakerId` | string | | `null` | 話者ID |
| `caption` | string | | `null` | スタイル指示。speakerId に caption が設定されている場合はリクエストの caption が優先され、未指定時は speaker の caption が使われる |
| `model` | object | | 下表 | モデル指定 |
| `sampling` | object | | 下表 | サンプリング設定 |
| `duration` | object | | 下表 | 音声長制御 |
| `guidance` | object | | 下表 | ガイダンス設定 |
| `truncation` | object | | 下表 | 拡散サンプリングの数値的補正 |
| `kvCache` | object | | 下表 | KVキャッシュ関連の挙動 |
| `tailTrim` | object | | 下表 | 末尾の無音トリム |
| `decode` | object | | 下表 | デコード方式 |
| `tokenLimits` | object | | 下表 | トークン長制限 |
| `output` | object | | 下表 | 出力設定 |
| `longText` | object/null | `null` | 長文分割設定（下表）。指定時はテキストを自動分割してセグメントごとに推論・結合する |

**model:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `name` | string/null | `null` | モデル名（null=サーバ設定値） |
| `loraAdapter` | string/null | `null` | LoRAアダプタ名（null=使用しない） |

**sampling:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `preset` | string | `"custom"` | `balanced` / `quality` / `speed` / `extreme` / `custom` |
| `numSteps` | int | `40` | 拡散ステップ数 |
| `numCandidates` | int | `1` | 候補数（最大4） |
| `seed` | int/null | `null` | 乱数シード |
| `tScheduleMode` | string | `"linear"` | tスケジュールモード (linear / sway) |
| `swayCoeff` | float | `-1.0` | sway coefficient |

**duration:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `seconds` | float/null | `null` | 生成秒数。nullならモデル内蔵のduration predictorが自動予測 |
| `durationScale` | float | `1.0` | duration predictorの出力倍率 |
| `minSeconds` | float | `0.5` | 生成秒数の下限 |

**guidance:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `mode` | string | `"independent"` | CFGガイダンスモード (independent / joint / alternating) |
| `cfgScale` | float/null | `null` | 全CFGスケールを一括指定（null=個別設定を使用） |
| `cfgScaleText` | float | `3.0` | テキストCFG |
| `cfgScaleCaption` | float | `3.0` | キャプションCFG |
| `cfgScaleSpeaker` | float | `5.0` | スピーカーCFG |
| `cfgMinT` | float | `0.5` | CFGが有効になるtの下限 |
| `cfgMaxT` | float | `1.0` | CFGが有効になるtの上限 |

**truncation:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `factor` | float/null | `null` | truncation factor（null=無効） |
| `rescaleK` | float/null | `null` | rescale k（null=無効） |
| `rescaleSigma` | float/null | `null` | rescale sigma（null=無効） |

**kvCache:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `contextKvCache` | bool | `true` | context KVキャッシュを使用 |
| `speakerKvScale` | float/null | `null` | speaker KVスケール（null=無効） |
| `speakerKvMinT` | float/null | `null` | speaker KVが有効になるtの下限 |
| `speakerKvMaxLayers` | int/null | `null` | speaker KVが適用されるレイヤー数上限 |
| `speakerUncondMode` | string | `"mask"` | speaker無条件時のモード (mask / zero) |

**tailTrim:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `trimTail` | bool | `true` | 末尾の無音をトリム |
| `tailWindowSize` | int | `20` | 末尾トリムの窓サイズ |
| `tailStdThreshold` | float | `0.05` | 末尾トリムの標準偏差しきい値 |
| `tailMeanThreshold` | float | `0.1` | 末尾トリムの平均しきい値 |

**decode:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `mode` | string | `"sequential"` | デコード方式 (sequential / batch) |

**tokenLimits:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `maxTextLen` | int/null | `null` | テキスト最大トークン長（null=モデル上限） |
| `maxCaptionLen` | int/null | `null` | キャプション最大トークン長（null=モデル上限） |

**output:**

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `format` | string | `"wav"` | 出力形式 |
| `mode` | string | `"buffer"` | `buffer`: 音声を直接返す / `inline`: base64でJSONに埋め込む |

**longText:**

指定時は長文モードが有効になり、テキストが句読点等で自動分割され、各セグメントを個別推論したのち前後無音トリムと無音区間挿入で結合される。sampling, guidance, kvCache, tailTrim 等の全パラメータはそのまま利用可能。

`durationScale` が 1.0 を前提とした場合のパラメータ感:

- `30 / 180 / 0.2`（デフォルト）: 各セグメントを個別に作ってつなげたのと等しい、最も自然な結果になる
- `30 / 200 / 0.2`: 少々早めの読み上げで、詰め込み気味のthe読み上げ感が出る
- `28 / 200 / 0.2`: セグション時間が足りていない感じになり、不自然になりやすい

キャラクターごとのおすすめ設定:

- **ゆっくり話すキャラ**: `maxSegmentChars: 150`, `segmentGapSeconds: 0.3`
- **標準的なキャラ**: デフォルト値（`180`, `0.2`）そのままでOK
- **早口・元気なキャラ**: `maxSegmentChars: 200`, `segmentGapSeconds: 0.15`
- **落ち着いたナレーション**: `maxSegmentSeconds: 30`, `maxSegmentChars: 160`, `segmentGapSeconds: 0.25`

上記はあくまで目安であり、実際のキャラクター特性やテキスト内容によって調整が必要な場合がある。

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `maxSegmentSeconds` | float | `30.0` | 1セグメントあたりの最大推定秒数 |
| `maxSegmentChars` | int | `180` | 1セグメントの最大文字数 |
| `charsPerSecond` | float | `10.0` | 1秒あたりの発話文字数推定値 |
| `minSegmentChars` | int | `4` | セグメント最小文字数（これ以下は前セグメントへ結合） |
| `segmentGapSeconds` | float | `0.2` | セグメント間無音区間（秒） |
| `segmentTrimSilenceDb` | float | `-40.0` | セグメント前後無音トリム閾値 (dB) |
| `maxBatchSegments` | int | `8` | 1バッチで同時処理するセグメント最大数。サーバ設定 (`MAX_BATCH_SEGMENTS`) を超える場合は上限にクランプされる |

inline レスポンスには `segments` 配列が含まれる。buffer モードでは `X-Segments` ヘッダーにセグメント数が入る。

**VoiceDesign + bufferモードの例:**

```bash
curl -X POST http://localhost:8000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは！","caption":"元気な幼女"}' \
  --output voice.wav
```

**話者参照 + inlineモードの例:**

```bash
curl -X POST http://localhost:8000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"今日はいい天気ですね。","speakerId":"a1b2c3d4-...","output":{"mode":"inline"}}'
```

**bufferモードのレスポンス:** 音声データをそのまま返します（`Content-Type: audio/wav`）

以下のレスポンスヘッダを含みます:

| ヘッダ | 説明 |
|---|---|
| `X-Request-Id` | リクエストID（UUID） |
| `X-Sample-Rate` | サンプリングレート |
| `X-Seed` | 使用したシード値 |

**inlineモードのレスポンス:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "succeeded",
  "model": {
    "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
  },
  "conditioning": {
    "speakerId": "a1b2c3d4-...",
    "caption": null,
    "mode": "speaker"
  },
  "audios": [
    {
      "index": 0,
      "contents": "UklGRiQAAABXQVZF...",
      "mimeType": "audio/wav",
      "sampleRate": 48000,
      "duration": 3.42
    }
  ],
  "seed": 123456789,
  "timings": {
    "totalToDecodeMs": 9051.4
  },
  "messages": []
}
```

`id` はリクエストごとに UUID で生成され、ログにも `req_id` として出力されます。すべてのレスポンスヘッダに `X-Request-Id` として付与されます。

---

### GET /v1/synthesize

音声合成を実行する（クエリベース・簡易）。POST /v1/synthesize と同じ合成エンジンを使うが、主要パラメータのみをクエリパラメータで受け付ける。完全制御が必要な場合は POST を使用すること。

**クエリパラメーター:**

| パラメーター | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `text` | string | ○ | -- | 読み上げ本文 |
| `speakerId` | string | -- | `null` | 話者ID |
| `caption` | string | -- | `null` | スタイル指示 |
| `seed` | int | -- | `null` | 乱数シード |
| `format` | string | -- | `"wav"` | 出力形式（`wav` / `mp3` / `flac`） |
| `method` | string | -- | `"buffer"` | 応答方式。`buffer`: 音声を直接返す / `inline`: JSON応答にbase64埋め込み |

無効な値が指定された場合は 422（バリデーションエラー）となる。

**例:**

```http
GET /v1/synthesize?text=こんにちは&speakerId=a1b2c3d4-...
```

```bash
curl "http://localhost:8000/v1/synthesize?text=こんにちは&caption=元気な幼女" --output voice.wav
```

レスポンス仕様は POST /v1/synthesize の buffer / inline モードに準ずる。

## ディレクトリ構造

```
src/configs/      -- 環境変数設定
src/lib/          -- 共通エラー定義
src/middleware/    -- ロギング・エラーハンドラ
src/routes/v1/    -- APIエンドポイント（speakers, synthesize）
src/schemas/      -- Pydanticリクエスト/レスポンスモデル
src/services/     -- 話者ストア・TTSランタイム管理
data/             -- speakers.json + latents/*.pt（実行時に生成）
models/           -- HuggingFace Hubキャッシュ（自動生成）
```
