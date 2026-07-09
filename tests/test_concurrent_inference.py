"""同時推論の動作確認用スクリプト

複数リクエストを同時に送信し、各リクエストの所要時間と全体の壁時間を計測する。
- 直列実行時: 各リクエストの合計時間 ≒ 全体壁時間
- 並列実行時: 全体壁時間 < 各リクエストの合計時間
"""

import asyncio
import time

import httpx

URL = "http://127.0.0.1:8000/v1/synthesize"
SPEAKER_ID = "da642a25-eaa1-4aec-8bed-bd6b58c9c5d2"

# テストケース: 短文 / 中文 の2パターン (VRAM 12GB 環境向け)
TEST_CASES = [
    {
        "label": "short",
        "body": {
            "text": "せんせいおはよ～\n今日も早いねぇ～",
            "speakerId": SPEAKER_ID,
            "sampling": {"numSteps": 18},
            "guidance": {"cfgScaleSpeaker": 5.5, "cfgMinT": 0.4},
            "output": {"format": "wav", "mode": "buffer"},
        },
    },
    {
        "label": "medium",
        "body": {
            "text": (
                "こんにちは！今日はとてもいい天気ですね。"
                "お散歩に行きたくなっちゃうような、そんな素敵な一日になりそうです。"
                "せんせいも一緒にでかけませんか？"
            ),
            "speakerId": SPEAKER_ID,
            "sampling": {"numSteps": 18},
            "guidance": {"cfgScaleSpeaker": 5.5, "cfgMinT": 0.4},
            "output": {"format": "wav", "mode": "buffer"},
        },
    },
    {
        "label": "s2",
        "body": {
            "text": "おやすみなさい、また明日ね。",
            "speakerId": SPEAKER_ID,
            "sampling": {"numSteps": 18},
            "guidance": {"cfgScaleSpeaker": 5.5, "cfgMinT": 0.4},
            "output": {"format": "wav", "mode": "buffer"},
        },
    },
    {
        "label": "s3",
        "body": {
            "text": "うん、わかった！すぐ行くね。",
            "speakerId": SPEAKER_ID,
            "sampling": {"numSteps": 18},
            "guidance": {"cfgScaleSpeaker": 5.5, "cfgMinT": 0.4},
            "output": {"format": "wav", "mode": "buffer"},
        },
    },
    {
        "label": "m2",
        "body": {
            "text": (
                "今日は何食べようか？ラーメンがいいなぁ。"
                "でも昨日もラーメンだったから、たまには違うものもいいかも。"
                "じゃあカレーにしようよ！チキンカレーが食べたいな。"
            ),
            "speakerId": SPEAKER_ID,
            "sampling": {"numSteps": 18},
            "guidance": {"cfgScaleSpeaker": 5.5, "cfgMinT": 0.4},
            "output": {"format": "wav", "mode": "buffer"},
        },
    },
]


async def send_one(
    client: httpx.AsyncClient,
    case: dict,
) -> dict:
    """1件の合成リクエストを送信し、所要時間と結果を返す"""
    label = case["label"]
    t0 = time.perf_counter()
    resp = await client.post(URL, json=case["body"], timeout=300.0)
    elapsed = time.perf_counter() - t0

    status = resp.status_code
    audio_size = len(resp.content) if status == 200 else 0

    return {
        "label": label,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "audio_bytes": audio_size,
    }


async def run_concurrent() -> None:
    """全テストケースを同時に送信する"""
    print("=" * 60)
    n = len(TEST_CASES)
    print(f"{n} requests sent simultaneously (MAX_PARALLELISM=1)")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [send_one(client, c) for c in TEST_CASES]
        results = await asyncio.gather(*tasks)
        wall_total = time.perf_counter() - wall_start

    sum_elapsed = sum(r["elapsed_s"] for r in results)
    for r in sorted(results, key=lambda x: x["elapsed_s"]):
        status_str = f"HTTP {r['status']}"
        size_str = f"{r['audio_bytes']:,} bytes" if r["audio_bytes"] else "(no body)"
        print(f"  [{r['label']:>7s}] {r['elapsed_s']:.3f}s  {status_str}  {size_str}")

    print("-" * 60)
    print(f"  Sum of per-request times : {sum_elapsed:.3f}s")
    print(f"  Wall time                : {wall_total:.3f}s")
    speedup = sum_elapsed / wall_total if wall_total > 0 else 0
    print(f"  Speedup ratio            : {speedup:.2f}x")

    if speedup > 1.5:
        print("  -> Parallel inference is working")
    elif speedup > 1.1:
        print("  -> Partial parallelism detected")
    else:
        print("  -> Serial execution (requests queued)")
    print()



async def run_serial() -> None:
    """比較用: 同じリクエストを直列送信する"""
    print("=" * 60)
    print("Serial baseline (one by one)")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        results = []
        for case in TEST_CASES:
            r = await send_one(client, case)
            results.append(r)
        wall_total = time.perf_counter() - wall_start

    sum_elapsed = sum(r["elapsed_s"] for r in results)
    for r in results:
        status_str = f"HTTP {r['status']}"
        size_str = f"{r['audio_bytes']:,} bytes" if r["audio_bytes"] else "(no body)"
        print(f"  [{r['label']:>7s}] {r['elapsed_s']:.3f}s  {status_str}  {size_str}")

    print("-" * 60)
    print(f"  Sum of per-request times : {sum_elapsed:.3f}s")
    print(f"  Wall time (serial)       : {wall_total:.3f}s")
    print()


async def main() -> None:
    await run_serial()
    await run_concurrent()


if __name__ == "__main__":
    asyncio.run(main())
