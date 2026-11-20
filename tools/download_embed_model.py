# -*- coding: utf-8 -*-
"""
tools/download_embed_model.py —— 下载本地中文语义向量模型（bge-small-zh-v1.5 ONNX）

为什么用这个模型：
  · 纯本地推理，onnxruntime + numpy 即可，不引入 torch（省 800MB+ 依赖）；
  · 中文语义检索质量在 24M 参数量级里属第一梯队，512 维，单条编码 ~10ms（CPU）；
  · 模型权重放在 `模型权重/bge-small-zh-v1.5/`，`.gitignore` 已排除，不进仓库。

来源：ModelScope 镜像 `Xenova/bge-small-zh-v1.5`（Transformers.js 官方 ONNX 分发）。
下载失败会自动重试（断点续传），全部成功后才写 `.ok` 标记。

用法：
    python tools/download_embed_model.py
    python tools/download_embed_model.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "模型权重" / "bge-small-zh-v1.5"
API = "https://modelscope.cn/api/v1/models/Xenova/bge-small-zh-v1.5/repo?Revision=master&FilePath="

FILES = [
    ("onnx/model.onnx", "model.onnx", 94_851_877),
    ("tokenizer.json", "tokenizer.json", None),
    ("config.json", "config.json", None),
]


def download(remote: str, local: Path, expect: int | None, retries: int = 4) -> bool:
    if local.exists() and (expect is None or local.stat().st_size == expect):
        print(f"  [=] {local.name} 已存在（{local.stat().st_size} B）")
        return True
    url = API + urllib.parse.quote(remote)
    for attempt in range(1, retries + 1):
        have = local.stat().st_size if local.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r, local.open("ab" if have else "wb") as f:
                total = r.headers.get("Content-Range") or r.headers.get("Content-Length") or ""
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            size = local.stat().st_size
            if expect is None or size == expect:
                print(f"  [√] {local.name} {size} B  {time.time()-t0:.1f}s  ({total})")
                return True
            print(f"  [!] {local.name} 大小不符 {size}/{expect}，第 {attempt} 次重试")
        except Exception as exc:
            print(f"  [!] {local.name} 第 {attempt} 次失败：{type(exc).__name__}")
            time.sleep(2)
    return local.exists() and (expect is None or local.stat().st_size == expect)


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 bge-small-zh-v1.5 ONNX 向量模型")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[i] 目标目录：{OUT_DIR}")
    ok = True
    for remote, name, expect in FILES:
        local = OUT_DIR / name
        if args.force and local.exists():
            local.unlink()
        ok = download(remote, local, expect) and ok
    if ok:
        (OUT_DIR / ".ok").write_text("bge-small-zh-v1.5 ready\n", encoding="utf-8")
        print("[√] 全部就绪")
        return 0
    print("[×] 下载未完成，请重试本脚本（支持断点续传）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
