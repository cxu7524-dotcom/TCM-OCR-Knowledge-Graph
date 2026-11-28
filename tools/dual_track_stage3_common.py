# -*- coding: utf-8 -*-
"""第三阶段小工具：产物规范化与指纹。

预测/识别文件里带耗时（timing_s），每次运行都不同。要让「同一算法两次运行」
可比，必须把耗时剥掉再算哈希；文件级哈希只用于核对「文件没被替换」。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

VOLATILE_KEYS = {'timing_s', 'timings', 'generated_at', 'seconds', 'mean_s_per_page'}


def strip_timings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_timings(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_timings(v) for v in obj]
    return obj


def content_sha256(payload: Any) -> str:
    """去掉耗时后的内容哈希：同算法同输入两次运行应完全相同。"""
    blob = json.dumps(strip_timings(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {'file_sha256': file_sha256(path), 'content_sha256': content_sha256(payload),
            'bytes': path.stat().st_size}
