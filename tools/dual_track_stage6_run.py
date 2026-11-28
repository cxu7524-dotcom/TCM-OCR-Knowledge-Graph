# -*- coding: utf-8 -*-
"""阶段 6 批量运行：精修 stage5-v4 的全部已确认组。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_refine import DEFAULT_CONFIG, refine_all  # noqa: E402
from dual_track_refine.api import S6  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage6 双轨子框精修')
    ap.add_argument('--out-dir', default=str(S6))
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args(argv)
    out = refine_all(Path(args.out_dir), args.only, DEFAULT_CONFIG)
    s = out['summary']
    print(f"页 {s['pages']}｜组 {s['groups']}｜改框 {s['changed_groups']}｜"
          f"待核 {s['pending_groups']}｜补笔 {s['supplement_accepted']}")
    print('flags', s['flags'])
    print('content_sha256', out['determinism']['content_sha256'][:16])
    print('→', args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
