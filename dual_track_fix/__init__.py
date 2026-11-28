# -*- coding: utf-8 -*-
"""stage8 双轨框选修正版（stage8-v2）。

模块分工：
  settings.py  有效配置的唯一来源（跑批与正常入口共用）
  slots.py     列槽切分（周期 + 低谷吸附）
  detect.py    修正版检测器（slots / clip / tighten 三个动作可单独关）
  relax.py     断组跨度上限的换算（作为调用参数使用，不再 monkeypatch）
"""
