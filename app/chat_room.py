# -*- coding: utf-8 -*-
"""app/chat_room.py —— 聊天室（多聊天室）服务端，数据落在 data/chat_rooms.json。

给「游客 + 五位编码」这套前端身份一个真能跨设备说话的通道：
同一台机器上的不同浏览器、同一局域网里的不同电脑，只要连同一个服务就能互通。

房间模型：
    官方 5 间（中药主题，固定）+ 游客自建（有上限，超了淘汰最久没人用的）。
    每间房各自维护消息、成员与序号，互不串台。

范围（刻意做小）：
    不做账号、私聊、审核。房间与消息落盘到 data/chat_rooms.json（改动后最多 2 秒
    写一次，原子替换），**重启服务不再清空**；在场状态与邀请只在内存里，属临时信息。
    不往同门的 SQLite 里加任何表——删掉本文件、data/chat_rooms.json 与 main.py 里
    那段路由即可完全回退。

对外主要函数：
    list_rooms()                 房间列表（官方 / 游客）
    create_room(name, desc, id)  开一间
    snapshot(room, since, id)    取 since 之后的新消息 + 在室成员 + 邀请
    post(room, id, text)         发一条（限速、清洗、截断）
    invite / respond / sent      按编号邀请与回应
    stats()                      自检用
"""
from __future__ import annotations

import atexit
import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_MESSAGES = 400          # 单间消息上限
MAX_TEXT = 600              # 单条消息最长字符数
MIN_GAP = 1.2               # 同一游客两条消息的最小间隔（秒）
ONLINE_TTL = 75.0           # 多久没动静算离线
MEMBER_TTL = 1800.0         # 多久没动静从成员表里清掉
MAX_MEMBERS = 200           # 单间成员表上限（超了丢最久没说话的）
INVITE_TTL = 600.0          # 邀请多久内有效（秒）
MAX_INVITES = 20            # 每人最多攒多少条未看邀请
MAX_INVITES_PER_MIN = 5     # 同一个人一分钟内最多发几条邀请（防刷编号）
MAX_ASK = 120               # 房间里一条提问最多多少字
MAX_BOT_TEXT = 1200         # 问答回复落进消息的字数上限（超出截断）
ASK_GAP = 12.0              # 同一游客两次提问的最小间隔（跑一次 RAG 要十几秒）
ASK_PER_5MIN = 3            # 五分钟内最多问几次
MAX_ROOMS = 40              # 游客自建房间总数上限
MAX_ROOMS_PER_OWNER = 3     # 一个人最多开几间
MAX_NAME = 16               # 房间名字数上限
MAX_DESC = 40               # 简介字数上限

# 落盘：data/chat_rooms.json（本机运行时数据，已在 .gitignore 里）
MAX_NOTICE = 300             # 公告字数上限
MAX_FILES = 12               # 单间文件数上限
MAX_FILE_BYTES = 5 * 1024 * 1024   # 单个文件 5 MB 上限

MAX_NOTICE_HIST = 12           # 公告最多留几版旧稿
MAX_ADMINS = 5                 # 单间最多几位管理员
MAX_MUTE_MIN = 1440            # 禁言最长 24 小时
MAX_READ_GUESTS = 300          # 已读游标只记最近这些人，免得越攒越长
FILES_DIR = Path(__file__).resolve().parent.parent / "data" / "room_files"

STORE_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_rooms.json"
SAVE_INTERVAL = 2.0         # 有改动后最多等这么久写一次
STORE_VERSION = 1

STATUS_TEXT = {"pending": "待回应", "accepted": "已接受",
               "declined": "已拒绝", "ignored": "已忽略"}

_GUEST_RE = re.compile(r"^\d{5}$")
_ID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"   # 去掉 0/o/1/l/i 之类易混字符

# 官方聊天室：id、名称、主题、一句话简介
OFFICIAL = (
    ("o-jianbie", "药材鉴别 · 眼观手摸", "药材鉴别",
     "看形色、闻气味、尝味道，说说你手上那一味。"),
    ("o-peiwu", "方剂配伍 · 君臣佐使", "方剂配伍",
     "谁为君、谁为臣，一张方子里的分寸。"),
    ("o-paozhi", "炮制火候 · 酒蒸醋炙", "炮制",
     "炒焦、酒蒸、醋炙，火候差一分，药性差一分。"),
    ("o-siqi", "四气五味 · 升降浮沉", "药性",
     "寒热温凉、酸苦甘辛咸，把药性说清楚。"),
)

_lock = threading.RLock()
_rooms: Dict[str, Dict[str, Any]] = {}                 # id -> 房间信息
_msgs: Dict[str, List[Dict[str, Any]]] = {}            # id -> 消息（seq 升序）
_members: Dict[str, Dict[str, Dict[str, Any]]] = {}    # id -> 游客 -> {name, at, first, count}
_seq: Dict[str, int] = {}                              # id -> 最新序号
_last_post: Dict[str, float] = {}                      # "房间|游客" -> 上次发言时间
_invites: Dict[str, List[Dict[str, Any]]] = {}         # 被邀请人 -> 邀请
_results: Dict[str, List[Dict[str, Any]]] = {}         # 邀请人 -> 处理结果
_inv_seq = 0
_inv_sent: Dict[str, List[float]] = {}                  # 邀请人 -> 最近发送时间（限速用）
_asks: Dict[str, List[float]] = {}                      # 提问人 -> 最近提问时间
_notices: Dict[str, Dict[str, Any]] = {}                 # 房间 -> 公告
_files: Dict[str, List[Dict[str, Any]]] = {}             # 房间 -> 文件清单（正文另存磁盘）
_file_seq = 0
_roles: Dict[str, Dict[str, str]] = {}       # 房间 -> 游客 -> "admin"（房主记在房间上）
_mutes: Dict[str, Dict[str, float]] = {}     # 房间 -> 游客 -> 禁言到什么时候（0 = 长期）
_kicked: Dict[str, Dict[str, float]] = {}    # 房间 -> 游客 -> 被移出的时间
_notice_hist: Dict[str, List[Dict[str, Any]]] = {}   # 房间 -> 旧公告（新的在前）
_reads: Dict[str, Dict[str, int]] = {}       # 游客 -> 房间 -> 已读到的 seq（红点用）

_dirty = False                                         # 有未落盘的改动
_last_saved = 0.0                                      # 上次成功落盘的时间
_LOAD_INFO: Dict[str, Any] = {}                        # 启动时读档结果（自检用）


def _boot() -> None:
    for rid, name, topic, desc in OFFICIAL:
        _rooms[rid] = {"id": rid, "name": name, "kind": "official", "topic": topic,
                       "desc": desc, "owner": "", "at": 0.0}
        _msgs[rid] = []
        _members[rid] = {}
        _seq[rid] = 0
        _files[rid] = []


# ---------------------------------------------------------------- 落盘
def _mark_dirty() -> None:
    """标记有改动，交给后台线程写；调用方已持锁，这里只翻个标志位。"""
    global _dirty
    _dirty = True


def _dump_locked() -> Dict[str, Any]:
    """把需要长期保留的东西打包成 dict。调用方已持锁，这里只读。"""
    rooms = [dict(r) for r in _rooms.values() if r.get("kind") == "guest"]
    rooms.sort(key=lambda r: float(r.get("at") or 0))
    msgs = {rid: list(v) for rid, v in _msgs.items() if rid in _rooms}
    seq = {rid: int(v) for rid, v in _seq.items() if rid in _rooms}
    notices = {rid: dict(v) for rid, v in _notices.items() if rid in _rooms and v}
    files = {rid: list(v) for rid, v in _files.items() if rid in _rooms and v}
    roles = {rid: dict(v) for rid, v in _roles.items() if rid in _rooms and v}
    mutes = {rid: dict(v) for rid, v in _mutes.items() if rid in _rooms and v}
    kicked = {rid: dict(v) for rid, v in _kicked.items() if rid in _rooms and v}
    hist = {rid: list(v) for rid, v in _notice_hist.items() if rid in _rooms and v}
    reads = {g: dict(v) for g, v in _reads.items() if v}
    return {"version": STORE_VERSION, "saved_at": time.time(),
            "rooms": rooms, "messages": msgs, "seq": seq,
            "notices": notices, "files": files,
            "roles": roles, "mutes": mutes, "kicked": kicked,
            "notice_hist": hist, "reads": reads}


def _write_file(payload: Dict[str, Any]) -> bool:
    """原子写：先写 .tmp 再 replace，写坏了也不会留下半个文件。"""
    tmp = STORE_FILE.parent / (STORE_FILE.name + ".tmp")
    try:
        STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, STORE_FILE)
        return True
    except Exception as e:                       # 磁盘满、没权限之类：不能把服务拖死
        print("  [聊天室] 落盘失败：%s" % e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def flush() -> bool:
    """立刻把改动写下去。返回是否真的写了。"""
    global _dirty, _last_saved
    with _lock:
        if not _dirty:
            return False
        payload = _dump_locked()
        _dirty = False
    if _write_file(payload):
        _last_saved = float(payload.get("saved_at") or time.time())
        return True
    with _lock:                                  # 写失败：下次再来
        globals()["_dirty"] = True
    return False


def _saver_loop() -> None:
    while True:
        time.sleep(SAVE_INTERVAL)
        try:
            flush()
        except Exception as e:
            print("  [聊天室] 后台落盘出错：%s" % e)


def _load() -> Dict[str, Any]:
    """读档：只恢复游客房 + 消息 + 序号。坏文件备份后照常启动。"""
    info: Dict[str, Any] = {"file": str(STORE_FILE), "ok": False,
                            "rooms": 0, "messages": 0, "error": ""}
    if not STORE_FILE.exists():
        return info
    try:
        raw = json.loads(STORE_FILE.read_text(encoding="utf-8-sig"))   # 容忍带 BOM 的文件
        if not isinstance(raw, dict):
            raise ValueError("顶层不是对象")
    except Exception as e:
        stamp = int(time.time())
        try:
            STORE_FILE.replace(STORE_FILE.parent / ("%s.bad-%d" % (STORE_FILE.name, stamp)))
        except Exception:
            pass
        info["error"] = "存档损坏，已另存为 .bad-%d：%s" % (stamp, e)
        print("  [聊天室] %s" % info["error"])
        return info

    rooms = raw.get("rooms") or []
    msgs = raw.get("messages") or {}
    seqs = raw.get("seq") or {}
    now = time.time()
    with _lock:
        for rid, n in (raw.get("notices") or {}).items():
            if rid not in _rooms or not isinstance(n, dict):
                continue
            body = _clean(n.get("text"))[:MAX_NOTICE]
            if body:
                _notices[rid] = {"text": body, "by": str(n.get("by") or ""),
                                 "by_name": _clean(n.get("by_name"))[:32],
                                 "at": float(n.get("at") or now)}
        for rid, lst in (raw.get("files") or {}).items():
            if rid not in _rooms or not isinstance(lst, list):
                continue
            keep: List[Dict[str, Any]] = []
            for f in lst[-MAX_FILES:]:
                if not isinstance(f, dict):
                    continue
                fid = str(f.get("id") or "")
                if not re.match(r"^f[0-9a-f]{5}$", fid):
                    continue
                if not (FILES_DIR / rid / fid).exists():   # 磁盘上没有就别报
                    continue
                keep.append({"id": fid, "name": _clean(f.get("name"))[:80],
                             "size": int(f.get("size") or 0),
                             "at": float(f.get("at") or now),
                             "by": str(f.get("by") or ""),
                             "by_name": _clean(f.get("by_name"))[:32]})
            if keep:
                _files[rid] = keep
        for r in rooms:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or "")
            if not rid or rid in _rooms or not re.match(r"^r[a-z0-9]{5}$", rid):
                continue
            owner = str(r.get("owner") or "")
            if not _GUEST_RE.match(owner):        # 没主人的房没人能删，不恢复
                continue
            name = _clean(r.get("name"))[:MAX_NAME] or ("聊天室 " + rid)
            _rooms[rid] = {"id": rid, "name": name, "kind": "guest", "topic": "",
                           "desc": _clean(r.get("desc"))[:MAX_DESC], "owner": owner,
                           "at": float(r.get("at") or now)}
            lst: List[Dict[str, Any]] = []
            for m in (msgs.get(rid) or [])[-MAX_MESSAGES:]:
                if not isinstance(m, dict):
                    continue
                try:
                    is_qa = m.get("kind") == "qa"
                    body = _clean(m.get("text"))[:MAX_BOT_TEXT if is_qa else MAX_TEXT]
                    if not body:
                        continue
                    gid = str(m.get("id") or "")
                    row = {"seq": int(m["seq"]), "id": gid,
                           "name": str(m.get("name") or name_of(gid)),
                           "text": body, "at": float(m.get("at") or 0)}
                    if is_qa:                     # 问答消息：提问人、问句、出处都留住
                        cites = []
                        for c in list(m.get("cites") or [])[:4]:
                            if not isinstance(c, dict):
                                continue
                            cites.append({"book": _clip(str(c.get("book") or ""), 40),
                                          "page_id": _clip(str(c.get("page_id") or ""), 40),
                                          "line": _clip(str(c.get("line") or ""), 24),
                                          "quote": _clip(str(c.get("quote") or ""), 140)})
                        row.update({"kind": "qa", "q": _clip(_clean(m.get("q")), MAX_ASK),
                                    "cites": cites, "refused": bool(m.get("refused")),
                                    "by": str(m.get("by") or ""),
                                    "by_name": str(m.get("by_name") or ""),
                                    "backend": _clip(str(m.get("backend") or ""), 24)})
                    lst.append(row)
                except Exception:
                    continue
            lst.sort(key=lambda m: m["seq"])
            _msgs[rid] = lst
            _members[rid] = {}
            # 序号只认实际落到消息里的最大值：手工改过存档也不会留下虚高的未读数
            _seq[rid] = max([m["seq"] for m in lst] + [0])
            info["rooms"] += 1
            info["messages"] += len(lst)

        for rid, book in (raw.get("roles") or {}).items():
            if rid not in _rooms or not isinstance(book, dict):
                continue
            keep = {str(g): "admin" for g, r in book.items()
                    if _GUEST_RE.match(str(g)) and str(r) == "admin"}
            keep = dict(list(keep.items())[:MAX_ADMINS])
            if keep:
                _roles[rid] = keep
        for rid, book in (raw.get("mutes") or {}).items():
            if rid not in _rooms or not isinstance(book, dict):
                continue
            keep = {}
            for g, v in book.items():
                if not _GUEST_RE.match(str(g)):
                    continue
                try:
                    until = float(v or 0)
                except (TypeError, ValueError):
                    continue
                if until == 0 or until > now:
                    keep[str(g)] = until
            if keep:
                _mutes[rid] = keep
        for rid, book in (raw.get("kicked") or {}).items():
            if rid not in _rooms or not isinstance(book, dict):
                continue
            keep = {str(g): float(v or now) for g, v in book.items()
                    if _GUEST_RE.match(str(g))}
            if keep:
                _kicked[rid] = keep
        for rid, lst in (raw.get("notice_hist") or {}).items():
            if rid not in _rooms or not isinstance(lst, list):
                continue
            keep = []
            for n in lst[-MAX_NOTICE_HIST:]:
                if not isinstance(n, dict):
                    continue
                body = _clean(n.get("text"))[:MAX_NOTICE]
                if not body:
                    continue
                keep.append({"text": body, "by": str(n.get("by") or ""),
                             "by_name": _clean(n.get("by_name"))[:32],
                             "at": float(n.get("at") or 0),
                             "replaced_at": float(n.get("replaced_at") or 0)})
            if keep:
                _notice_hist[rid] = keep
        for g, book in (raw.get("reads") or {}).items():
            if not _GUEST_RE.match(str(g)) or not isinstance(book, dict):
                continue
            keep = {}
            for rid, v in book.items():
                if rid in _rooms:
                    try:
                        keep[str(rid)] = int(v or 0)
                    except (TypeError, ValueError):
                        pass
            if keep:
                _reads[str(g)] = keep
        while len(_reads) > MAX_READ_GUESTS:
            _reads.pop(next(iter(_reads)), None)

        # 游客房超过上限：丢最久没动静的（和运行时策略一致）
        guests = sorted([r for r in _rooms.values() if r.get("kind") == "guest"],
                        key=lambda r: float(r.get("at") or 0))
        for room in guests[:max(0, len(guests) - MAX_ROOMS)]:
            rid = str(room["id"])
            for book in (_rooms, _msgs, _members, _seq):
                book.pop(rid, None)
            info["rooms"] -= 1
    info["ok"] = True
    return info


def _flush_at_exit() -> None:
    try:
        flush()
    except Exception:
        pass


class SlowDown(Exception):
    """发得太快。wait 是还差多少秒。"""

    def __init__(self, wait: float):
        self.wait = max(0.0, float(wait))
        super().__init__("slow down")


# ---------------------------------------------------------------- 基础
def name_of(guest_id: str) -> str:
    """服务端统一生成昵称，避免前端伪造名字。"""
    return "游客" + str(guest_id)


def _clean(text: Any) -> str:
    s = str(text or "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)   # 去控制字符，保留 \n
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def lan_ip() -> str:
    """本机在局域网里的地址（邀请链接要用）。取不到返回空串。"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))       # 只为让系统选出口网卡，不发包
            ip = s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""
    return "" if ip.startswith("127.") else ip


def _row(rid: str, now: float, gid: str = "") -> Dict[str, Any]:
    room = _rooms[rid]
    mem = _members.get(rid) or {}
    online = sum(1 for m in mem.values() if now - float(m.get("at") or 0) <= ONLINE_TTL)
    owner = str(room.get("owner") or "")
    row = {"id": rid, "name": room["name"], "kind": room["kind"],
           "topic": room.get("topic") or "", "desc": room.get("desc") or "",
           "owner": owner, "owner_name": name_of(owner) if owner else "",
           "at": room.get("at") or 0, "online": online,
           "members": len(mem), "messages": len(_msgs.get(rid) or []),
           "seq": _seq.get(rid, 0)}
    if gid:
        read = int((_reads.get(gid) or {}).get(rid) or 0)
        row["unread"] = max(0, int(row["seq"]) - read)
        row["role"] = _role_locked(rid, gid)
        row["muted"] = float((_mutes.get(rid) or {}).get(gid) or 0) > now
    return row


# ---------------------------------------------------------------- 房间
def list_rooms(guest_id: str = "") -> Dict[str, Any]:
    now = time.time()
    gid = str(guest_id or "").strip()
    if gid and not _GUEST_RE.match(gid):
        gid = ""
    with _lock:
        official = [_row(r[0], now, gid) for r in OFFICIAL if r[0] in _rooms]
        guest = [_row(rid, now, gid) for rid, r in _rooms.items() if r.get("kind") == "guest"]
        guest.sort(key=lambda x: float(x.get("at") or 0), reverse=True)
        return {"official": official, "guest": guest,
                "total": len(official) + len(guest),
                "max_rooms": MAX_ROOMS, "max_per_owner": MAX_ROOMS_PER_OWNER,
                "lan": lan_ip(), "server_time": now}


def _new_room_id() -> str:
    while True:
        rid = "r" + "".join(random.choice(_ID_ALPHABET) for _ in range(5))
        if rid not in _rooms:
            return rid


def _drop_oldest_guest() -> None:
    """游客房满了：先丢没有人在线、且最久没动静的那间。"""
    now = time.time()
    cand = []
    for rid, room in _rooms.items():
        if room.get("kind") != "guest":
            continue
        online = any(now - float(m.get("at") or 0) <= ONLINE_TTL
                     for m in (_members.get(rid) or {}).values())
        if not online:
            cand.append((float(room.get("at") or 0), rid))
    if not cand:
        raise ValueError("聊天室已满，等有人离开再开")
    cand.sort()
    rid = cand[0][1]
    for book in (_rooms, _msgs, _members, _seq, _roles, _mutes, _kicked, _notice_hist):
        book.pop(rid, None)
    _mark_dirty()


def create_room(name: str, desc: str = "", owner: str = "") -> Dict[str, Any]:
    """游客开一间。校验不过抛 ValueError。"""
    nm = _clean(name).replace("\n", " ")
    ds = _clean(desc).replace("\n", " ")
    own = str(owner or "").strip()
    if not _GUEST_RE.match(own):
        raise ValueError("身份无效")
    if len(nm) < 2:
        raise ValueError("名称至少两个字")
    if len(nm) > MAX_NAME:
        raise ValueError("名称最多 %d 个字" % MAX_NAME)
    if len(ds) > MAX_DESC:
        raise ValueError("简介最多 %d 个字" % MAX_DESC)
    now = time.time()
    with _lock:
        mine = [r for r in _rooms.values()
                if r.get("kind") == "guest" and str(r.get("owner") or "") == own]
        if len(mine) >= MAX_ROOMS_PER_OWNER:
            raise ValueError("一个人最多开 %d 间" % MAX_ROOMS_PER_OWNER)
        guests = [r for r in _rooms.values() if r.get("kind") == "guest"]
        if len(guests) >= MAX_ROOMS:
            _drop_oldest_guest()
        rid = _new_room_id()
        _rooms[rid] = {"id": rid, "name": nm, "kind": "guest", "topic": "",
                       "desc": ds, "owner": own, "at": now}
        _msgs[rid] = []
        _members[rid] = {}
        _seq[rid] = 0
        _mark_dirty()
        return _row(rid, now)


def delete_room(rid: str, owner: str) -> Dict[str, Any]:
    """创建者删掉自己的游客聊天室；官方房一律不许删。"""
    r = str(rid or "").strip()
    o = str(owner or "").strip()
    if not _GUEST_RE.match(o):
        raise ValueError("身份无效")
    with _lock:
        room = _rooms.get(r)
        if not room:
            raise ValueError("聊天室不存在")
        if room.get("kind") == "official":
            raise ValueError("官方聊天室不能删除")
        if room.get("owner") != o:
            raise ValueError("只有创建者能删除这间聊天室")
        _rooms.pop(r, None)
        _msgs.pop(r, None)
        _members.pop(r, None)
        _seq.pop(r, None)
        _notices.pop(r, None)
        _files.pop(r, None)
        _roles.pop(r, None)
        _mutes.pop(r, None)
        _kicked.pop(r, None)
        _notice_hist.pop(r, None)
        for book in _reads.values():
            book.pop(r, None)
        shutil.rmtree(FILES_DIR / r, ignore_errors=True)
        for k in [k for k in _last_post if str(k).startswith(r + "|")]:
            _last_post.pop(k, None)
        for k in list(_invites):                       # 值是一串邀请，不是单条
            keep = [x for x in (_invites.get(k) or []) if str(x.get("room")) != r]
            if keep:
                _invites[k] = keep
            else:
                _invites.pop(k, None)
        _mark_dirty()
    return {"ok": True, "id": r}


def room_meta(rid: str) -> Dict[str, Any]:
    key = str(rid or "").strip()
    now = time.time()
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        return _row(key, now)


# ---------------------------------------------------------------- 消息
def _prune(rid: str, now: float) -> None:
    msgs = _msgs.get(rid) or []
    if len(msgs) > MAX_MESSAGES:
        del msgs[: len(msgs) - MAX_MESSAGES]
    mem = _members.get(rid) or {}
    if len(mem) > MAX_MEMBERS:
        keep = sorted(mem.items(), key=lambda kv: float(kv[1].get("at") or 0),
                      reverse=True)[:MAX_MEMBERS]
        _members[rid] = dict(keep)
    for gid in [g for g, m in mem.items() if now - float(m.get("at") or 0) > MEMBER_TTL]:
        mem.pop(gid, None)


def post(room: str, guest_id: str, text: str, reply: Any = None) -> Dict[str, Any]:
    rid = str(room or "").strip()
    gid = str(guest_id or "").strip()
    if rid not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    body = _clean(text)
    if not body:
        raise ValueError("说点什么再发")
    if len(body) > MAX_TEXT:
        raise ValueError("一条最多 %d 字" % MAX_TEXT)
    now = time.time()
    with _lock:
        _assert_can_speak_locked(rid, gid)
        key = rid + "|" + gid
        last = float(_last_post.get(key) or 0)
        if now - last < MIN_GAP:
            raise SlowDown(MIN_GAP - (now - last))
        _last_post[key] = now
        _seq[rid] = int(_seq.get(rid) or 0) + 1
        msg = {"seq": _seq[rid], "id": gid, "name": name_of(gid), "text": body, "at": now}
        quoted = _reply_of_locked(rid, reply)
        if quoted:
            msg.update(quoted)
        mentioned = _mentions_locked(rid, body)
        if mentioned:
            msg["mentions"] = mentioned
        _msgs.setdefault(rid, []).append(msg)
        mem = _members.setdefault(rid, {})
        cur = dict(mem.get(gid) or {})
        cur["name"] = msg["name"]
        cur["at"] = now
        cur["first"] = float(cur.get("first") or now)
        cur["count"] = int(cur.get("count") or 0) + 1
        mem[gid] = cur
        _prune(rid, now)
        _mark_dirty()
        return msg


def _clip(text: str, limit: int) -> str:
    """按字截断，超出补省略号（不用于校验，只用于展示）。"""
    s = str(text or "").strip()
    return s if len(s) <= limit else s[: max(0, limit - 1)].rstrip() + "…"


def reserve_ask(guest_id: str) -> None:
    """提问前的闸门：太密就抛 ValueError。问答要跑本机模型，不能让一个人刷。"""
    gid = str(guest_id or "").strip()
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    now = time.time()
    with _lock:
        hist = [x for x in (_asks.get(gid) or []) if now - float(x) < 300.0]
        if hist and now - float(hist[-1]) < ASK_GAP:
            wait = ASK_GAP - (now - float(hist[-1]))
            raise ValueError("刚问过一条，%d 秒后再问" % max(1, int(wait + 0.999)))
        if len(hist) >= ASK_PER_5MIN:
            raise ValueError("五分钟内已经问了 %d 次，歇一会儿再问" % ASK_PER_5MIN)
        hist.append(now)
        _asks[gid] = hist[-10:]


def post_bot(room: str, question: str, text: str, cites: Optional[List[Dict[str, Any]]] = None,
             refused: bool = False, by: str = "", backend: str = "") -> Dict[str, Any]:
    """把一次问答的结果作为一条消息落进房间（id 固定 bot、kind=qa）。

    问答本身在 main.py 里跑（只有那里知道 RAG 与模型），这里只管落一条可溯源的消息。
    """
    rid = str(room or "").strip()
    if rid not in _rooms:
        raise ValueError("聊天室不存在")
    body = _clean(text)
    if not body:
        raise ValueError("答案是空的")
    clean = []
    for c in list(cites or [])[:4]:
        if not isinstance(c, dict):
            continue
        clean.append({
            "book": _clip(str(c.get("book") or ""), 40),
            "page_id": _clip(str(c.get("page_id") or ""), 40),
            "line": _clip(str(c.get("line") or ""), 24),
            "quote": _clip(str(c.get("quote") or ""), 140),
        })
    now = time.time()
    with _lock:
        _seq[rid] = int(_seq.get(rid) or 0) + 1
        msg = {"seq": _seq[rid], "id": "bot", "name": "AI 中医师",
               "text": _clip(body, MAX_BOT_TEXT), "at": now, "kind": "qa",
               "q": _clip(_clean(question), MAX_ASK), "cites": clean,
               "refused": bool(refused), "by": str(by or ""),
               "by_name": name_of(by) if _GUEST_RE.match(str(by or "")) else "",
               "backend": _clip(str(backend or ""), 24)}
        _msgs.setdefault(rid, []).append(msg)
        _prune(rid, now)
        _mark_dirty()
        return dict(msg)


# ---------------------------------------------------------------- 公告
def _notice_locked(rid: str) -> Dict[str, Any]:
    n = _notices.get(rid) or {}
    return {"text": n.get("text") or "", "by": n.get("by") or "",
            "by_name": n.get("by_name") or "", "at": float(n.get("at") or 0),
            "history": [dict(x) for x in (_notice_hist.get(rid) or [])]}


def _can_manage_locked(rid: str, gid: str) -> bool:
    """官方房没有房主，谁都能改公告（署名留痕）；游客房房主与管理员能改。"""
    room = _rooms.get(rid) or {}
    if not _GUEST_RE.match(str(gid or "")):
        return False
    if room.get("kind") == "official":
        return True
    return _role_locked(rid, str(gid or "")) in ("owner", "admin")


def notice(rid: str) -> Dict[str, Any]:
    """看公告：没设过就返回空文本。"""
    key = str(rid or "").strip()
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        return _notice_locked(key)


def can_manage(rid: str, guest_id: str) -> bool:
    key = str(rid or "").strip()
    with _lock:
        if key not in _rooms:
            return False
        return _can_manage_locked(key, str(guest_id or "").strip())


def set_notice(rid: str, guest_id: str, text: str) -> Dict[str, Any]:
    """设置 / 清空公告。传空字符串就是撤下公告。"""
    key = str(rid or "").strip()
    gid = str(guest_id or "").strip()
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    body = _clean(text)
    if len(body) > MAX_NOTICE:
        raise ValueError("公告最多 %d 字" % MAX_NOTICE)
    now = time.time()
    with _lock:
        if not _can_manage_locked(key, gid):
            raise ValueError("这一间只有房主或管理员能改公告")
        old = _notices.get(key) or {}
        if old.get("text"):                       # 换稿前把旧的压进历史
            hist = _notice_hist.setdefault(key, [])
            hist.insert(0, {"text": old["text"], "by": old.get("by") or "",
                            "by_name": old.get("by_name") or "",
                            "at": float(old.get("at") or 0), "replaced_at": now})
            del hist[MAX_NOTICE_HIST:]
        if not body:
            _notices.pop(key, None)
            _mark_dirty()
            return {"text": "", "by": gid, "by_name": name_of(gid), "at": now,
                    "cleared": True,
                    "history": [dict(x) for x in (_notice_hist.get(key) or [])]}
        _notices[key] = {"text": body, "by": gid, "by_name": name_of(gid), "at": now}
        _mark_dirty()
        out = dict(_notices[key])
        out["cleared"] = False
        out["history"] = [dict(x) for x in (_notice_hist.get(key) or [])]
        return out


# ---------------------------------------------------------------- 文件
def list_files(rid: str) -> Dict[str, Any]:
    key = str(rid or "").strip()
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        rows = [dict(f) for f in (_files.get(key) or [])]
    rows.sort(key=lambda f: float(f.get("at") or 0), reverse=True)
    return {"room": key, "files": rows, "max_files": MAX_FILES,
            "max_bytes": MAX_FILE_BYTES}


def save_file(rid: str, guest_id: str, name: str, data: bytes) -> Dict[str, Any]:
    """收一个文件：正文落磁盘 + 清单里记一条。校验不过抛 ValueError。"""
    global _file_seq
    key = str(rid or "").strip()
    gid = str(guest_id or "").strip()
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    raw_name = str(name or "").replace("\\", "/").split("/")[-1].strip()
    if not raw_name:
        raise ValueError("文件名为空")
    clean_name = _clean(raw_name)[:80]
    blob = bytes(data or b"")
    if not blob:
        raise ValueError("文件是空的")
    if len(blob) > MAX_FILE_BYTES:
        raise ValueError("文件超过 %d MB" % (MAX_FILE_BYTES // 1024 // 1024))
    now = time.time()
    with _lock:
        rows = _files.setdefault(key, [])
        if len(rows) >= MAX_FILES:
            raise ValueError("这间房最多存 %d 个文件" % MAX_FILES)
        _file_seq += 1
        fid = "f%05x" % (_file_seq & 0xFFFFF)
        folder = FILES_DIR / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / fid).write_bytes(blob)
        row = {"id": fid, "name": clean_name, "size": len(blob), "at": now,
               "by": gid, "by_name": name_of(gid)}
        rows.append(row)
        _mark_dirty()
        return dict(row)


def file_path(rid: str, fid: str) -> Dict[str, Any]:
    """下载用的磁盘路径 + 原始文件名。"""
    key = str(rid or "").strip()
    f = str(fid or "").strip()
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        hit = [x for x in (_files.get(key) or []) if str(x.get("id")) == f]
        if not hit:
            raise ValueError("文件不存在")
        path = FILES_DIR / key / f
        if not path.exists():
            raise ValueError("文件已被清理")
        return {"path": path, "name": hit[0].get("name") or f,
                "size": int(hit[0].get("size") or 0)}


# ---------------------------------------------------------------- 搜索
def search(rid: str, q: str, limit: int = 40) -> Dict[str, Any]:
    """在房间消息里找字：不区分大小写，新的排前面。"""
    key = str(rid or "").strip()
    needle = str(q or "").strip().lower()
    if not needle:
        raise ValueError("先写要搜的词")
    if len(needle) > 30:
        raise ValueError("搜索词太长（30 字以内）")
    cap = max(1, min(80, int(limit or 40)))
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        rows = []
        for m in reversed(_msgs.get(key) or []):
            hay = (str(m.get("text") or "") + " " + str(m.get("q") or "")).lower()
            if needle in hay:
                rows.append({"seq": m.get("seq"), "id": m.get("id"),
                             "name": m.get("name") or "", "at": m.get("at") or 0,
                             "text": m.get("text") or "",
                             "kind": m.get("kind") or "", "q": m.get("q") or ""})
                if len(rows) >= cap:
                    break
    return {"q": q, "total": len(rows), "rows": rows}


# ---------------------------------------------------------------- 拉取
def _take_invites(gid: str, now: float, all_pending: bool = False) -> List[Dict[str, Any]]:
    """取出还没处理、且还没提示过的邀请。调用方已持锁。"""
    out = []
    for x in (_invites.get(gid) or []):
        if now - float(x.get("at") or 0) > INVITE_TTL:
            continue
        if str(x.get("status")) != "pending":
            continue
        if x.get("delivered") and not all_pending:
            continue
        x["delivered"] = True
        rid = str(x.get("room") or "")
        out.append({"id": x.get("id"), "from": x.get("from"),
                    "name": name_of(str(x.get("from"))),
                    "room": rid, "room_name": (_rooms.get(rid) or {}).get("name") or "",
                    "at": x.get("at")})
    return out


def inbox(gid: str) -> Dict[str, Any]:
    """只取邀请与处理结果，不登记在场、不碰任何房间。

    给「还在列表页、没进任何房间」的人用，否则他收不到邀请。
    """
    g = str(gid or "").strip()
    if not _GUEST_RE.match(g):
        raise ValueError("身份无效")
    now = time.time()
    with _lock:
        return {"invites": _take_invites(g, now), "results": _results.pop(g, []),
                "server_time": now}


def snapshot(room: str = "", since: int = 0, guest_id: str = "") -> Dict[str, Any]:
    rid = str(room or "").strip()
    if rid not in _rooms:
        raise ValueError("聊天室不存在")
    gid = str(guest_id or "").strip()
    now = time.time()
    with _lock:
        latest = int(_seq.get(rid) or 0)
        cursor = max(0, int(since or 0))
        if cursor > latest:            # 服务重启：序号归零，客户端从头同步
            cursor = 0
        msgs = [m for m in (_msgs.get(rid) or []) if int(m["seq"]) > cursor]
        invites, results = [], []
        if _GUEST_RE.match(gid):
            mem = _members.setdefault(rid, {})
            cur = dict(mem.get(gid) or {})
            cur["name"] = name_of(gid)
            cur["at"] = now
            cur["first"] = float(cur.get("first") or now)
            cur["count"] = int(cur.get("count") or 0)
            mem[gid] = cur
            invites = _take_invites(gid, now, all_pending=(cursor == 0))
            results = _results.pop(gid, [])
            _prune(rid, now)
            book = _reads.setdefault(gid, {})
            book[rid] = latest                    # 正在看这一间：算已读
            if len(book) > MAX_ROOMS + 8:
                for stale in list(book)[: len(book) - MAX_ROOMS - 8]:
                    book.pop(stale, None)
        members = [
            {"id": g, "name": m.get("name") or name_of(g), "at": m.get("at") or 0,
             "first": float(m.get("first") or m.get("at") or 0),
             "count": int(m.get("count") or 0),
             "role": _role_locked(rid, g),
             "muted": float((_mutes.get(rid) or {}).get(g) or 0) > now,
             "online": (now - float(m.get("at") or 0)) <= ONLINE_TTL}
            for g, m in (_members.get(rid) or {}).items()
        ]
        members.sort(key=lambda m: (not m["online"], -float(m["at"])))
        me: Dict[str, Any] = {}
        if _GUEST_RE.match(gid):
            role = _role_locked(rid, gid)
            me = {"id": gid, "role": role,
                  "can_manage": _can_manage_locked(rid, gid),
                  "can_admin": role in ("owner", "admin"),
                  "official": (_rooms.get(rid) or {}).get("kind") == "official",
                  "muted_until": float((_mutes.get(rid) or {}).get(gid) or 0)}
        return {"room": rid, "seq": latest, "messages": msgs, "members": members,
                "total": len(_msgs.get(rid) or []),
                "online": sum(1 for m in members if m["online"]),
                "me": me,
                "invites": invites, "results": results, "server_time": now}


# ---------------------------------------------------------------- 邀请
def invite(from_id: str, to_id: str, room: str = "") -> Dict[str, Any]:
    """按编号邀请某位游客进指定聊天室。校验不过抛 ValueError。"""
    global _inv_seq
    fid = str(from_id or "").strip()
    tid = str(to_id or "").strip()
    rid = str(room or "").strip()
    if not _GUEST_RE.match(fid):
        raise ValueError("邀请人身份无效")
    if not _GUEST_RE.match(tid):
        raise ValueError("请输入五位游客编号")
    if fid == tid:
        raise ValueError("不能邀请自己")
    if rid not in _rooms:
        raise ValueError("聊天室不存在")
    now = time.time()
    with _lock:
        lst = [x for x in (_invites.get(tid) or [])
               if now - float(x.get("at") or 0) <= INVITE_TTL]
        if any(str(x.get("from")) == fid and str(x.get("room")) == rid
               and str(x.get("status")) == "pending" for x in lst):
            raise ValueError("已经邀请过 TA 了，等 TA 回应")
        hist = [x for x in (_inv_sent.get(fid) or []) if now - float(x) < 60.0]
        if len(hist) >= MAX_INVITES_PER_MIN:
            wait = 60.0 - (now - min(hist))
            raise ValueError("邀请发得太密，%d 秒后再试" % max(1, int(wait + 0.999)))
        hist.append(now)
        _inv_sent[fid] = hist[-(MAX_INVITES_PER_MIN * 2):]
        _inv_seq += 1
        inv = {"id": "i%d" % _inv_seq, "from": fid, "room": rid, "at": now,
               "status": "pending", "delivered": False}
        lst.append(inv)
        _invites[tid] = lst[-MAX_INVITES:]
    return {"id": inv["id"], "from": fid, "from_name": name_of(fid),
            "to": tid, "to_name": name_of(tid), "room": rid,
            "room_name": (_rooms.get(rid) or {}).get("name") or "", "at": now}


def respond(invite_id: str, guest_id: str, action: str) -> Dict[str, Any]:
    """被邀请人处理邀请：accept / decline / ignore。"""
    iid = str(invite_id or "").strip()
    gid = str(guest_id or "").strip()
    act = str(action or "").strip().lower()
    if act not in ("accept", "decline", "ignore"):
        raise ValueError("未知操作")
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    now = time.time()
    with _lock:
        hit = None
        for x in (_invites.get(gid) or []):
            if str(x.get("id")) == iid:
                hit = x
                break
        if hit is None:
            raise ValueError("邀请不存在或已失效")
        if str(hit.get("status")) != "pending":
            raise ValueError("这条邀请已经处理过了")
        status = {"accept": "accepted", "decline": "declined", "ignore": "ignored"}[act]
        hit["status"] = status
        rid = str(hit.get("room") or "")
        if status == "accepted":                  # 房主重新邀请：之前的「移出」作废
            (_kicked.get(rid) or {}).pop(gid, None)
        frm = str(hit.get("from"))
        _results.setdefault(frm, []).append({
            "id": iid, "by": gid, "by_name": name_of(gid), "room": rid,
            "room_name": (_rooms.get(rid) or {}).get("name") or "",
            "action": act, "status": status, "at": now})
        _results[frm] = _results[frm][-50:]
    return {"id": iid, "status": status, "status_text": STATUS_TEXT[status],
            "by": gid, "by_name": name_of(gid), "action": act, "room": rid}


def sent(gid: str, room: str = "") -> List[Dict[str, Any]]:
    """我发出的邀请。room 非空时只看这一间（聊天室里「审核邀请」用）。"""
    g = str(gid or "").strip()
    want = str(room or "").strip()
    if not _GUEST_RE.match(g):
        raise ValueError("身份无效")
    now = time.time()
    rows = []
    with _lock:
        for to_id, lst in _invites.items():
            for x in lst:
                if str(x.get("from")) != g:
                    continue
                rid = str(x.get("room") or "")
                if want and rid != want:
                    continue
                if now - float(x.get("at") or 0) > INVITE_TTL * 6:
                    continue
                st = str(x.get("status") or "pending")
                mem = (_members.get(rid) or {}).get(to_id) or {}
                rows.append({"id": x.get("id"), "to": to_id, "to_name": name_of(to_id),
                             "room": rid,
                             "room_name": (_rooms.get(rid) or {}).get("name") or "",
                             "status": st, "status_text": STATUS_TEXT.get(st, ""),
                             "joined": bool(mem),
                             "online": bool(mem) and
                             (now - float(mem.get("at") or 0)) <= ONLINE_TTL,
                             "at": x.get("at")})
    rows.sort(key=lambda r: float(r.get("at") or 0), reverse=True)
    return rows[:40]


def revoke(invite_id: str, owner: str) -> Dict[str, Any]:
    """撤回一条还没被回应的邀请。"""
    iid = str(invite_id or "").strip()
    own = str(owner or "").strip()
    if not _GUEST_RE.match(own):
        raise ValueError("身份无效")
    with _lock:
        for to_id in list(_invites):
            lst = _invites.get(to_id) or []
            for x in lst:
                if str(x.get("id")) != iid:
                    continue
                if str(x.get("from")) != own:
                    raise ValueError("只能撤回自己发出的邀请")
                if str(x.get("status")) != "pending":
                    raise ValueError("这条邀请已被回应，撤不回了")
                _invites[to_id] = [y for y in lst if str(y.get("id")) != iid]
                _mark_dirty()
                return {"ok": True, "id": iid, "to": to_id, "room": x.get("room")}
        raise ValueError("邀请不存在或已失效")


# ---------------------------------------------------------------- 角色与管理
def _role_locked(rid: str, gid: str) -> str:
    """这一间里 TA 的角色：owner / admin / 空（普通成员）。"""
    room = _rooms.get(rid) or {}
    g = str(gid or "")
    if room.get("kind") == "guest" and str(room.get("owner") or "") == g:
        return "owner"
    return str((_roles.get(rid) or {}).get(g) or "")


def _assert_can_speak_locked(rid: str, gid: str) -> None:
    """能不能开口：被移出的不行，被禁言的到点前不行。"""
    if (_kicked.get(rid) or {}).get(gid):
        raise ValueError("你已被移出这一间，让房主重新邀请你")
    book = _mutes.get(rid) or {}
    until = book.get(gid)
    if not until:
        return
    now = time.time()
    if until > now:
        raise ValueError("你被禁言了，还有 %d 分钟" % max(1, int((until - now) / 60 + 0.999)))
    book.pop(gid, None)                            # 到期自动解禁


def _reply_of_locked(rid: str, reply: Any) -> Dict[str, Any]:
    """引用：只认这一间里存在的 seq，摘一小段存进消息（原消息被清掉也还看得见）。"""
    try:
        seq = int(reply)
    except (TypeError, ValueError):
        return {}
    if seq <= 0:
        return {}
    for m in (_msgs.get(rid) or []):
        if int(m.get("seq") or 0) == seq:
            if m.get("kind") == "qa":
                who = "AI 中医师"
                body = str(m.get("q") or "")
            else:
                who = str(m.get("name") or "")
                body = str(m.get("text") or "")
            return {"reply": seq, "reply_name": who, "reply_text": _clip(body, 80)}
    return {}


def _mentions_locked(rid: str, body: str) -> List[str]:
    """@名字 → 游客编号。只认这一间在册的人，最多 8 个。"""
    mem = _members.get(rid) or {}
    out: List[str] = []
    for gid, m in mem.items():
        nm = str(m.get("name") or name_of(gid))
        if nm and ("@" + nm) in body and gid not in out:
            out.append(gid)
            if len(out) >= 8:
                break
    return out


def _manager_locked(rid: str, gid: str) -> str:
    """要动别人的人，起码得是房主或管理员。"""
    role = _role_locked(rid, gid)
    if role not in ("owner", "admin"):
        raise ValueError("这一间只有房主或管理员能管成员")
    return role


def set_role(rid: str, actor: str, target: str, role: str) -> Dict[str, Any]:
    """房主任命 / 撤销管理员。"""
    key = str(rid or "").strip()
    a = str(actor or "").strip()
    t = str(target or "").strip()
    want = str(role or "").strip().lower()
    if want not in ("admin", "member"):
        raise ValueError("角色只有管理员和普通成员")
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(a) or not _GUEST_RE.match(t):
        raise ValueError("身份无效")
    if a == t:
        raise ValueError("房主不用任命自己")
    with _lock:
        room = _rooms[key]
        if room.get("kind") != "guest":
            raise ValueError("官方房不做成员管理")
        if str(room.get("owner") or "") != a:
            raise ValueError("只有房主能任命管理员")
        if t not in (_members.get(key) or {}):
            raise ValueError("TA 还没进这一间")
        book = _roles.setdefault(key, {})
        if want == "admin":
            if t not in book and len(book) >= MAX_ADMINS:
                raise ValueError("管理员最多 %d 位" % MAX_ADMINS)
            book[t] = "admin"
        else:
            book.pop(t, None)
        if not book:
            _roles.pop(key, None)
        _mark_dirty()
        return {"ok": True, "room": key, "id": t, "name": name_of(t),
                "role": want, "role_text": "管理员" if want == "admin" else "成员"}


def mute(rid: str, actor: str, target: str, minutes: int = 10) -> Dict[str, Any]:
    """禁言 N 分钟；minutes=0 表示解除。"""
    key = str(rid or "").strip()
    a = str(actor or "").strip()
    t = str(target or "").strip()
    try:
        mins = int(minutes)
    except (TypeError, ValueError):
        raise ValueError("禁言时长要写分钟数")
    if mins < 0 or mins > MAX_MUTE_MIN:
        raise ValueError("禁言时长 0–%d 分钟" % MAX_MUTE_MIN)
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(a) or not _GUEST_RE.match(t):
        raise ValueError("身份无效")
    now = time.time()
    with _lock:
        if (_rooms[key] or {}).get("kind") != "guest":
            raise ValueError("官方房不做成员管理")
        actor_role = _manager_locked(key, a)
        if t == a:
            raise ValueError("不能对自己下手")
        if _role_locked(key, t) == "owner":
            raise ValueError("房主不能被禁言")
        if actor_role == "admin" and _role_locked(key, t) == "admin":
            raise ValueError("管理员之间不能互相禁言")
        book = _mutes.setdefault(key, {})
        if mins == 0:
            book.pop(t, None)
            until = 0.0
        else:
            until = now + mins * 60.0
            book[t] = until
        if not book:
            _mutes.pop(key, None)
        _mark_dirty()
        return {"ok": True, "room": key, "id": t, "name": name_of(t),
                "minutes": mins, "until": until,
                "text": "已解除禁言" if mins == 0 else "已禁言 %d 分钟" % mins}


def kick(rid: str, actor: str, target: str) -> Dict[str, Any]:
    """把一个人移出这一间（他再发言会被拒，重新邀请可解）。"""
    key = str(rid or "").strip()
    a = str(actor or "").strip()
    t = str(target or "").strip()
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(a) or not _GUEST_RE.match(t):
        raise ValueError("身份无效")
    now = time.time()
    with _lock:
        if (_rooms[key] or {}).get("kind") != "guest":
            raise ValueError("官方房不做成员管理")
        actor_role = _manager_locked(key, a)
        if t == a:
            raise ValueError("不能移出自己")
        if _role_locked(key, t) == "owner":
            raise ValueError("房主不能被移出")
        if actor_role == "admin" and _role_locked(key, t) == "admin":
            raise ValueError("管理员之间不能互相移出")
        if t not in (_members.get(key) or {}):
            raise ValueError("TA 不在这间房里")
        (_members.get(key) or {}).pop(t, None)
        (_roles.get(key) or {}).pop(t, None)
        (_mutes.get(key) or {}).pop(t, None)
        _kicked.setdefault(key, {})[t] = now
        _mark_dirty()
        return {"ok": True, "room": key, "id": t, "name": name_of(t),
                "text": "已移出 %s" % name_of(t)}


def roles_of(rid: str) -> Dict[str, Any]:
    """这一间的角色表（给界面画标签用）。"""
    key = str(rid or "").strip()
    with _lock:
        if key not in _rooms:
            raise ValueError("聊天室不存在")
        room = _rooms[key]
        return {"room": key, "kind": room.get("kind") or "",
                "owner": str(room.get("owner") or ""),
                "admins": list((_roles.get(key) or {}).keys()),
                "mutes": dict(_mutes.get(key) or {}),
                "kicked": dict(_kicked.get(key) or {})}


# ---------------------------------------------------------------- 导出
def export_markdown(rid: str, guest_id: str) -> Dict[str, Any]:
    """把这一间导成一份 Markdown：公告、成员、聊天记录、文件。"""
    key = str(rid or "").strip()
    gid = str(guest_id or "").strip()
    if key not in _rooms:
        raise ValueError("聊天室不存在")
    if not _GUEST_RE.match(gid):
        raise ValueError("身份无效")
    now = time.time()

    def ts(at: Any) -> str:
        try:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(at or 0)))
        except Exception:
            return ""

    with _lock:
        room = dict(_rooms[key])
        note = _notice_locked(key)
        mem = [{"id": g, "name": m.get("name") or name_of(g),
                "role": _role_locked(key, g)}
               for g, m in (_members.get(key) or {}).items()]
        msgs = [dict(m) for m in (_msgs.get(key) or [])]
        files = [dict(f) for f in (_files.get(key) or [])]
    mem.sort(key=lambda m: ({"owner": 0, "admin": 1}.get(m["role"], 2), m["id"]))
    files.sort(key=lambda f: float(f.get("at") or 0))

    kind = "官方房" if room.get("kind") == "official" else "游客房"
    out: List[str] = ["# " + str(room["name"]), ""]
    out.append("> %s · %d 条消息 · 导出人 游客%s · %s" % (kind, len(msgs), gid, ts(now)))
    if room.get("desc"):
        out += ["", "**简介**：" + str(room["desc"])]
    if note.get("text"):
        out += ["", "**公告**：" + note["text"] +
                "（%s · %s）" % (note.get("by_name") or note.get("by") or "?", ts(note.get("at")))]

    ROLE_TEXT = {"owner": "房主", "admin": "管理员"}
    out += ["", "## 成员（%d）" % len(mem)]
    for m in mem:
        out.append("- 游客%s（%s）" % (m["id"], ROLE_TEXT.get(m["role"], "成员")))
    if not mem:
        out.append("- （暂无登记）")

    out += ["", "## 聊天记录"]
    if not msgs:
        out.append("（还没有人开口）")
    for m in msgs:
        if m.get("kind") == "qa":
            out.append("")
            out.append("**AI 中医师** · %s" % ts(m.get("at")))
            if m.get("q"):
                out.append("> 问：%s" % m["q"])
            out.append("")
            out.append(str(m.get("text") or ""))
            for c in (m.get("cites") or []):
                cite = " · ".join([x for x in [
                    ("《" + c.get("book") + "》") if c.get("book") else "",
                    c.get("line") or "", c.get("page_id") or ""] if x])
                out.append("- 出处 %s" % cite)
                if c.get("quote"):
                    out.append("  > " + str(c["quote"]).replace("\n", " "))
            if m.get("by"):
                out.append("")
                out.append("（提问人：游客%s）" % m["by"])
            continue
        out.append("")
        out.append("**%s** · %s" % (m.get("name") or "", ts(m.get("at"))))
        if m.get("reply"):
            out.append("> 引用 %s：%s" % (m.get("reply_name") or "", m.get("reply_text") or ""))
        out.append("")
        out.append(str(m.get("text") or ""))

    if files:
        out += ["", "## 文件（%d）" % len(files)]
        for f in files:
            kb = max(1, int(round(int(f.get("size") or 0) / 1024.0)))
            out.append("- %s（%d KB，%s，%s）" % (f.get("name") or f.get("id"),
                                                kb, f.get("by_name") or "", ts(f.get("at"))))

    safe = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(room["name"]))[:40] or key
    name = "%s-%s.md" % (safe, time.strftime("%Y%m%d-%H%M"))
    return {"filename": name, "text": "\n".join(out).strip() + "\n", "messages": len(msgs)}


def stats() -> Dict[str, Any]:
    with _lock:
        return {"rooms": len(_rooms),
                "official": sum(1 for r in _rooms.values() if r.get("kind") == "official"),
                "guest_rooms": sum(1 for r in _rooms.values() if r.get("kind") == "guest"),
                "messages": sum(len(v) for v in _msgs.values()),
                "members": sum(len(v) for v in _members.values()),
                "notices": sum(1 for v in _notices.values() if v),
                "files": sum(len(v) for v in _files.values()),
                "admins": sum(len(v) for v in _roles.values()),
                "muted": sum(len(v) for v in _mutes.values()),
                "kicked": sum(len(v) for v in _kicked.values()),
                "notice_hist": sum(len(v) for v in _notice_hist.values()),
                "max_rooms": MAX_ROOMS, "max_text": MAX_TEXT, "min_gap": MIN_GAP,
                "invites_per_min": MAX_INVITES_PER_MIN,
                "ask_gap": ASK_GAP, "ask_per_5min": ASK_PER_5MIN,
                "online_ttl": ONLINE_TTL, "lan": lan_ip(),
                "backend": "json", "persist": True,
                "store": str(STORE_FILE), "saved_at": _last_saved,
                "loaded": dict(_LOAD_INFO), "dirty": _dirty}


# ---------------------------------------------------------------- 启动
_boot()                                   # 官方四间永远按代码为准
_LOAD_INFO = _load()                      # 再把落盘的游客房与消息接回来
threading.Thread(target=_saver_loop, daemon=True, name="chat-room-saver").start()
atexit.register(_flush_at_exit)
