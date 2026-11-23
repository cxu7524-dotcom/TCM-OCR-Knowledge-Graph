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
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

MAX_MESSAGES = 400          # 单间消息上限
MAX_TEXT = 600              # 单条消息最长字符数
MIN_GAP = 1.2               # 同一游客两条消息的最小间隔（秒）
ONLINE_TTL = 75.0           # 多久没动静算离线
MEMBER_TTL = 1800.0         # 多久没动静从成员表里清掉
MAX_MEMBERS = 200           # 单间成员表上限（超了丢最久没说话的）
INVITE_TTL = 600.0          # 邀请多久内有效（秒）
MAX_INVITES = 20            # 每人最多攒多少条未看邀请
MAX_ROOMS = 40              # 游客自建房间总数上限
MAX_ROOMS_PER_OWNER = 3     # 一个人最多开几间
MAX_NAME = 16               # 房间名字数上限
MAX_DESC = 40               # 简介字数上限

# 落盘：data/chat_rooms.json（本机运行时数据，已在 .gitignore 里）
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
    return {"version": STORE_VERSION, "saved_at": time.time(),
            "rooms": rooms, "messages": msgs, "seq": seq}


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
                    body = _clean(m.get("text"))[:MAX_TEXT]
                    if not body:
                        continue
                    gid = str(m.get("id") or "")
                    lst.append({"seq": int(m["seq"]), "id": gid,
                                "name": str(m.get("name") or name_of(gid)),
                                "text": body, "at": float(m.get("at") or 0)})
                except Exception:
                    continue
            lst.sort(key=lambda m: m["seq"])
            _msgs[rid] = lst
            _members[rid] = {}
            _seq[rid] = max([int(seqs.get(rid) or 0)] + [m["seq"] for m in lst] + [0])
            info["rooms"] += 1
            info["messages"] += len(lst)

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


def _row(rid: str, now: float) -> Dict[str, Any]:
    room = _rooms[rid]
    mem = _members.get(rid) or {}
    online = sum(1 for m in mem.values() if now - float(m.get("at") or 0) <= ONLINE_TTL)
    owner = str(room.get("owner") or "")
    return {"id": rid, "name": room["name"], "kind": room["kind"],
            "topic": room.get("topic") or "", "desc": room.get("desc") or "",
            "owner": owner, "owner_name": name_of(owner) if owner else "",
            "at": room.get("at") or 0, "online": online,
            "members": len(mem), "messages": len(_msgs.get(rid) or []),
            "seq": _seq.get(rid, 0)}


# ---------------------------------------------------------------- 房间
def list_rooms() -> Dict[str, Any]:
    now = time.time()
    with _lock:
        official = [_row(r[0], now) for r in OFFICIAL if r[0] in _rooms]
        guest = [_row(rid, now) for rid, r in _rooms.items() if r.get("kind") == "guest"]
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
    for book in (_rooms, _msgs, _members, _seq):
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
        for k in [k for k in _last_post if str(k).startswith(r + "|")]:
            _last_post.pop(k, None)
        for k in [k for k in _invites if (_invites.get(k) or {}).get("room") == r]:
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


def post(room: str, guest_id: str, text: str) -> Dict[str, Any]:
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
        key = rid + "|" + gid
        last = float(_last_post.get(key) or 0)
        if now - last < MIN_GAP:
            raise SlowDown(MIN_GAP - (now - last))
        _last_post[key] = now
        _seq[rid] = int(_seq.get(rid) or 0) + 1
        msg = {"seq": _seq[rid], "id": gid, "name": name_of(gid), "text": body, "at": now}
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
        members = [
            {"id": g, "name": m.get("name") or name_of(g), "at": m.get("at") or 0,
             "first": float(m.get("first") or m.get("at") or 0),
             "count": int(m.get("count") or 0),
             "online": (now - float(m.get("at") or 0)) <= ONLINE_TTL}
            for g, m in (_members.get(rid) or {}).items()
        ]
        members.sort(key=lambda m: (not m["online"], -float(m["at"])))
        return {"room": rid, "seq": latest, "messages": msgs, "members": members,
                "total": len(_msgs.get(rid) or []),
                "online": sum(1 for m in members if m["online"]),
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
        frm = str(hit.get("from"))
        _results.setdefault(frm, []).append({
            "id": iid, "by": gid, "by_name": name_of(gid), "room": rid,
            "room_name": (_rooms.get(rid) or {}).get("name") or "",
            "action": act, "status": status, "at": now})
        _results[frm] = _results[frm][-50:]
    return {"id": iid, "status": status, "status_text": STATUS_TEXT[status],
            "by": gid, "by_name": name_of(gid), "action": act, "room": rid}


def sent(gid: str) -> List[Dict[str, Any]]:
    """我发出的邀请（对话框里的记录）。"""
    g = str(gid or "").strip()
    if not _GUEST_RE.match(g):
        raise ValueError("身份无效")
    now = time.time()
    rows = []
    with _lock:
        for to_id, lst in _invites.items():
            for x in lst:
                if str(x.get("from")) != g:
                    continue
                if now - float(x.get("at") or 0) > INVITE_TTL * 6:
                    continue
                st = str(x.get("status") or "pending")
                rid = str(x.get("room") or "")
                rows.append({"id": x.get("id"), "to": to_id, "to_name": name_of(to_id),
                             "room": rid, "room_name": (_rooms.get(rid) or {}).get("name") or "",
                             "status": st, "status_text": STATUS_TEXT.get(st, ""),
                             "at": x.get("at")})
    rows.sort(key=lambda r: float(r.get("at") or 0), reverse=True)
    return rows[:20]


def stats() -> Dict[str, Any]:
    with _lock:
        return {"rooms": len(_rooms),
                "official": sum(1 for r in _rooms.values() if r.get("kind") == "official"),
                "guest_rooms": sum(1 for r in _rooms.values() if r.get("kind") == "guest"),
                "messages": sum(len(v) for v in _msgs.values()),
                "members": sum(len(v) for v in _members.values()),
                "max_rooms": MAX_ROOMS, "max_text": MAX_TEXT, "min_gap": MIN_GAP,
                "online_ttl": ONLINE_TTL, "lan": lan_ip(),
                "backend": "json", "persist": True,
                "store": str(STORE_FILE), "saved_at": _last_saved,
                "loaded": dict(_LOAD_INFO), "dirty": _dirty}


# ---------------------------------------------------------------- 启动
_boot()                                   # 官方四间永远按代码为准
_LOAD_INFO = _load()                      # 再把落盘的游客房与消息接回来
threading.Thread(target=_saver_loop, daemon=True, name="chat-room-saver").start()
atexit.register(_flush_at_exit)
