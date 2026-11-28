# -*- coding: utf-8 -*-
"""app/chat_room.py 的单元测试：房间、消息、邀请、落盘。

约定：
- 不碰真实的 data/chat_rooms.json —— 每个用例把 STORE_FILE 指到临时目录；
- 不依赖真实等待 —— 发言间隔（MIN_GAP）在 setUp 里调到 0，需要测限速时再单独开；
- 每个用例开始前把模块状态清成"只有官方四间"。
"""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from app import chat_room

ME = "41574"
OTHER = "31659"
THIRD = "58104"


def reset_state() -> None:
    """清空全部运行时状态，只留官方四间。调用方需持锁或独占。"""
    with chat_room._lock:
        for book in (chat_room._rooms, chat_room._msgs, chat_room._members,
                     chat_room._seq, chat_room._last_post, chat_room._invites,
                     chat_room._results, chat_room._inv_sent, chat_room._asks,
                     chat_room._roles, chat_room._mutes, chat_room._kicked,
                     chat_room._notice_hist, chat_room._reads):
            book.clear()
        chat_room._boot()
        chat_room._dirty = False


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tcm-chat-"))
        self.orig_store = chat_room.STORE_FILE
        self.orig_gap = chat_room.MIN_GAP
        chat_room.STORE_FILE = self.tmp / "chat_rooms.json"
        chat_room.MIN_GAP = 0.0
        reset_state()

    def tearDown(self):
        chat_room.MIN_GAP = self.orig_gap
        reset_state()                      # 先把 _dirty 归零，免得后台线程把这些写到真文件
        chat_room.STORE_FILE = self.orig_store
        shutil.rmtree(self.tmp, ignore_errors=True)

    # 小工具
    def make_room(self, name="测试间", owner=ME, desc=""):
        return chat_room.create_room(name, desc, owner)

    def make_room_with_msgs(self, texts=("一", "二"), owner=ME):
        rid = self.make_room(owner=owner)["id"]
        for t in texts:
            chat_room.post(rid, owner, t)
        return rid


class RoomsTest(Base):
    def test_boot_has_four_official_rooms(self):
        d = chat_room.list_rooms()
        self.assertEqual(len(d["official"]), 4)
        self.assertEqual(d["guest"], [])
        self.assertEqual([r["id"] for r in d["official"]],
                         [r[0] for r in chat_room.OFFICIAL])
        self.assertTrue(all(r["kind"] == "official" for r in d["official"]))

    def test_create_room_returns_row(self):
        row = self.make_room("炮制小间", desc="酒蒸醋炙")
        self.assertEqual(row["name"], "炮制小间")
        self.assertEqual(row["kind"], "guest")
        self.assertEqual(row["owner"], ME)
        self.assertEqual(row["owner_name"], "游客" + ME)
        self.assertEqual(row["messages"], 0)
        self.assertEqual(row["online"], 0)
        self.assertTrue(row["id"].startswith("r"))
        self.assertEqual(len(chat_room.list_rooms()["guest"]), 1)

    def test_create_room_validates(self):
        with self.assertRaises(ValueError):
            self.make_room("一")                      # 名字太短
        with self.assertRaises(ValueError):
            self.make_room("字" * (chat_room.MAX_NAME + 1))
        with self.assertRaises(ValueError):
            self.make_room("正常名字", desc="字" * (chat_room.MAX_DESC + 1))
        with self.assertRaises(ValueError):
            self.make_room("正常名字", owner="123")   # 不是五位
        with self.assertRaises(ValueError):
            self.make_room("正常名字", owner="")
        self.assertEqual(chat_room.list_rooms()["guest"], [])

    def test_name_and_desc_are_cleaned(self):
        row = self.make_room("  炮制\n小间  ", desc="酒蒸\x00醋炙")
        self.assertEqual(row["name"], "炮制 小间")
        self.assertEqual(row["desc"], "酒蒸醋炙")

    def test_one_owner_cannot_exceed_room_quota(self):
        for i in range(chat_room.MAX_ROOMS_PER_OWNER):
            self.make_room("第%d间" % (i + 1))
        with self.assertRaises(ValueError) as ctx:
            self.make_room("再来一间")
        self.assertIn("最多", str(ctx.exception))
        self.assertEqual(len(chat_room.list_rooms()["guest"]),
                         chat_room.MAX_ROOMS_PER_OWNER)

    def test_full_house_drops_idle_oldest_room(self):
        orig = chat_room.MAX_ROOMS
        chat_room.MAX_ROOMS = 2
        try:
            a = self.make_room("甲间", owner=ME)["id"]
            b = self.make_room("乙间", owner=OTHER)["id"]
            with chat_room._lock:            # 把甲间做成"更久没动静"
                chat_room._rooms[a]["at"] = time.time() - 3600
            c = self.make_room("丙间", owner=THIRD)["id"]
            ids = [r["id"] for r in chat_room.list_rooms()["guest"]]
            self.assertNotIn(a, ids)
            self.assertIn(b, ids)
            self.assertIn(c, ids)
        finally:
            chat_room.MAX_ROOMS = orig

    def test_delete_room_by_owner(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "说一句")
        self.assertEqual(chat_room.delete_room(rid, ME)["ok"], True)
        with self.assertRaises(ValueError):
            chat_room.room_meta(rid)
        self.assertNotIn(rid, chat_room._msgs)
        self.assertNotIn(rid, chat_room._seq)
        self.assertEqual(chat_room.list_rooms()["guest"], [])

    def test_delete_room_guards(self):
        official = chat_room.OFFICIAL[0][0]
        rid = self.make_room()["id"]
        with self.assertRaises(ValueError) as e1:
            chat_room.delete_room(official, ME)
        self.assertIn("官方", str(e1.exception))
        with self.assertRaises(ValueError) as e2:
            chat_room.delete_room(rid, OTHER)
        self.assertIn("创建者", str(e2.exception))
        with self.assertRaises(ValueError):
            chat_room.delete_room(rid, "123")
        with self.assertRaises(ValueError):
            chat_room.delete_room("rxxxxx", ME)
        self.assertIn(rid, chat_room._rooms)          # 都失败了，房还在

    def test_delete_room_drops_pending_invites_to_that_room(self):
        rid = self.make_room()["id"]
        chat_room.invite(ME, OTHER, rid)
        self.assertTrue(chat_room._invites.get(OTHER))
        chat_room.delete_room(rid, ME)
        self.assertFalse([x for x in (chat_room._invites.get(OTHER) or [])
                          if x.get("room") == rid])


class MessagesTest(Base):
    def test_post_and_snapshot(self):
        rid = self.make_room()["id"]
        m1 = chat_room.post(rid, ME, "第一句")
        m2 = chat_room.post(rid, OTHER, "第二句")
        self.assertEqual([m1["seq"], m2["seq"]], [1, 2])
        self.assertEqual(m1["name"], "游客" + ME)
        snap = chat_room.snapshot(rid, 0, ME)
        self.assertEqual([m["text"] for m in snap["messages"]], ["第一句", "第二句"])
        self.assertEqual(snap["seq"], 2)
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["online"], 2)           # 两人都发过言，75 秒内算在线

    def test_snapshot_since_only_newer(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        chat_room.post(rid, ME, "二")
        snap = chat_room.snapshot(rid, 1, ME)
        self.assertEqual([m["text"] for m in snap["messages"]], ["二"])

    def test_snapshot_moves_cursor_back_when_seq_reset(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        snap = chat_room.snapshot(rid, 99, ME)        # 客户端游标比服务端大
        self.assertEqual(len(snap["messages"]), 1)
        self.assertEqual(snap["seq"], 1)

    def test_post_validates(self):
        rid = self.make_room()["id"]
        for bad in ("", "   ", "\x00\x01"):
            with self.assertRaises(ValueError):
                chat_room.post(rid, ME, bad)
        with self.assertRaises(ValueError):
            chat_room.post(rid, ME, "字" * (chat_room.MAX_TEXT + 1))
        with self.assertRaises(ValueError):
            chat_room.post(rid, "123", "身份不对")
        with self.assertRaises(ValueError):
            chat_room.post("rnope1", ME, "房不存在")
        self.assertEqual(chat_room.list_rooms()["guest"][0]["messages"], 0)

    def test_post_cleans_control_chars(self):
        rid = self.make_room()["id"]
        msg = chat_room.post(rid, ME, "  上\n\n\n\n下\x07  ")
        self.assertEqual(msg["text"], "上\n\n下")

    def test_rate_limit_raises_slowdown(self):
        rid = self.make_room()["id"]
        chat_room.MIN_GAP = 1.2
        chat_room.post(rid, ME, "一")
        with self.assertRaises(chat_room.SlowDown) as ctx:
            chat_room.post(rid, ME, "二")
        self.assertGreater(ctx.exception.wait, 0)
        chat_room.post(rid, OTHER, "别人不受影响")    # 限速是按人算的
        self.assertEqual(chat_room.snapshot(rid, 0, ME)["total"], 2)

    def test_message_cap_keeps_newest(self):
        orig = chat_room.MAX_MESSAGES
        chat_room.MAX_MESSAGES = 3
        try:
            rid = self.make_room()["id"]
            for i in range(5):
                chat_room.post(rid, ME, "第%d句" % (i + 1))
            snap = chat_room.snapshot(rid, 0, ME)
            self.assertEqual([m["text"] for m in snap["messages"]],
                             ["第3句", "第4句", "第5句"])
            self.assertEqual(snap["seq"], 5)
        finally:
            chat_room.MAX_MESSAGES = orig

    def test_members_carry_online_flag(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "在吗")
        snap = chat_room.snapshot(rid, 0, OTHER)      # OTHER 露个面
        by_id = {m["id"]: m for m in snap["members"]}
        self.assertIn(ME, by_id)
        self.assertIn(OTHER, by_id)
        self.assertTrue(by_id[OTHER]["online"])
        with chat_room._lock:                         # 把 ME 弄成很久以前
            chat_room._members[rid][ME]["at"] = time.time() - 300   # 离线但还没被清
        rows = {m["id"]: m for m in chat_room.snapshot(rid, 0, OTHER)["members"]}
        self.assertFalse(rows[ME]["online"])
        self.assertTrue(rows[OTHER]["online"])

    def test_unknown_room_snapshot(self):
        with self.assertRaises(ValueError):
            chat_room.snapshot("rnope1", 0, ME)


class InvitesTest(Base):
    def test_invite_then_inbox(self):
        rid = self.make_room()["id"]
        got = chat_room.invite(ME, OTHER, rid)
        self.assertEqual(got["from"], ME)
        self.assertEqual(got["to"], OTHER)
        self.assertEqual(got["from_name"], "游客" + ME)
        box = chat_room.inbox(OTHER)
        self.assertEqual(len(box["invites"]), 1)
        inv = box["invites"][0]
        self.assertEqual(inv["name"], "游客" + ME)
        self.assertEqual(inv["room_name"], "测试间")
        self.assertEqual(chat_room.inbox(OTHER)["invites"], [])   # 只提示一次

    def test_pending_invite_returns_again_on_first_load(self):
        rid = self.make_room()["id"]
        chat_room.invite(ME, OTHER, rid)
        chat_room.inbox(OTHER)
        again = chat_room.snapshot(rid, 0, OTHER)     # 首次进房间（cursor=0）
        self.assertEqual(len(again["invites"]), 1)

    def test_invite_validates(self):
        rid = self.make_room()["id"]
        with self.assertRaises(ValueError):
            chat_room.invite(ME, ME, rid)             # 自己邀自己
        with self.assertRaises(ValueError):
            chat_room.invite("123", OTHER, rid)
        with self.assertRaises(ValueError):
            chat_room.invite(ME, "abcde", rid)
        with self.assertRaises(ValueError):
            chat_room.invite(ME, OTHER, "rnope1")
        chat_room.invite(ME, OTHER, rid)
        with self.assertRaises(ValueError) as ctx:    # 重复邀请
            chat_room.invite(ME, OTHER, rid)
        self.assertIn("已经邀请", str(ctx.exception))

    def test_respond_accept_notifies_sender(self):
        rid = self.make_room()["id"]
        inv = chat_room.invite(ME, OTHER, rid)
        out = chat_room.respond(inv["id"], OTHER, "accept")
        self.assertEqual(out["status"], "accepted")
        self.assertEqual(out["status_text"], "已接受")
        self.assertEqual(chat_room.inbox(OTHER)["invites"], [])   # 处理完不再提示
        results = chat_room.inbox(ME)["results"]
        self.assertEqual(results[0]["action"], "accept")
        self.assertEqual(results[0]["by_name"], "游客" + OTHER)
        self.assertEqual(chat_room.inbox(ME)["results"], [])      # 结果也只给一次

    def test_respond_twice_and_unknown(self):
        rid = self.make_room()["id"]
        inv = chat_room.invite(ME, OTHER, rid)
        chat_room.respond(inv["id"], OTHER, "decline")
        with self.assertRaises(ValueError):
            chat_room.respond(inv["id"], OTHER, "accept")
        with self.assertRaises(ValueError):
            chat_room.respond("i9999", OTHER, "accept")
        with self.assertRaises(ValueError):
            chat_room.respond(inv["id"], OTHER, "随便")     # 未知动作

    def test_sent_lists_only_my_invites(self):
        rid = self.make_room()["id"]
        chat_room.invite(ME, OTHER, rid)
        chat_room.invite(THIRD, OTHER, rid)
        mine = chat_room.sent(ME)
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["to"], OTHER)
        self.assertEqual(mine[0]["status"], "pending")
        self.assertEqual(mine[0]["status_text"], "待回应")
        self.assertEqual(chat_room.sent(OTHER), [])

    def test_invite_rate_limit(self):
        rid = self.make_room()["id"]
        for _ in range(chat_room.MAX_INVITES_PER_MIN):
            chat_room.invite(ME, OTHER, rid)
            chat_room._invites.get(OTHER, []).clear()      # 清掉重复判定，只测限速
        with self.assertRaises(ValueError) as ctx:
            chat_room.invite(ME, OTHER, rid)
        self.assertIn("太密", str(ctx.exception))
        chat_room.invite(THIRD, OTHER, rid)                # 别人不受影响

    def test_invite_expires(self):
        rid = self.make_room()["id"]
        chat_room.invite(ME, OTHER, rid)
        with chat_room._lock:
            chat_room._invites[OTHER][0]["at"] = time.time() - chat_room.INVITE_TTL - 5
        self.assertEqual(chat_room.inbox(OTHER)["invites"], [])


class PersistenceTest(Base):
    def test_flush_writes_and_load_restores(self):
        rid = self.make_room("落盘间", desc="重启还在")["id"]
        chat_room.post(rid, ME, "一")
        chat_room.post(rid, ME, "二")
        self.assertTrue(chat_room.flush())
        self.assertTrue(chat_room.STORE_FILE.exists())
        raw = json.loads(chat_room.STORE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], chat_room.STORE_VERSION)
        self.assertEqual([r["name"] for r in raw["rooms"]], ["落盘间"])

        reset_state()                                  # 模拟重启
        info = chat_room._load()
        self.assertTrue(info["ok"])
        self.assertEqual((info["rooms"], info["messages"]), (1, 2))
        rooms = chat_room.list_rooms()["guest"]
        self.assertEqual(rooms[0]["id"], rid)
        self.assertEqual(rooms[0]["name"], "落盘间")
        self.assertEqual(rooms[0]["messages"], 2)
        snap = chat_room.snapshot(rid, 0, ME)
        self.assertEqual([m["text"] for m in snap["messages"]], ["一", "二"])
        self.assertEqual(snap["seq"], 2)

    def test_flush_is_noop_when_clean(self):
        self.assertFalse(chat_room.flush())

    def test_delete_is_persisted(self):
        rid = self.make_room()["id"]
        chat_room.flush()
        chat_room.delete_room(rid, ME)
        chat_room.flush()
        reset_state()
        chat_room._load()
        self.assertEqual(chat_room.list_rooms()["guest"], [])

    def test_load_skips_rooms_without_valid_owner(self):
        payload = {"version": 1, "rooms": [
            {"id": "rabcde", "name": "没人认领", "owner": "", "at": 1.0},
            {"id": "rbcdef", "name": "正常", "owner": ME, "at": 2.0},
        ], "messages": {}, "seq": {}}
        chat_room.STORE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        info = chat_room._load()
        self.assertEqual(info["rooms"], 1)
        self.assertEqual([r["name"] for r in chat_room.list_rooms()["guest"]], ["正常"])

    def test_load_caps_messages_and_fixes_seq(self):
        orig = chat_room.MAX_MESSAGES
        chat_room.MAX_MESSAGES = 2
        try:
            msgs = [{"seq": i, "id": ME, "name": "游客" + ME, "text": "第%d" % i, "at": 1.0}
                    for i in range(1, 5)]
            payload = {"version": 1,
                       "rooms": [{"id": "rabcde", "name": "旧间", "owner": ME, "at": 1.0}],
                       "messages": {"rabcde": msgs}, "seq": {"rabcde": 1}}
            chat_room.STORE_FILE.write_text(json.dumps(payload), encoding="utf-8")
            chat_room._load()
            snap = chat_room.snapshot("rabcde", 0, ME)
            self.assertEqual([m["text"] for m in snap["messages"]], ["第3", "第4"])
            self.assertEqual(snap["seq"], 4)
        finally:
            chat_room.MAX_MESSAGES = orig

    def test_corrupt_store_is_backed_up(self):
        chat_room.STORE_FILE.write_text("{ 这不是 JSON ][", encoding="utf-8")
        info = chat_room._load()
        self.assertFalse(info["ok"])
        self.assertIn("损坏", info["error"])
        self.assertEqual(chat_room.list_rooms()["guest"], [])
        bad = list(self.tmp.glob("chat_rooms.json.bad-*"))
        self.assertEqual(len(bad), 1)
        self.assertFalse(chat_room.STORE_FILE.exists())

    def test_store_with_bom_still_loads(self):
        payload = {"version": 1,
                   "rooms": [{"id": "rabcde", "name": "带BOM", "owner": ME, "at": 1.0}],
                   "messages": {}, "seq": {}}
        chat_room.STORE_FILE.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
        info = chat_room._load()
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual([r["name"] for r in chat_room.list_rooms()["guest"]], ["带BOM"])

    def test_missing_store_is_fine(self):
        info = chat_room._load()
        self.assertFalse(info["ok"])
        self.assertEqual(info["error"], "")
        self.assertEqual(len(chat_room.list_rooms()["official"]), 4)


class AskTest(Base):
    """房间里问古籍库：落一条问答消息 + 提问限速。"""

    def test_post_bot_message_shape(self):
        rid = self.make_room()["id"]
        msg = chat_room.post_bot(rid, "小柴胡汤的君药是哪一味", "君药是柴胡。",
                                 cites=[{"book": "伤寒论", "line": "L3", "quote": "柴胡半斤"}],
                                 by=ME, backend="rag")
        self.assertEqual(msg["id"], "bot")
        self.assertEqual(msg["name"], "AI 中医师")
        self.assertEqual(msg["kind"], "qa")
        self.assertEqual(msg["seq"], 1)
        self.assertEqual(msg["by"], ME)
        self.assertEqual(msg["by_name"], "游客" + ME)
        self.assertEqual(msg["cites"][0]["book"], "伤寒论")
        self.assertFalse(msg["refused"])
        snap = chat_room.snapshot(rid, 0, ME)
        self.assertEqual(snap["messages"][0]["kind"], "qa")
        self.assertEqual(snap["total"], 1)          # 问答也算消息，占 400 条的名额

    def test_post_bot_validates(self):
        rid = self.make_room()["id"]
        with self.assertRaises(ValueError):
            chat_room.post_bot("rnope1", "问", "答")
        with self.assertRaises(ValueError):
            chat_room.post_bot(rid, "问", "   ")

    def test_post_bot_truncates_long_answer(self):
        rid = self.make_room()["id"]
        msg = chat_room.post_bot(rid, "问", "字" * (chat_room.MAX_BOT_TEXT + 50))
        self.assertEqual(len(msg["text"]), chat_room.MAX_BOT_TEXT)
        self.assertTrue(msg["text"].endswith("…"))

    def test_post_bot_keeps_at_most_four_cites(self):
        rid = self.make_room()["id"]
        cites = [{"book": "书%d" % i, "line": "L%d" % i, "quote": "原文" * 100}
                 for i in range(9)]
        msg = chat_room.post_bot(rid, "问", "答", cites=cites)
        self.assertEqual(len(msg["cites"]), 4)
        self.assertLessEqual(len(msg["cites"][0]["quote"]), 140)

    def test_post_bot_refused_flag(self):
        rid = self.make_room()["id"]
        msg = chat_room.post_bot(rid, "我这个病吃多少", "越界题，不答。", refused=True)
        self.assertTrue(msg["refused"])
        self.assertEqual(msg["cites"], [])

    def test_reserve_ask_rate_limits(self):
        chat_room.reserve_ask(ME)
        with self.assertRaises(ValueError) as ctx:
            chat_room.reserve_ask(ME)
        self.assertIn("刚问过", str(ctx.exception))
        chat_room.reserve_ask(OTHER)                     # 别人不受影响
        with chat_room._lock:                            # 把记录往前挪，绕开 12 秒间隔
            chat_room._asks[ME] = [time.time() - 60.0] * chat_room.ASK_PER_5MIN
        with self.assertRaises(ValueError) as ctx2:
            chat_room.reserve_ask(ME)
        self.assertIn("五分钟", str(ctx2.exception))

    def test_reserve_ask_rejects_bad_id(self):
        with self.assertRaises(ValueError):
            chat_room.reserve_ask("123")


    def test_qa_message_survives_restart(self):
        rid = self.make_room()["id"]
        chat_room.post_bot(rid, "麻黄汤的君药", "麻黄。" * 300,
                           cites=[{"book": "伤寒论", "page_id": "p1", "line": "L1",
                                   "quote": "麻黄三两"}],
                           by=ME, backend="rag")
        chat_room.flush()
        reset_state()
        chat_room._load()
        msg = chat_room.snapshot(rid, 0, ME)["messages"][0]
        self.assertEqual(msg["kind"], "qa")
        self.assertEqual(msg["q"], "麻黄汤的君药")
        self.assertEqual(msg["by_name"], "游客" + ME)
        self.assertEqual(msg["backend"], "rag")
        self.assertEqual(msg["cites"][0]["quote"], "麻黄三两")
        self.assertGreater(len(msg["text"]), chat_room.MAX_TEXT)   # 答复没被 600 字砍掉


class StatsTest(Base):
    def test_stats_shape(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        st = chat_room.stats()
        self.assertEqual(st["backend"], "json")
        self.assertEqual(st["official"], 4)
        self.assertEqual(st["guest_rooms"], 1)
        self.assertEqual(st["messages"], 1)
        self.assertEqual(st["invites_per_min"], chat_room.MAX_INVITES_PER_MIN)
        self.assertEqual(Path(st["store"]), chat_room.STORE_FILE)


if __name__ == "__main__":
    unittest.main()


class ReplyMentionTest(Base):
    """引用回复与 @ 点名"""

    def test_reply_carries_quoted_excerpt(self):
        rid = self.make_room()["id"]
        first = chat_room.post(rid, ME, "第一句，等下被引用。")
        got = chat_room.post(rid, OTHER, "引用你上面那句。", reply=first["seq"])
        self.assertEqual(got["reply"], first["seq"])
        self.assertEqual(got["reply_name"], "游客" + ME)
        self.assertEqual(got["reply_text"], "第一句，等下被引用。")

    def test_reply_to_qa_uses_question(self):
        rid = self.make_room()["id"]
        bot = chat_room.post_bot(rid, "甘草有何用", "甘草和中。", cites=[], by=ME)
        got = chat_room.post(rid, OTHER, "接着问", reply=bot["seq"])
        self.assertEqual(got["reply_name"], "AI 中医师")
        self.assertEqual(got["reply_text"], "甘草有何用")

    def test_reply_to_unknown_seq_is_dropped(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "只有这一条")
        for bad_seq in (0, -3, 999, "x", None):
            got = chat_room.post(rid, OTHER, "引用不存在", reply=bad_seq)
            self.assertNotIn("reply", got)
            self.assertNotIn("reply_text", got)

    def test_mentions_only_known_members(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, OTHER, "先报到")            # OTHER 进册
        got = chat_room.post(rid, ME, "@游客%s 你看这段" % OTHER)
        self.assertEqual(got["mentions"], [OTHER])

        outsiders = chat_room.post(rid, ME, "@游客99999 不在这一间")
        self.assertNotIn("mentions", outsiders)

    def test_mentions_capped_at_eight(self):
        rid = self.make_room()["id"]
        ids = ["1%04d" % i for i in range(10)]
        for g in ids:
            chat_room.post(rid, g, "报到")
        text = " ".join("@游客" + g for g in ids)
        got = chat_room.post(rid, ME, text)
        self.assertEqual(len(got["mentions"]), 8)
        self.assertTrue(all(m in ids for m in got["mentions"]))


class UnreadTest(Base):
    """未读红点"""

    def test_unread_counts_from_my_last_read(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        chat_room.post(rid, OTHER, "二")
        row = [r for r in chat_room.list_rooms(OTHER)["guest"] if r["id"] == rid][0]
        self.assertEqual(row["unread"], 2)
        snap = chat_room.snapshot(rid, 0, OTHER)
        self.assertEqual(snap["seq"], 2)
        row = [r for r in chat_room.list_rooms(OTHER)["guest"] if r["id"] == rid][0]
        self.assertEqual(row["unread"], 0)

        chat_room.post(rid, ME, "三")
        row = [r for r in chat_room.list_rooms(OTHER)["guest"] if r["id"] == rid][0]
        self.assertEqual(row["unread"], 1)

    def test_listing_without_id_has_no_unread(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        row = [r for r in chat_room.list_rooms()["guest"] if r["id"] == rid][0]
        self.assertNotIn("unread", row)

    def test_bad_id_is_ignored(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, ME, "一")
        for gid in ("abc", "1234", ""):
            row = [r for r in chat_room.list_rooms(gid)["guest"] if r["id"] == rid][0]
            self.assertNotIn("unread", row)


class RoleTest(Base):
    """房主 / 管理员"""

    def setUp(self):
        super().setUp()
        self.rid = self.make_room()["id"]
        chat_room.post(self.rid, OTHER, "报到")
        chat_room.post(self.rid, THIRD, "我也在")

    def test_owner_can_appoint_and_revoke_admin(self):
        got = chat_room.set_role(self.rid, ME, OTHER, "admin")
        self.assertEqual(got["role_text"], "管理员")
        self.assertEqual(chat_room.roles_of(self.rid)["admins"], [OTHER])
        chat_room.set_role(self.rid, ME, OTHER, "member")
        self.assertEqual(chat_room.roles_of(self.rid)["admins"], [])

    def test_only_owner_can_appoint(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        with self.assertRaises(ValueError) as ctx:
            chat_room.set_role(self.rid, OTHER, THIRD, "admin")
        self.assertIn("只有房主", str(ctx.exception))

    def test_admin_quota(self):
        for i in range(chat_room.MAX_ADMINS):
            gid = "2%04d" % i
            chat_room.post(self.rid, gid, "报到")
            chat_room.set_role(self.rid, ME, gid, "admin")
        with self.assertRaises(ValueError) as ctx:
            chat_room.set_role(self.rid, ME, OTHER, "admin")
        self.assertIn("最多", str(ctx.exception))

    def test_role_validations(self):
        with self.assertRaises(ValueError):
            chat_room.set_role(self.rid, ME, ME, "admin")          # 房主不用任命自己
        with self.assertRaises(ValueError):
            chat_room.set_role(self.rid, ME, "99999", "admin")     # 不在这一间
        with self.assertRaises(ValueError):
            chat_room.set_role(self.rid, ME, OTHER, "boss")        # 没有这种角色
        with self.assertRaises(ValueError) as ctx:
            chat_room.set_role("o-peiwu", ME, OTHER, "admin")
        self.assertIn("官方房", str(ctx.exception))

    def test_snapshot_reports_my_role(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        me = chat_room.snapshot(self.rid, 0, ME)["me"]
        self.assertEqual(me["role"], "owner")
        self.assertTrue(me["can_admin"])
        self.assertFalse(me["official"])
        other = chat_room.snapshot(self.rid, 0, OTHER)["me"]
        self.assertEqual(other["role"], "admin")
        self.assertTrue(other["can_manage"])
        plain = chat_room.snapshot(self.rid, 0, THIRD)["me"]
        self.assertEqual(plain["role"], "")
        self.assertFalse(plain["can_admin"])

    def test_members_rows_carry_role(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        rows = {m["id"]: m for m in chat_room.snapshot(self.rid, 0, ME)["members"]}
        self.assertEqual(rows[ME]["role"], "owner")
        self.assertEqual(rows[OTHER]["role"], "admin")
        self.assertEqual(rows[THIRD]["role"], "")


class MuteKickTest(Base):
    """禁言与移出"""

    def setUp(self):
        super().setUp()
        self.rid = self.make_room()["id"]
        chat_room.post(self.rid, OTHER, "报到")
        chat_room.post(self.rid, THIRD, "我也在")

    def test_mute_blocks_post_then_allows_after_expiry(self):
        chat_room.mute(self.rid, ME, OTHER, 10)
        with self.assertRaises(ValueError) as ctx:
            chat_room.post(self.rid, OTHER, "我还能说吗")
        self.assertIn("禁言", str(ctx.exception))

        with chat_room._lock:                        # 把到期时间拨到过去
            chat_room._mutes[self.rid][OTHER] = time.time() - 1
        chat_room.post(self.rid, OTHER, "到期了")
        self.assertNotIn(OTHER, chat_room._mutes.get(self.rid) or {})

    def test_unmute_and_validation(self):
        chat_room.mute(self.rid, ME, OTHER, 5)
        chat_room.mute(self.rid, ME, OTHER, 0)
        self.assertNotIn(OTHER, chat_room._mutes.get(self.rid) or {})
        for bad in (-1, chat_room.MAX_MUTE_MIN + 1):
            with self.assertRaises(ValueError):
                chat_room.mute(self.rid, ME, OTHER, bad)
        with self.assertRaises(ValueError):
            chat_room.mute(self.rid, ME, OTHER, "十")

    def test_owner_and_admins_are_protected(self):
        with self.assertRaises(ValueError) as ctx:
            chat_room.mute(self.rid, OTHER, ME, 5)     # OTHER 还不是管理员
        self.assertIn("房主或管理员", str(ctx.exception))

        chat_room.set_role(self.rid, ME, OTHER, "admin")
        with self.assertRaises(ValueError) as ctx:
            chat_room.mute(self.rid, OTHER, ME, 5)
        self.assertIn("房主不能被禁言", str(ctx.exception))

        chat_room.set_role(self.rid, ME, THIRD, "admin")
        with self.assertRaises(ValueError) as ctx:
            chat_room.mute(self.rid, OTHER, THIRD, 5)
        self.assertIn("管理员之间", str(ctx.exception))

        with self.assertRaises(ValueError):
            chat_room.mute(self.rid, ME, ME, 5)

    def test_admin_can_mute_plain_member(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        chat_room.mute(self.rid, OTHER, THIRD, 5)
        with self.assertRaises(ValueError):
            chat_room.post(self.rid, THIRD, "说不了")
        self.assertTrue(any(m["muted"] for m in chat_room.snapshot(self.rid, 0, OTHER)["members"]
                            if m["id"] == THIRD))

    def test_kick_removes_member_and_blocks_speech(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        got = chat_room.kick(self.rid, ME, OTHER)
        self.assertEqual(got["name"], "游客" + OTHER)
        self.assertEqual(chat_room.roles_of(self.rid)["admins"], [])
        with self.assertRaises(ValueError) as ctx:
            chat_room.post(self.rid, OTHER, "我还在吗")
        self.assertIn("移出", str(ctx.exception))

    def test_kick_protections(self):
        chat_room.set_role(self.rid, ME, OTHER, "admin")
        with self.assertRaises(ValueError):
            chat_room.kick(self.rid, OTHER, ME)
        with self.assertRaises(ValueError):
            chat_room.kick(self.rid, ME, ME)
        with self.assertRaises(ValueError):
            chat_room.kick(self.rid, ME, "99999")
        with self.assertRaises(ValueError) as ctx:
            chat_room.kick("o-peiwu", ME, OTHER)
        self.assertIn("官方房", str(ctx.exception))

    def test_reinvite_clears_kick(self):
        chat_room.kick(self.rid, ME, OTHER)
        inv = chat_room.invite(ME, OTHER, self.rid)
        chat_room.respond(inv["id"], OTHER, "accept")
        chat_room.post(self.rid, OTHER, "回来了")
        self.assertNotIn(OTHER, chat_room._kicked.get(self.rid) or {})


class NoticeHistTest(Base):
    """公告历史"""

    def test_newer_notice_pushes_old_one_into_history(self):
        rid = self.make_room()["id"]
        chat_room.set_notice(rid, ME, "第一版")
        chat_room.set_notice(rid, ME, "第二版")
        n = chat_room.notice(rid)
        self.assertEqual(n["text"], "第二版")
        self.assertEqual([h["text"] for h in n["history"]], ["第一版"])
        self.assertEqual(n["history"][0]["by"], ME)

    def test_history_keeps_latest_first_and_is_capped(self):
        rid = self.make_room()["id"]
        for i in range(chat_room.MAX_NOTICE_HIST + 4):
            chat_room.set_notice(rid, ME, "第%d版" % i)
        hist = chat_room.notice(rid)["history"]
        self.assertEqual(len(hist), chat_room.MAX_NOTICE_HIST)
        self.assertEqual(hist[0]["text"], "第%d版" % (chat_room.MAX_NOTICE_HIST + 2))

    def test_clearing_records_history(self):
        rid = self.make_room()["id"]
        chat_room.set_notice(rid, ME, "留着")
        got = chat_room.set_notice(rid, ME, "")
        self.assertTrue(got["cleared"])
        self.assertEqual(chat_room.notice(rid)["text"], "")
        self.assertEqual(chat_room.notice(rid)["history"][0]["text"], "留着")

    def test_admin_can_edit_notice_but_member_cannot(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, OTHER, "报到")
        chat_room.post(rid, THIRD, "报到")
        with self.assertRaises(ValueError) as ctx:
            chat_room.set_notice(rid, OTHER, "我来改")
        self.assertIn("房主或管理员", str(ctx.exception))
        chat_room.set_role(rid, ME, OTHER, "admin")
        chat_room.set_notice(rid, OTHER, "管理员改的")
        self.assertEqual(chat_room.notice(rid)["text"], "管理员改的")
        self.assertTrue(chat_room.can_manage(rid, OTHER))
        self.assertFalse(chat_room.can_manage(rid, THIRD))


class ExportTest(Base):
    """导出 Markdown"""

    def test_export_contains_everything(self):
        rid = self.make_room("验收小间", desc="用来导出")["id"]
        first = chat_room.post(rid, ME, "第一句，会被引用")
        chat_room.set_notice(rid, ME, "先报书名")
        chat_room.post(rid, OTHER, "引用一下", reply=first["seq"])
        chat_room.post_bot(rid, "甘草有何用", "甘草和中。", cites=[], by=OTHER)
        chat_room.set_role(rid, ME, OTHER, "admin")
        chat_room.save_file(rid, ME, "笔记.txt", "甘草 味甘平。".encode())

        got = chat_room.export_markdown(rid, OTHER)
        text = got["text"]
        self.assertTrue(got["filename"].endswith(".md"))
        self.assertIn("验收小间", text)
        self.assertIn("先报书名", text)
        self.assertIn("## 成员", text)
        self.assertIn("房主", text)
        self.assertIn("管理员", text)
        self.assertIn("## 聊天记录", text)
        self.assertIn("> 引用 游客%s：第一句，会被引用" % ME, text)
        self.assertIn("**AI 中医师**", text)
        self.assertIn("## 文件（1）", text)
        self.assertIn("笔记.txt", text)
        self.assertEqual(got["messages"], 3)

    def test_export_validates(self):
        rid = self.make_room()["id"]
        with self.assertRaises(ValueError):
            chat_room.export_markdown("nope", ME)
        with self.assertRaises(ValueError):
            chat_room.export_markdown(rid, "abc")

    def test_empty_room_export(self):
        rid = self.make_room("空房")["id"]
        got = chat_room.export_markdown(rid, ME)
        self.assertIn("（还没有人开口）", got["text"])
        self.assertEqual(got["messages"], 0)


class NewStatePersistTest(Base):
    """新状态要能存下来、读回来"""

    def test_roundtrip(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, OTHER, "报到")
        chat_room.set_role(rid, ME, OTHER, "admin")
        chat_room.set_notice(rid, ME, "留一版")
        chat_room.set_notice(rid, ME, "再留一版")
        chat_room.snapshot(rid, 0, OTHER)
        chat_room.flush()

        reset_state()
        chat_room._load()
        self.assertEqual(chat_room.roles_of(rid)["admins"], [OTHER])
        self.assertEqual(chat_room.notice(rid)["text"], "再留一版")
        self.assertEqual(chat_room.notice(rid)["history"][0]["text"], "留一版")
        self.assertGreater(int(chat_room._reads.get(OTHER, {}).get(rid) or 0), 0)

    def test_delete_room_clears_new_state(self):
        rid = self.make_room()["id"]
        chat_room.post(rid, OTHER, "报到")
        chat_room.set_role(rid, ME, OTHER, "admin")
        chat_room.mute(rid, ME, OTHER, 5)
        chat_room.set_notice(rid, ME, "公告")
        chat_room.delete_room(rid, ME)
        for book in (chat_room._roles, chat_room._mutes, chat_room._notice_hist):
            self.assertNotIn(rid, book)
        self.assertFalse(any(rid in v for v in chat_room._reads.values()))
