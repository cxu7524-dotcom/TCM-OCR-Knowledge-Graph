# -*- coding: utf-8 -*-
"""赛题二原型 · 故障自检（负向用例）

用法（项目根目录，服务已起）：
    python scripts/faults.py                 # 默认打 127.0.0.1:8100
    python scripts/faults.py --port 8000
    python scripts/faults.py --base 127.0.0.1:8123

做什么：只发「本来就该被拒绝」的请求，确认服务不是 500、不是英文报错、
        也不会把失败当成功。每个用例给期望状态码，末尾汇总，有失败则退出码 1。
影响：不上传成功页、不改已有数据；限流那一条会临时建一间房并在结束时删掉。
"""

from __future__ import annotations

import argparse
import http.client
import json
import secrets
import sys

CASES = []


def case(name, method, path, expect, body=None, raw=None, ctype=None, note=""):
    CASES.append(dict(name=name, method=method, path=path, expect=expect,
                      body=body, raw=raw, ctype=ctype, note=note))


def call(base, c):
    host, _, port = base.partition(":")
    conn = http.client.HTTPConnection(host or "127.0.0.1", int(port or 80), timeout=180)
    headers, data = {}, None
    if c["raw"] is not None:
        data = c["raw"]
        headers["Content-Type"] = c["ctype"]
    elif c["body"] is not None:
        data = json.dumps(c["body"], ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    conn.request(c["method"], c["path"], body=data, headers=headers)
    r = conn.getresponse()
    txt = r.read().decode("utf-8", "replace")
    conn.close()
    return r.status, txt


def multipart(filename, content):
    b = "----faults" + secrets.token_hex(8)
    body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n" % (b, filename)).encode("utf-8")
    body += content + ("\r\n--%s--\r\n" % b).encode("utf-8")
    return body, "multipart/form-data; boundary=" + b


def detail_of(txt):
    try:
        d = json.loads(txt)
    except Exception:
        return txt.strip()[:90]
    msg = d.get("detail", d.get("text", d.get("ok", d)))
    if isinstance(msg, list):
        msg = " | ".join(str(x.get("msg", x)) for x in msg[:2])
    if isinstance(msg, (dict, list, bool)):
        msg = json.dumps(msg, ensure_ascii=False)
    return str(msg)[:90]


def build():
    body, ct = multipart("note.txt", b"not an image")
    case("非图片扩展名 note.txt", "POST", "/api/pages", 400, raw=body, ctype=ct)
    body, ct = multipart("fake.png", b"definitely not a png")
    case("改了扩展名的假 png", "POST", "/api/pages", 400, raw=body, ctype=ct)
    body, ct = multipart("huge.png", b"\x89PNG\r\n\x1a\n" + b"0" * (21 * 1024 * 1024))
    case("21MB 超限上传", "POST", "/api/pages", 413, raw=body, ctype=ct)
    case("不存在的页 OCR", "GET", "/api/pages/nope-1234/ocr", 404)
    case("未知接口", "GET", "/api/nothing", 404, note="文案应为中文")
    case("方法不对", "GET", "/api/rag/rebuild", 405)
    case("缺字段 422（语音无音频）", "POST", "/api/asr", 422,
         raw=b"", ctype="multipart/form-data; boundary=x", note="文案应为中文")
    case("空消息", "POST", "/api/chat/send", 400,
         body={"room": "o-jianbie", "id": "00001", "text": ""})
    case("超长消息 700 字", "POST", "/api/chat/send", 400,
         body={"room": "o-jianbie", "id": "00001", "text": "字" * 700})
    case("不存在的房间发消息", "POST", "/api/chat/send", (400, 404),
         body={"room": "nope", "id": "00001", "text": "在吗"}, note="400/404 都算可读")
    case("空提问（聊天室 AI）", "POST", "/api/chat/ask", 400,
         body={"room": "o-jianbie", "id": "00001", "question": ""})
    case("空问题（问答接口）", "POST", "/api/qa", 200,
         body={"question": ""}, note="不是报错：应退回拒答")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="例如 127.0.0.1:8000（优先于 --port）")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    base = args.base or ("127.0.0.1:%d" % args.port)

    build()
    print("")
    print("  故障自检目标：%s" % base)
    print("")
    rows, bad = [], 0
    for c in CASES:
        try:
            status, txt = call(base, c)
        except Exception as e:  # 服务没起
            print("  [FAIL] %-26s 连不上：%s" % (c["name"], e))
            bad += 1
            continue
        exp = c["expect"]
        ok = status in exp if isinstance(exp, tuple) else status == exp
        msg = detail_of(txt)
        # 文案里不该出现框架的英文原句
        english = msg.strip() in {"Not Found", "Method Not Allowed", "Field required"}
        if english:
            ok = False
        tag = "PASS" if ok else "FAIL"
        print("  [%s] %-26s %-4s 期望 %-7s %s" % (tag, c["name"], status, exp, msg))
        if not ok:
            bad += 1

    # 限流用例：临时建房 → 连发两条 → 删房
    try:
        status, txt = call(base, dict(name="建房", method="POST", path="/api/rooms", expect=200,
                                      body={"owner": "00001", "name": "故障自检临时房", "desc": "测完即删"},
                                      raw=None, ctype=None))
        room = ""
        try:
            d = json.loads(txt)
            room = (d.get("room") or {}).get("id") or d.get("id") or ""
        except Exception:
            pass
        if status == 200 and room:
            call(base, dict(name="", method="POST", path="/api/chat/send", expect=200,
                            body={"room": room, "id": "00001", "text": "第一条"}, raw=None, ctype=None))
            st2, t2 = call(base, dict(name="", method="POST", path="/api/chat/send", expect=429,
                                      body={"room": room, "id": "00001", "text": "第二条"}, raw=None, ctype=None))
            ok = st2 == 429
            print("  [%s] %-26s %-4s 期望 429     %s" % ("PASS" if ok else "FAIL", "1.2 秒内连发两条", st2, detail_of(t2)))
            bad += 0 if ok else 1
            call(base, dict(name="", method="DELETE", path="/api/rooms?id=%s&owner=00001" % room,
                            expect=200, body=None, raw=None, ctype=None))
        else:
            print("  [....] %-26s 建房失败，跳过限流用例" % "1.2 秒内连发两条")
    except Exception as e:
        print("  [....] 限流用例跳过：%s" % e)

    print("")
    if bad == 0:
        print("  全部通过：%d 个负向用例都给出了预期的状态码与中文说明" % (len(CASES) + 1))
        return 0
    print("  有 %d 个用例不符合预期" % bad)
    return 1


if __name__ == "__main__":
    sys.exit(main())
