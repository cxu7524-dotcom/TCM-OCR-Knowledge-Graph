#!/usr/bin/env sh
# ============================================================
#  赛题二原型 · 一键启动（Linux / macOS）
#
#  用法：
#      sh scripts/start.sh                 # 默认 8100，占用则顺延 8000 / 8101-8103
#      TCM_PORT=8123 sh scripts/start.sh   # 指定端口
#      TCM_LLM_BACKEND=none sh scripts/start.sh   # 只走规则链路
#
#  前提：python3 + `pip install -r requirements.txt`；
#        本机 Ollama 可选（没有则问答降级，界面会提示）。
# ============================================================
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
: "${TCM_LLM_BACKEND:=ollama}"
: "${TCM_LLM_MODEL:=qwen2.5:7b}"
export TCM_LLM_BACKEND TCM_LLM_MODEL

if [ -x "$root/.venv/bin/python" ]; then
    py="$root/.venv/bin/python"
else
    py=$(command -v python3 || command -v python || true)
    if [ -z "${py:-}" ]; then
        echo "[ERROR] 没找到 python3：先装 Python 3.10+，或建 .venv" >&2
        exit 1
    fi
    echo "  [note] 没有 .venv，改用系统 Python：$py"
fi

if [ ! -f "$root/requirements.txt" ]; then
    echo "[ERROR] 当前目录不像项目根目录（缺 requirements.txt）：$root" >&2
    exit 1
fi

# 端口占用检测：0=空闲，1=被占
port_busy() {
    "$py" - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

chosen=""
skipped=""
for p in "${TCM_PORT:-8100}" 8000 8101 8102 8103; do
    case " $skipped " in
        *" $p "*) continue ;;
    esac
    if port_busy "$p"; then
        skipped="$skipped $p"
        continue
    fi
    chosen="$p"
    break
done

if [ -z "$chosen" ]; then
    echo "[ERROR] 端口 8100 / 8000 / 8101-8103 都被占用：先关掉占用的服务，或用 TCM_PORT=8123" >&2
    exit 1
fi

echo ""
echo "  中医古籍 OCR 与知识图谱问答原型"
echo "  地址：  http://127.0.0.1:$chosen"
echo "  模型：  $TCM_LLM_BACKEND / $TCM_LLM_MODEL"
echo "  页面：  / 四步演示台   /home 智能问答   /lounge 聊天室"
echo "  声明：  仅供参考、非诊疗建议"
echo ""

exec "$py" app/main.py --port "$chosen"
