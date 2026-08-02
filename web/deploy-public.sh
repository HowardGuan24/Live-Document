#!/usr/bin/env bash
# Live-Document 公网一键部署（Radeon Cloud Notebook）
# 前提：Notebook 为新建 Pod（含 FRP_BROKER_URL 环境变量），仓库已克隆到本机。
# 用法：bash web/deploy-public.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LIVE_DOC_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
BACKEND="$ROOT/web/backend"
FRONTEND="$ROOT/web/frontend"
PORT="${LIVE_DOC_PORT:-8000}"
LOG_DIR="${LIVE_DOC_LOG_DIR:-/tmp/live-doc}"
mkdir -p "$LOG_DIR"

echo "==> [1/4] 构建前端"
(cd "$FRONTEND" && npm install --no-audit --no-fund >/dev/null && npm run build)

echo "==> [2/4] 启动后端（鉴权 + 前端托管，端口 $PORT）"
pkill -f 'uvicorn app.main:app' 2>/dev/null || true
sleep 1
(
  cd "$BACKEND"
  LIVE_DOC_SERVE_FRONTEND=1 LIVE_DOC_HOST=0.0.0.0 LIVE_DOC_PORT="$PORT" \
    nohup "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG_DIR/backend.log" 2>&1 &
)
for i in $(seq 1 24); do
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
  sleep 0.5
done
TOKEN="$(cat "$BACKEND/data/auth_token.txt" 2>/dev/null || true)"
echo "    访问令牌: ${TOKEN:-见 $LOG_DIR/backend.log}"

echo "==> [3/4] 安装 rc-tunnel（若未安装）"
if [ ! -x "$HOME/.local/bin/rc-tunnel" ]; then
  /var/run/secrets/frp-self-service/install
fi
"$HOME/.local/bin/rc-tunnel" version >/dev/null 2>&1 || {
  echo "rc-tunnel 安装失败：请确认这是新创建的 Notebook Pod（含 FRP_BROKER_URL）。" >&2
  exit 1
}

echo "==> [4/4] 暴露公网端口 $PORT"
nohup "$HOME/.local/bin/rc-tunnel" expose --port "$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
TPID=$!
for i in $(seq 1 20); do
  grep -qEo 'https://rc-[a-z0-9]+\.radeon\.firstdg\.ai' "$LOG_DIR/tunnel.log" 2>/dev/null && break
  sleep 1
done
sleep 3
echo "    rc-tunnel PID: $TPID"
cat "$LOG_DIR/tunnel.log"
echo
echo "================= 部署完成 ================="
echo "公网地址: $(grep -Eo 'https://rc-[a-z0-9]+\.radeon\.firstdg\.ai' "$LOG_DIR/tunnel.log" | head -1 || echo '见 tunnel.log')"
echo "本地地址: http://127.0.0.1:$PORT"
echo "访问令牌: ${TOKEN:-见 $LOG_DIR/backend.log}（登录页面需要）"
