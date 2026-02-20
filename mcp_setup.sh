#!/bin/bash
# =============================================================
#  CuraNova — FastAPI Server Bootstrap
#  Usage:
#    bash server.sh
# =============================================================

PORT=8000
SERVER_DIR="."

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   CuraNova — FastAPI Server Setup    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# 1. Check uv
# ─────────────────────────────────────────────
echo "────────────────────────────────────────"
echo " Step 1 — Checking uv installation"
echo "────────────────────────────────────────"

if ! command -v uv &>/dev/null; then
    echo "  uv not found — installing …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✓ uv found: $(uv --version)"
fi

# ─────────────────────────────────────────────
# 2. Navigate to working directory
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 2 — Entering working directory"
echo "────────────────────────────────────────"

if [ ! -d "$SERVER_DIR" ]; then
    echo "✗ Directory '$SERVER_DIR' not found. Exiting."
    exit 1
fi

cd "$SERVER_DIR" || exit 1
echo "✓ Now in: $(pwd)"

# ─────────────────────────────────────────────
# 3. Sync dependencies
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 3 — Syncing dependencies"
echo "────────────────────────────────────────"

uv sync
echo "✓ Dependencies synced"

# ─────────────────────────────────────────────
# 4. Start FastAPI (foreground — blocks)
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 4 — Starting FastAPI / uvicorn"
echo " Ctrl+C to stop"
echo "────────────────────────────────────────"
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Server is LIVE                      ║"
echo "║                                      ║"
echo "║  Local : http://localhost:$PORT         ║"
echo "║  Docs  : http://localhost:$PORT/docs    ║"
echo "╚══════════════════════════════════════╝"
echo ""

trap 'echo ""; echo "Shutting down …"; exit 0' INT TERM

uv run uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info