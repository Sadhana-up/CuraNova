#!/bin/bash
# =============================================================
#  CuraNova — Client Bootstrap
#  Usage:
#    bash client_server.sh
# =============================================================

PORT=3000
CLIENT_DIR="curanova-client"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   CuraNova — Client Server Setup     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# 1. Check bun
# ─────────────────────────────────────────────
echo "────────────────────────────────────────"
echo " Step 1 — Checking bun installation"
echo "────────────────────────────────────────"

if ! command -v bun &>/dev/null; then
    echo "  bun not found — installing …"
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
else
    echo "✓ bun found: $(bun --version)"
fi

# ─────────────────────────────────────────────
# 2. Navigate to client directory
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 2 — Entering client directory"
echo "────────────────────────────────────────"

if [ ! -d "$CLIENT_DIR" ]; then
    echo "✗ Directory '$CLIENT_DIR' not found. Exiting."
    exit 1
fi

cd "$CLIENT_DIR" || exit 1
echo "✓ Now in: $(pwd)"

# ─────────────────────────────────────────────
# 3. Install dependencies
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 3 — Installing dependencies"
echo "────────────────────────────────────────"

bun install
echo "✓ Dependencies installed"

# ─────────────────────────────────────────────
# 4. Build
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 4 — Building Next.js app"
echo "────────────────────────────────────────"

bun run build
echo "✓ Build complete"

# ─────────────────────────────────────────────
# 5. Start (foreground — blocks)
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 5 — Starting Next.js server"
echo " Ctrl+C to stop"
echo "────────────────────────────────────────"
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Client is LIVE                      ║"
echo "║                                      ║"
echo "║  Local : http://localhost:$PORT          ║"
echo "╚══════════════════════════════════════╝"
echo ""

trap 'echo ""; echo "Shutting down …"; exit 0' INT TERM

bun start