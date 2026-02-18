#!/bin/bash
# =============================================================
#  CuraNova — Colab Server Bootstrap
#  Upload this file to Colab and run:  !bash colab_server.sh
# =============================================================

set -e  # exit on any error

REPO_URL="https://github.com/Paurakh977/CuraNova.git"   
REPO_DIR="CuraNova"                                    
PORT=8000

# ─────────────────────────────────────────────
# 0. GPU check
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║     CuraNova — Colab Server Setup    ║"
echo "╚══════════════════════════════════════╝"
echo ""
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); \
            print(f'CUDA device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU — consider enabling GPU in Runtime > Change runtime type\"}')" 2>/dev/null || true

# ─────────────────────────────────────────────
# 1. Collect secrets interactively
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 1 — Enter your API tokens"
echo "────────────────────────────────────────"

read -rsp "  HuggingFace token (hf_...): " HF_TOKEN
echo ""

read -rsp "  Ngrok auth token: " NGROK_AUTH_TOKEN
echo ""
echo ""

# ─────────────────────────────────────────────
# 2. Install system packages
# ─────────────────────────────────────────────
echo "────────────────────────────────────────"
echo " Step 2 — Installing packages"
echo "────────────────────────────────────────"

pip install -q \
    transformers \
    pillow \
    torch \
    matplotlib \
    requests \
    huggingface_hub \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    python-dotenv \
    pyngrok \
    accelerate

echo "✓ Packages installed"

# ─────────────────────────────────────────────
# 3. Clone / update repo
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 3 — Cloning repository"
echo "────────────────────────────────────────"

if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo already exists — pulling latest …"
    cd "$REPO_DIR" && git pull && cd ..
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
echo "✓ Repo ready at $(pwd)"

# ─────────────────────────────────────────────
# 4. Write .env
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 4 — Writing .env"
echo "────────────────────────────────────────"

cat > .env <<EOF
HF_TOKEN=${HF_TOKEN}
NGROK_AUTH_TOKEN=${NGROK_AUTH_TOKEN}
PORT=${PORT}
EOF

echo "✓ .env written"

# ─────────────────────────────────────────────
# 5. Start ngrok tunnel in background
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 5 — Starting ngrok tunnel"
echo "────────────────────────────────────────"

python3 - <<PYEOF
from pyngrok import ngrok, conf
import os, time

conf.get_default().auth_token = os.environ.get("NGROK_AUTH_TOKEN", "${NGROK_AUTH_TOKEN}")

# kill any existing tunnels
ngrok.kill()
time.sleep(1)

tunnel = ngrok.connect(${PORT}, "http")
public_url = tunnel.public_url

print(f"""
╔══════════════════════════════════════════════════════╗
║  ✓ ngrok tunnel is LIVE                              ║
║                                                      ║
║  Public URL : {public_url:<38}║
║  Health     : {public_url}/health{' '*max(0,32-len('/health'))}║
║                                                      ║
║  Use this URL in your VPS backend as COLAB_BASE_URL  ║
╚══════════════════════════════════════════════════════╝
""")

# Save to file so you can cat it later
with open("/tmp/ngrok_url.txt", "w") as f:
    f.write(public_url)
PYEOF

export NGROK_AUTH_TOKEN="${NGROK_AUTH_TOKEN}"

# ─────────────────────────────────────────────
# 6. Launch FastAPI server (foreground)
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 6 — Starting FastAPI server"
echo " Press Ctrl+C to stop"
echo "────────────────────────────────────────"
echo ""

export HF_TOKEN="${HF_TOKEN}"

python3 -m uvicorn server:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    --no-access-log