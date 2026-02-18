#!/bin/bash
# =============================================================
#  CuraNova — Colab Server Bootstrap
#  Upload this file to Colab and run:  !bash colab_server.sh
# =============================================================

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
python3 -c "
import torch
print(f'GPU available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device   : {torch.cuda.get_device_name(0)}')
else:
    print('CUDA device   : CPU — enable GPU via Runtime > Change runtime type')
" 2>/dev/null || true

# ─────────────────────────────────────────────
# 1. Collect secrets interactively
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 1 — Enter your API tokens"
echo "────────────────────────────────────────"

read -rsp "  HuggingFace token (hf_...): " HF_TOKEN
echo ""
if [ -z "$HF_TOKEN" ]; then
    echo "✗ HF_TOKEN cannot be empty. Exiting."
    exit 1
fi

read -rsp "  Ngrok auth token: " NGROK_AUTH_TOKEN
echo ""
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "✗ NGROK_AUTH_TOKEN cannot be empty. Exiting."
    exit 1
fi
echo ""

# Export so all child processes inherit them
export HF_TOKEN
export NGROK_AUTH_TOKEN
export PORT

# ─────────────────────────────────────────────
# 2. Install packages
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
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
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
    cd "$REPO_DIR" && git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

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
# 5. Start ngrok in background via a persistent
#    Python process that keeps the tunnel alive
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 5 — Starting ngrok tunnel"
echo "────────────────────────────────────────"

# Write the ngrok keeper to a temp file so it
# runs as a separate long-lived background process
cat > /tmp/ngrok_keeper.py << 'PYEOF'
import os, time, signal, sys
from pyngrok import ngrok, conf

auth_token = os.environ["NGROK_AUTH_TOKEN"]
port       = int(os.environ.get("PORT", 8000))

conf.get_default().auth_token = auth_token
ngrok.kill()   # clean up any stale tunnel from a previous run
time.sleep(1)

tunnel     = ngrok.connect(port, "http")
public_url = tunnel.public_url

url_line   = public_url
health_line = public_url + "/health"
docs_line   = public_url + "/docs"

print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✓ ngrok tunnel is LIVE                                  ║
║                                                          ║
║  Public URL : {url_line:<44}║
║  Health     : {health_line:<44}║
║  API Docs   : {docs_line:<44}║
║                                                          ║
║  Copy Public URL into your VPS as COLAB_BASE_URL         ║
╚══════════════════════════════════════════════════════════╝
""", flush=True)

# Write URL so the shell script can read it back
with open("/tmp/ngrok_url.txt", "w") as f:
    f.write(public_url)

def _shutdown(sig, frame):
    print("\nngrok keeper: closing tunnel …")
    ngrok.kill()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# Block forever — keeping the tunnel open
while True:
    time.sleep(30)
PYEOF

# Launch keeper in background
python3 /tmp/ngrok_keeper.py &
NGROK_PID=$!
echo "  ngrok keeper PID: $NGROK_PID"

# Poll until URL file appears (max 30 s)
echo "  Waiting for tunnel to establish …"
TUNNEL_READY=0
for i in $(seq 1 30); do
    if [ -f /tmp/ngrok_url.txt ]; then
        PUBLIC_URL=$(cat /tmp/ngrok_url.txt)
        echo "✓ Tunnel ready: $PUBLIC_URL"
        TUNNEL_READY=1
        break
    fi
    sleep 1
done

if [ "$TUNNEL_READY" -eq 0 ]; then
    echo "✗ Tunnel failed to start within 30 s."
    echo "  Check your NGROK_AUTH_TOKEN and try again."
    kill "$NGROK_PID" 2>/dev/null || true
    exit 1
fi

# ─────────────────────────────────────────────
# 6. Launch FastAPI server (foreground — blocks)
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 6 — Starting FastAPI / uvicorn"
echo " Ctrl+C stops both the server and ngrok"
echo "────────────────────────────────────────"
echo ""

# On exit (Ctrl+C or crash) also kill the ngrok keeper
trap 'echo ""; echo "Shutting down …"; kill "$NGROK_PID" 2>/dev/null || true; exit 0' INT TERM

python3 -m uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info

# If uvicorn exits by itself, clean up ngrok too
kill "$NGROK_PID" 2>/dev/null || true