#!/bin/bash
# =============================================================
#  CuraNova — Colab Server Bootstrap
#  Usage:
#    git clone https://github.com/Paurakh977/CuraNova.git
#    cd CuraNova
#    bash colab_server.sh
# =============================================================

PORT=8000

# ─────────────────────────────────────────────
# 0. GPU check
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║     CuraNova — Colab Server Setup    ║"
echo "╚══════════════════════════════════════╝"
echo ""
python3 -c "import torch; avail=torch.cuda.is_available(); print(f'GPU available : {avail}'); print(f'CUDA device   : {torch.cuda.get_device_name(0)}' if avail else 'CUDA device   : CPU — enable GPU via Runtime > Change runtime type')" 2>/dev/null || true

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
# 3. Write .env
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 3 — Writing .env"
echo "────────────────────────────────────────"

printf "HF_TOKEN=%s\nNGROK_AUTH_TOKEN=%s\nPORT=%s\n" \
    "$HF_TOKEN" "$NGROK_AUTH_TOKEN" "$PORT" > .env

echo "✓ .env written"

# ─────────────────────────────────────────────
# 4. Start ngrok tunnel
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 4 — Starting ngrok tunnel"
echo "────────────────────────────────────────"

python3 -c "
import textwrap, pathlib
script = textwrap.dedent('''
    import os, time, signal, sys
    from pyngrok import ngrok, conf

    auth_token = os.environ[\"NGROK_AUTH_TOKEN\"]
    port       = int(os.environ.get(\"PORT\", 8000))

    conf.get_default().auth_token = auth_token
    ngrok.kill()
    time.sleep(1)

    tunnel     = ngrok.connect(port, \"http\")
    public_url = tunnel.public_url

    print(\"\", flush=True)
    print(\"╔══════════════════════════════════════════════════════════╗\", flush=True)
    print(\"║  ngrok tunnel is LIVE                                    ║\", flush=True)
    print(\"║                                                          ║\", flush=True)
    print(f\"║  Public URL : {public_url:<44}║\", flush=True)
    print(f\"║  Health     : {public_url + \"/health\":<44}║\", flush=True)
    print(f\"║  API Docs   : {public_url + \"/docs\":<44}║\", flush=True)
    print(\"║                                                          ║\", flush=True)
    print(\"║  Copy Public URL into your VPS as COLAB_BASE_URL         ║\", flush=True)
    print(\"╚══════════════════════════════════════════════════════════╝\", flush=True)
    print(\"\", flush=True)

    with open(\"/tmp/ngrok_url.txt\", \"w\") as f:
        f.write(public_url)

    def _shutdown(sig, frame):
        ngrok.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    while True:
        time.sleep(30)
''').lstrip()
pathlib.Path('/tmp/ngrok_keeper.py').write_text(script)
"

rm -f /tmp/ngrok_url.txt

python3 /tmp/ngrok_keeper.py &
NGROK_PID=$!
echo "  ngrok keeper PID: $NGROK_PID"

echo "  Waiting for tunnel to establish …"
TUNNEL_READY=0
for i in $(seq 1 40); do
    if [ -f /tmp/ngrok_url.txt ]; then
        PUBLIC_URL=$(cat /tmp/ngrok_url.txt)
        echo "✓ Tunnel ready: $PUBLIC_URL"
        TUNNEL_READY=1
        break
    fi
    sleep 1
done

if [ "$TUNNEL_READY" -eq 0 ]; then
    echo "✗ Tunnel failed to start within 40 s."
    echo "  Check your NGROK_AUTH_TOKEN and try again."
    kill "$NGROK_PID" 2>/dev/null || true
    exit 1
fi

# ─────────────────────────────────────────────
# 5. Launch FastAPI server (foreground — blocks)
# ─────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Step 5 — Starting FastAPI / uvicorn"
echo " Ctrl+C stops both server and ngrok"
echo "────────────────────────────────────────"
echo ""

trap 'echo ""; echo "Shutting down …"; kill "$NGROK_PID" 2>/dev/null || true; exit 0' INT TERM

python3 -m uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info

kill "$NGROK_PID" 2>/dev/null || true