import time
import base64
import asyncio
import requests
import websockets
import json
from colorama import init, Fore, Style

COLAB_BASE_URL = "https://a83f-136-118-65-76.ngrok-free.app"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
PROMPT = "Describe this chest X-ray. What do you see?"

init(autoreset=True)

def dbg(msg):
    print(Fore.MAGENTA + f"[CLIENT DEBUG] {msg}", flush=True)

# ─────────────────────────────────────────────
# 1. Download image
# ─────────────────────────────────────────────
print(Fore.CYAN + "⏳ Downloading image …")
try:
    img_resp = requests.get(IMAGE_URL, headers={"User-Agent": "tutorial"}, timeout=30)
    img_resp.raise_for_status()
    img_bytes = img_resp.content
    print(Fore.GREEN + f"✓ Image downloaded ({len(img_bytes) / 1024:.1f} KB)")
except Exception as e:
    print(Fore.RED + f"✗ Failed to download image: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# 2. Convert to base64
# ─────────────────────────────────────────────
image_b64 = base64.b64encode(img_bytes).decode("utf-8")
dbg(f"base64 payload length: {len(image_b64)} chars")

# ─────────────────────────────────────────────
# 3. Build WebSocket URL
# ─────────────────────────────────────────────
ws_base = COLAB_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
WS_URL = f"{ws_base}/ws/analyze/image-base64"
dbg(f"WebSocket URL: {WS_URL}")

# ─────────────────────────────────────────────
# 4. Stream via WebSocket
# ─────────────────────────────────────────────
async def stream_response():
    print(Fore.CYAN + f"\n🔌 Connecting to {WS_URL} …")

    try:
        async with websockets.connect(
            WS_URL,
            additional_headers={"User-Agent": "CuraNova-test"},
            open_timeout=30,
            ping_timeout=120,
            close_timeout=10,
            max_size=10 * 1024 * 1024,  # 10 MB — large enough for big b64 payload
        ) as ws:
            print(Fore.GREEN + "✓ Connected! Sending request …")
            dbg("Sending JSON payload …")

            payload = json.dumps({
                "image_b64": image_b64,
                "prompt": PROMPT,
                "max_new_tokens": 500,
            })
            dbg(f"Payload size: {len(payload) / 1024:.1f} KB")
            await ws.send(payload)
            dbg("Payload sent. Waiting for tokens …")

            print(Style.BRIGHT + "\n── Model Response ──────────────────────────────\n")

            start = time.perf_counter()
            full_response = []
            msg_count = 0

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    dbg("⚠️  120s timeout waiting for next message — server may have stalled")
                    break

                msg_count += 1

                # All messages are now JSON objects from the fixed server
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    # Fallback: server sent raw text (shouldn't happen with new server)
                    dbg(f"RAW (non-JSON) msg #{msg_count}: {repr(raw[:80])}")
                    print(raw, end="", flush=True)
                    full_response.append(raw)
                    continue

                # Debug: log first 5 and every 20th message
                if msg_count <= 5 or msg_count % 20 == 0:
                    dbg(f"msg #{msg_count}: {repr(str(msg)[:100])}")

                if "token" in msg:
                    # This is a streamed token
                    token = msg["token"]
                    print(token, end="", flush=True)
                    full_response.append(token)

                elif msg.get("status") == "done":
                    elapsed = time.perf_counter() - start
                    print(Fore.GREEN + f"\n\n── Done in {elapsed:.2f}s — {msg_count} messages received ──")
                    dbg(f"Total chars in response: {sum(len(t) for t in full_response)}")
                    break

                elif "error" in msg:
                    print(Fore.RED + f"\n✗ Server error: {msg['error']}")
                    dbg(f"Full error message: {msg}")
                    break

                else:
                    # Unknown JSON — log and skip
                    dbg(f"⚠️  Unknown msg shape: {repr(str(msg)[:200])}")

            return "".join(full_response)

    except websockets.exceptions.ConnectionClosedError as e:
        print(Fore.RED + f"\n✗ Connection closed unexpectedly: {e}")
        dbg(f"Close code: {e.code}, reason: {e.reason}")
    except websockets.exceptions.InvalidURI:
        print(Fore.RED + f"✗ Invalid WebSocket URI: {WS_URL}")
    except websockets.exceptions.WebSocketException as e:
        print(Fore.RED + f"✗ WebSocket error: {type(e).__name__}: {e}")
    except Exception as e:
        print(Fore.RED + f"✗ Unexpected error: {type(e).__name__}: {e}")

    return None


if __name__ == "__main__":
    result = asyncio.run(stream_response())
    if result:
        print(Fore.CYAN + f"\n[CLIENT] Full response length: {len(result)} chars")
    else:
        print(Fore.RED + "\n[CLIENT] No response received.")