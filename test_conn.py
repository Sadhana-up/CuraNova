import time
import base64
import asyncio
import requests
import websockets
import json
from colorama import init, Fore, Style

COLAB_BASE_URL = "https://be95-35-198-249-79.ngrok-free.app"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
PROMPT = "Describe this chest X-ray. What do you see?"

init(autoreset=True)

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
# 2. Convert to base64 (WebSocket payload)
# ─────────────────────────────────────────────
image_b64 = base64.b64encode(img_bytes).decode("utf-8")

# ─────────────────────────────────────────────
# 3. Build WebSocket URL
#    http(s)://host  →  ws(s)://host
# ─────────────────────────────────────────────
ws_base = COLAB_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
WS_URL = f"{ws_base}/ws/analyze/image-base64"

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
            ping_timeout=60,
        ) as ws:
            print(Fore.GREEN + "✓ Connected! Sending request …\n")

            payload = json.dumps({
                "image_b64": image_b64,
                "prompt": PROMPT,
                "max_new_tokens": 500,
            })
            await ws.send(payload)

            print(Style.BRIGHT + "── Model Response ──────────────────────────────\n")

            start = time.perf_counter()
            full_response = []

            while True:
                raw = await ws.recv()

                # Try to parse as JSON (control message)
                try:
                    msg = json.loads(raw)
                    if msg.get("status") == "done":
                        elapsed = time.perf_counter() - start
                        print(Fore.GREEN + f"\n\n── Done in {elapsed:.2f}s ──────────────────────────")
                        break
                    elif "error" in msg:
                        print(Fore.RED + f"\n✗ Server error: {msg['error']}")
                        break
                except json.JSONDecodeError:
                    # Plain token — print immediately without newline
                    print(raw, end="", flush=True)
                    full_response.append(raw)

            return "".join(full_response)

    except websockets.exceptions.InvalidURI:
        print(Fore.RED + f"✗ Invalid WebSocket URI: {WS_URL}")
    except websockets.exceptions.WebSocketException as e:
        print(Fore.RED + f"✗ WebSocket error: {e}")
    except Exception as e:
        print(Fore.RED + f"✗ Unexpected error: {e}")

    return None


if __name__ == "__main__":
    asyncio.run(stream_response())