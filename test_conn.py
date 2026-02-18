import time
import requests
from colorama import init, Fore, Style

COLAB_BASE_URL = "https://be95-35-198-249-79.ngrok-free.app"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
PROMPT = "Describe this chest X-ray. What do you see?"

init(autoreset=True)

try:
    img_resp = requests.get(IMAGE_URL, headers={"User-Agent": "tutorial"}, timeout=30)
    img_resp.raise_for_status()
    img_bytes = img_resp.content
except Exception as e:
    print(Fore.RED + f"Failed to download image: {e}")
    raise SystemExit(1)

start = time.perf_counter()
try:
    response = requests.post(
        f"{COLAB_BASE_URL}/analyze/image-upload",
        files={"file": ("chest_xray.png", img_bytes, "image/png")},
        data={"prompt": PROMPT, "max_new_tokens": "500"},
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    if response.ok:
        color = Fore.GREEN
    else:
        color = Fore.YELLOW

    print(color + f"Request finished in {elapsed:.2f}s — status: {response.status_code}")

    ctype = response.headers.get("Content-Type", "")
    try:
        if "application/json" in ctype:
            print(Style.BRIGHT + str(response.json()))
        else:
            print(response.text)
    except Exception:
        print(response.text)

except Exception as e:
    elapsed = time.perf_counter() - start
    print(Fore.RED + f"Request failed after {elapsed:.2f}s: {e}")