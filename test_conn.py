import requests

COLAB_BASE_URL = "https://4a81-34-50-187-249.ngrok-free.app"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
PROMPT = "Describe this chest X-ray. What do you see?"

img_bytes = requests.get(IMAGE_URL, headers={"User-Agent": "tutorial"}).content

response = requests.post(
    f"{COLAB_BASE_URL}/analyze/image-upload",
    files={"file": ("chest_xray.png", img_bytes, "image/png")},
    data={"prompt": PROMPT, "max_new_tokens": "500"},
    timeout=120,
)

print(response.json())