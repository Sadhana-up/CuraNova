import os
import io
import base64
import threading
import asyncio
import requests
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, TextIteratorStreamer
from huggingface_hub import login
import uvicorn
from dotenv import load_dotenv

load_dotenv()


# Global model cache — loaded ONCE at startup
MODEL_CACHE: dict = {}

_SENTINEL = object()  # signals end of token stream


def get_pipeline():
    if "pipe" not in MODEL_CACHE:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again in a moment.")
    return MODEL_CACHE["pipe"]


def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    login(token=hf_token)
    print("✓ Logged in to Hugging Face")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading MedGemma on {device} …")

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device=device,
        token=hf_token,
    )

    MODEL_CACHE["pipe"] = pipe
    print(f"✓ Model loaded on: {pipe.device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield
    MODEL_CACHE.clear()
    print("Model unloaded.")


app = FastAPI(
    title="CuraNova",
    description="Medical AI Backend powered by MedGemma",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────

class HealthRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None

class HealthResponse(BaseModel):
    response: str
    status: str

class TextAnalysisRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 500

class ImageURLRequest(BaseModel):
    image_url: str
    prompt: Optional[str] = "Describe this medical image. What do you see?"
    max_new_tokens: Optional[int] = 500


# ── Non-streaming inference ───────────────────────────────────

def run_inference(image: Optional[Image.Image], prompt: str, max_new_tokens: int = 500) -> str:
    pipe = get_pipeline()
    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    output = pipe(text=messages, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"][-1]["content"]


# ── Async streaming inference ─────────────────────────────────

async def run_inference_streaming_async(
    image: Optional[Image.Image],
    prompt: str,
    max_new_tokens: int = 500,
):
    """
    Runs model.generate() in a background thread and yields tokens
    asynchronously via asyncio.Queue — never blocks the event loop.
    """
    pipe = get_pipeline()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    inputs = pipe.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(pipe.model.device)

    def _generate():
        """Runs in background thread; pushes tokens into asyncio queue."""
        try:
            pipe.model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item  # plain token string


# ── REST Routes ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "CuraNova API is running!", "model": "google/medgemma-1.5-4b-it"}


@app.get("/health")
async def health_check():
    model_ready = "pipe" in MODEL_CACHE
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "message": "Model is ready!" if model_ready else "Model is still loading, please wait ...",
    }


@app.post("/analyze", response_model=HealthResponse)
async def analyze(data: HealthRequest):
    try:
        response_text = run_inference(image=None, prompt=data.query)
        return HealthResponse(response=response_text, status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/text", response_model=HealthResponse)
async def analyze_text(data: TextAnalysisRequest):
    try:
        response_text = run_inference(image=None, prompt=data.prompt, max_new_tokens=data.max_new_tokens)
        return HealthResponse(response=response_text, status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image-url", response_model=HealthResponse)
async def analyze_image_url(data: ImageURLRequest):
    try:
        resp = requests.get(data.image_url, headers={"User-Agent": "CuraNova"}, stream=True, timeout=15)
        resp.raise_for_status()
        image = Image.open(resp.raw).convert("RGB")
        response_text = run_inference(image=image, prompt=data.prompt, max_new_tokens=data.max_new_tokens)
        return HealthResponse(response=response_text, status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image-upload", response_model=HealthResponse)
async def analyze_image_upload(
    file: UploadFile = File(...),
    prompt: str = Form(default="Describe this medical image. What do you see?"),
    max_new_tokens: int = Form(default=500),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        response_text = run_inference(image=image, prompt=prompt, max_new_tokens=max_new_tokens)
        return HealthResponse(response=response_text, status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image-base64", response_model=HealthResponse)
async def analyze_image_base64(
    image_b64: str = Form(...),
    prompt: str = Form(default="Describe this medical image. What do you see?"),
    max_new_tokens: int = Form(default=500),
):
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        response_text = run_inference(image=image, prompt=prompt, max_new_tokens=max_new_tokens)
        return HealthResponse(response=response_text, status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket Streaming Endpoints ─────────────────────────────

@app.websocket("/ws/analyze/text")
async def ws_analyze_text(websocket: WebSocket):
    """
    Client sends: {"prompt": "...", "max_new_tokens": 500}
    Server streams tokens, then sends: {"status": "done"}
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 500))

        if "pipe" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        async for token in run_inference_streaming_async(None, prompt, max_new_tokens):
            await websocket.send_text(token)
            await asyncio.sleep(0)  # yield to event loop → forces immediate flush

        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("Client disconnected from /ws/analyze/text")
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/analyze/image-url")
async def ws_analyze_image_url(websocket: WebSocket):
    """
    Client sends: {"image_url": "https://...", "prompt": "...", "max_new_tokens": 500}
    Server streams tokens, then sends: {"status": "done"}
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        image_url = data.get("image_url", "")
        prompt = data.get("prompt", "Describe this medical image. What do you see?")
        max_new_tokens = int(data.get("max_new_tokens", 500))

        if "pipe" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        resp = requests.get(image_url, headers={"User-Agent": "CuraNova"}, stream=True, timeout=15)
        resp.raise_for_status()
        image = Image.open(resp.raw).convert("RGB")

        async for token in run_inference_streaming_async(image, prompt, max_new_tokens):
            await websocket.send_text(token)
            await asyncio.sleep(0)

        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("Client disconnected from /ws/analyze/image-url")
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/analyze/image-base64")
async def ws_analyze_image_base64(websocket: WebSocket):
    """
    Client sends: {"image_b64": "<base64>", "prompt": "...", "max_new_tokens": 500}
    Server streams tokens, then sends: {"status": "done"}
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        image_b64 = data.get("image_b64", "")
        prompt = data.get("prompt", "Describe this medical image. What do you see?")
        max_new_tokens = int(data.get("max_new_tokens", 500))

        if "pipe" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        async for token in run_inference_streaming_async(image, prompt, max_new_tokens):
            await websocket.send_text(token)
            await asyncio.sleep(0)  # yield → forces immediate network flush

        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("Client disconnected from /ws/analyze/image-base64")
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# Entry point
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)