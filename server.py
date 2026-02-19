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
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
from huggingface_hub import login
import uvicorn
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_CACHE: dict = {}
_SENTINEL = object()


def get_artifacts():
    if "model" not in MODEL_CACHE:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again in a moment.")
    return MODEL_CACHE["model"], MODEL_CACHE["processor"]


def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    login(token=hf_token)
    print("✓ Logged in to Hugging Face", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading MedGemma on {device} …", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
        token=hf_token,
    )
    model.eval()

    MODEL_CACHE["model"] = model
    MODEL_CACHE["processor"] = processor
    print(f"✓ Model loaded on: {next(model.parameters()).device}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield
    MODEL_CACHE.clear()
    print("Model unloaded.")


app = FastAPI(title="CuraNova", version="1.0.0", lifespan=lifespan)

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


# ── Shared input builder ──────────────────────────────────────

def build_inputs(model, processor, image: Optional[Image.Image], prompt: str):
    content = []
    if image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if image:
        inputs = processor(text=text_input, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text_input, return_tensors="pt")

    return inputs.to(model.device)


# ── Non-streaming inference ───────────────────────────────────

def run_inference(image: Optional[Image.Image], prompt: str, max_new_tokens: int = 500) -> str:
    model, processor = get_artifacts()
    inputs = build_inputs(model, processor, image, prompt)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Async streaming inference ─────────────────────────────────


async def run_inference_streaming_async(
    image: Optional[Image.Image],
    prompt: str,
    max_new_tokens: int = 500,
):
    model, processor = get_artifacts()
    loop = asyncio.get_running_loop()
    # Queue to bridge the sync streamer thread and the async websocket
    queue = asyncio.Queue()

    inputs = build_inputs(model, processor, image, prompt)
    print(f"[SERVER] 🔢 input_ids shape: {inputs['input_ids'].shape}", flush=True)

    # 1. Create the Streamer
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0, # Timeout if generation stalls
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=False,
    )

    # 2. Thread A: Run the Model Generation (Producer)
    # This thread runs the heavy model.generate() which blocks until finished.
    # It populates the 'streamer' object.
    def run_generation_thread():
        try:
            print("[SERVER] 🚀 Generation thread started", flush=True)
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as exc:
            print(f"[SERVER] ❌ Generation error: {exc}", flush=True)
            # We can't easily push to queue here, the consumer thread handles exceptions
            pass 
        finally:
            print("[SERVER] 🏁 Generation thread finished", flush=True)

    # 3. Thread B: Consume the Streamer (Bridge)
    # This thread iterates over the streamer (which blocks waiting for tokens)
    # and pushes them into the asyncio Queue for the WebSocket.
    def consume_streamer_thread():
        try:
            token_count = 0
            # This loop blocks waiting for new tokens from Thread A
            for token in streamer:
                token_count += 1
                loop.call_soon_threadsafe(queue.put_nowait, token)
                
                # Debug logging for first few tokens
                if token_count <= 3:
                    print(f"[SERVER] 📤 Token #{token_count}: {repr(token)}", flush=True)
            
            print(f"[SERVER] ✅ Streamer finished. Total tokens: {token_count}", flush=True)
        except Exception as exc:
            print(f"[SERVER] ❌ Streamer consumption error: {exc}", flush=True)
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            # Signal the async loop that we are done
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    # Start the threads
    t_gen = threading.Thread(target=run_generation_thread, daemon=True)
    t_gen.start()
    
    t_cons = threading.Thread(target=consume_streamer_thread, daemon=True)
    t_cons.start()

    # 4. Async Main Loop: Yield from Queue to WebSocket
    while True:
        # Wait for the next token from the queue
        item = await queue.get()
        
        if item is _SENTINEL:
            break
            
        if isinstance(item, Exception):
            # If an error occurred inside the threads, re-raise it here
            raise item
            
        yield item

        
# ── REST Routes ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "CuraNova API is running!", "model": MODEL_ID}


@app.get("/health")
async def health_check():
    model_ready = "model" in MODEL_CACHE
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "message": "Model is ready!" if model_ready else "Model is still loading, please wait ...",
    }


@app.post("/analyze", response_model=HealthResponse)
async def analyze(data: HealthRequest):
    try:
        return HealthResponse(response=run_inference(None, data.query), status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/text", response_model=HealthResponse)
async def analyze_text(data: TextAnalysisRequest):
    try:
        return HealthResponse(response=run_inference(None, data.prompt, data.max_new_tokens), status="success")
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
        return HealthResponse(response=run_inference(image, data.prompt, data.max_new_tokens), status="success")
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
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return HealthResponse(response=run_inference(image, prompt, max_new_tokens), status="success")
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
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        return HealthResponse(response=run_inference(image, prompt, max_new_tokens), status="success")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket Streaming Endpoints ─────────────────────────────

@app.websocket("/ws/analyze/text")
async def ws_analyze_text(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] 🔌 WS /text connected", flush=True)
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 500))
        print(f"[SERVER] 📩 prompt={prompt[:80]!r}", flush=True)

        if "model" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        sent = 0
        async for token in run_inference_streaming_async(None, prompt, max_new_tokens):
            await websocket.send_json({"token": token})
            sent += 1
            await asyncio.sleep(0)

        print(f"[SERVER] ✅ Sent {sent} token messages.", flush=True)
        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("[SERVER] ⚠️  Client disconnected /ws/text", flush=True)
    except Exception as e:
        print(f"[SERVER] ❌ {e}", flush=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/analyze/image-url")
async def ws_analyze_image_url(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] 🔌 WS /image-url connected", flush=True)
    try:
        data = await websocket.receive_json()
        image_url = data.get("image_url", "")
        prompt = data.get("prompt", "Describe this medical image. What do you see?")
        max_new_tokens = int(data.get("max_new_tokens", 500))
        print(f"[SERVER] 📩 url={image_url!r}", flush=True)

        if "model" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        resp = requests.get(image_url, headers={"User-Agent": "CuraNova"}, stream=True, timeout=15)
        resp.raise_for_status()
        image = Image.open(resp.raw).convert("RGB")
        print(f"[SERVER] 🖼️  Image: {image.size}", flush=True)

        sent = 0
        async for token in run_inference_streaming_async(image, prompt, max_new_tokens):
            await websocket.send_json({"token": token})
            sent += 1
            await asyncio.sleep(0)

        print(f"[SERVER] ✅ Sent {sent} token messages.", flush=True)
        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("[SERVER] ⚠️  Client disconnected /ws/image-url", flush=True)
    except Exception as e:
        print(f"[SERVER] ❌ {e}", flush=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/analyze/image-base64")
async def ws_analyze_image_base64(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] 🔌 WS /image-base64 connected", flush=True)
    try:
        data = await websocket.receive_json()
        image_b64 = data.get("image_b64", "")
        prompt = data.get("prompt", "Describe this medical image. What do you see?")
        max_new_tokens = int(data.get("max_new_tokens", 500))
        print(f"[SERVER] 📩 b64 len={len(image_b64)} prompt={prompt[:60]!r}", flush=True)

        if "model" not in MODEL_CACHE:
            await websocket.send_json({"error": "Model is not loaded yet."})
            await websocket.close()
            return

        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        print(f"[SERVER] 🖼️  Image: {image.size}", flush=True)

        sent = 0
        async for token in run_inference_streaming_async(image, prompt, max_new_tokens):
            await websocket.send_json({"token": token})
            sent += 1
            await asyncio.sleep(0)

        print(f"[SERVER] ✅ Sent {sent} token messages.", flush=True)
        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("[SERVER] ⚠️  Client disconnected /ws/image-base64", flush=True)
    except Exception as e:
        print(f"[SERVER] ❌ {e}", flush=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)