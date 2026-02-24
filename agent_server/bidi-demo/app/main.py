"""FastAPI application — ADK Bidi-streaming with medical analysis trigger.

Flow
────
1.  All client messages (text, image, URL) travel through the main ADK
    WebSocket (/ws/{user_id}/{session_id}).

2.  The Gemini agent decides whether to call a medical analysis tool based
    on the content of the message.

3.  When a tool IS called, it returns a special signal string starting with
    "__MEDICAL_STREAM__:" followed by a JSON payload.

4.  The downstream_task in main.py detects this signal inside any incoming
    ADK event, strips it out, and forwards a `medical_stream_trigger` event
    to the Next.js client.

5.  The Next.js client receives the trigger, opens its own WebSocket to
    /ws/analyze, and streams the medical analysis into the green bubble —
    exactly as before, but now entirely under agent control.
"""

import asyncio
import base64
import json
import logging
import os
import warnings
from pathlib import Path

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load environment variables BEFORE importing agent
load_dotenv(Path(__file__).parent / ".env")

# pylint: disable=wrong-import-position
from google_search_agent.agent import agent, _SIGNAL_PREFIX  # noqa: E402

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "bidi-demo"

# ── App & services ────────────────────────────────────────────────────────────
app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ── Proxy endpoint (kept for backward compat / direct calls) ─────────────────

@app.websocket("/ws/analyze")
async def analyze_proxy(websocket: WebSocket) -> None:
    """WebSocket proxy — streams tokens from the Colab medical AI server.

    Handles ALL scenarios:
      1) Text-only prompt            → /ws/analyze/text
      2) Image URL + prompt          → /ws/analyze/image-url
      3) Image upload (base64)       → /ws/analyze/image-base64
      Mixed / multi-image            → /ws/analyze/unified

    JSON payload (all image fields optional):
    {
      "prompt":       "...",
      "max_new_tokens": 500,
      "image_url":    "https://...",
      "image_url_2":  "https://...",
      "image_b64":    "<base64>",
      "image_b64_2":  "<base64>"
    }
    """
    await websocket.accept()

    try:
        request_data = await websocket.receive_text()
        request = json.loads(request_data)

        prompt         = request.get("prompt", "Describe this medical image in detail.")
        max_new_tokens = request.get("max_new_tokens", 500)
        image_url      = request.get("image_url", "")
        image_url_2    = request.get("image_url_2", "")
        image_b64      = request.get("image_b64", "")
        image_b64_2    = request.get("image_b64_2", "")

        logger.info(
            f"[analyze_proxy] prompt={prompt[:60]!r} "
            f"url1={'yes' if image_url else 'no'} url2={'yes' if image_url_2 else 'no'} "
            f"b64_1={'yes' if image_b64 else 'no'} b64_2={'yes' if image_b64_2 else 'no'}"
        )

        colab_base_url = os.getenv("COLAB_BASE_URL", "")
        if not colab_base_url:
            await websocket.send_json({"error": "COLAB_BASE_URL is not configured."})
            await websocket.close()
            return

        ws_base = (
            colab_base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            .rstrip("/")
        )

        has_b64  = bool(image_b64 or image_b64_2)
        has_url  = bool(image_url or image_url_2)
        is_mixed = (has_b64 and has_url) or bool(image_b64 and image_b64_2) or bool(image_url and image_url_2)

        if is_mixed:
            endpoint = f"{ws_base}/ws/analyze/unified"
            payload = {
                "prompt": prompt, "max_new_tokens": max_new_tokens,
                "image_url": image_url, "image_url_2": image_url_2,
                "image_b64": image_b64, "image_b64_2": image_b64_2,
            }
            logger.info(f"[analyze_proxy] → unified (endpoint: {endpoint})")
        elif has_b64:
            endpoint = f"{ws_base}/ws/analyze/image-base64"
            payload = {"image_b64": image_b64, "prompt": prompt, "max_new_tokens": max_new_tokens}
            logger.info(f"[analyze_proxy] → image-base64 (endpoint: {endpoint})")
        elif has_url:
            endpoint = f"{ws_base}/ws/analyze/image-url"
            payload = {"image_url": image_url, "prompt": prompt, "max_new_tokens": max_new_tokens}
            logger.info(f"[analyze_proxy] → image-url (endpoint: {endpoint})")
        else:
            endpoint = f"{ws_base}/ws/analyze/text"
            payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
            logger.info(f"[analyze_proxy] → text (endpoint: {endpoint})")

        # Connect with ngrok skip header to bypass free-tier interstitial page
        extra_headers = {
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "CuraNovaProxy/1.0"
        }
        
        logger.info(f"[analyze_proxy] Connecting to {endpoint} w/ headers...")
        
        # Determine strict header argument based on websockets version or trial
        # For newer websockets (13+), additional_headers might be preferred or extra_headers
        # But if extra_headers is failing at create_connection, it means it's being passed as a kwarg to the loop
        # We will try 'additional_headers' if available, or just pass headers in the standard way
        
        connection_args = {
            "ping_interval": None,
            "max_size": None
        }
        
        # Try to use 'additional_headers' which is often the safe way to pass headers in newer libs
        # without them being passed to create_connection
        async with websockets.connect(
            endpoint, 
            additional_headers=extra_headers,
            **connection_args
        ) as colab_ws:
            logger.info("[analyze_proxy] Connected to Colab backend!")
            await colab_ws.send(json.dumps(payload))
            async for message in colab_ws:
                data = json.loads(message)
                if "token" in data:
                    await websocket.send_json({"token": data["token"]})
                elif data.get("status") == "done":
                    await websocket.send_json({"status": "done"})
                    break
                elif "error" in data:
                    await websocket.send_json({"error": data["error"]})
                    break

    except WebSocketDisconnect:
        logger.debug("[analyze_proxy] Client disconnected")
    except Exception as e:
        logger.error(f"[analyze_proxy] Error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Helper: extract signal payload from ADK events ───────────────────────────

def _extract_signal(event_json: str) -> tuple[str | None, str]:
    """Scan an ADK event JSON for a __MEDICAL_STREAM__ signal.

    We use a multi-stage approach to ensure we never miss a signal:
      1. Structured search through content parts (text and functionResponse).
      2. If not found, a 'Nuclear' raw string scan of the entire JSON blob.

    Returns:
        (signal_payload_json, cleaned_event_json)
    """
    try:
        event = json.loads(event_json)
    except json.JSONDecodeError:
        return None, event_json

    signal_payload: str | None = None
    modified = False

    content = event.get("content") or {}
    parts = content.get("parts") or []

    # ── Stage 1: Structured Search ───────────────────────────────────────────
    new_parts = []
    for part in parts:
        found_in_this_part = False

        # Check text
        text = part.get("text") or ""
        if _SIGNAL_PREFIX in text:
            prefix_idx = text.index(_SIGNAL_PREFIX)
            # Try to extract JSON payload carefully to preserve surrounding text
            raw_payload = text[prefix_idx + len(_SIGNAL_PREFIX):]
            json_start = raw_payload.find("{")
            
            extracted_json = None
            extracted_end_idx = -1

            if json_start != -1:
                depth = 0
                for i, ch in enumerate(raw_payload[json_start:], start=json_start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            extracted_json = raw_payload[json_start : i + 1]
                            extracted_end_idx = i + 1
                            break
            
            if extracted_json:
                signal_payload = extracted_json
                # Reconstruct text without the signal part
                text_before = text[:prefix_idx].strip()
                text_after = raw_payload[extracted_end_idx:].strip()
                cleaned_text = (text_before + " " + text_after).strip()
                
                if cleaned_text:
                    new_parts.append({**part, "text": cleaned_text})
                
                modified = True
                found_in_this_part = True
                logger.info(f"[SIGNAL] Found in text part. Payload: {signal_payload[:60]}")
            else:
                # Fallback: take everything after prefix if JSON parsing fails here
                # (downstream task might still try to parse it, but text is lost)
                signal_payload = raw_payload.strip()
                cleaned_text = text[:prefix_idx].strip()
                if cleaned_text:
                    new_parts.append({**part, "text": cleaned_text})
                modified = True
                found_in_this_part = True
                logger.info(f"[SIGNAL] Found in text part (fallback). Payload starts: {signal_payload[:60]}")

        # Check functionResponse
        if not found_in_this_part:
            func_resp = part.get("functionResponse") or {}
            if func_resp:
                response_obj = func_resp.get("response") or {}
                result = response_obj.get("result") or response_obj.get("output") or ""
                
                # Handle string result
                if isinstance(result, str) and _SIGNAL_PREFIX in result:
                    idx = result.index(_SIGNAL_PREFIX) + len(_SIGNAL_PREFIX)
                    signal_payload = result[idx:].strip()
                    modified = True
                    found_in_this_part = True
                    logger.info(f"[SIGNAL] Found in functionResponse (str). Payload starts: {signal_payload[:60]}")
                
                # Handle dict result
                elif isinstance(result, dict):
                    inner = result.get("result") or result.get("output") or ""
                    if isinstance(inner, str) and _SIGNAL_PREFIX in inner:
                        idx = inner.index(_SIGNAL_PREFIX) + len(_SIGNAL_PREFIX)
                        signal_payload = inner[idx:].strip()
                        modified = True
                        found_in_this_part = True
                        logger.info(f"[SIGNAL] Found in functionResponse (dict). Payload starts: {signal_payload[:60]}")

        if not found_in_this_part:
            new_parts.append(part)
    
    if modified:
        # Commit the modification to the event object
        if "content" not in event:
            event["content"] = {}
        event["content"]["parts"] = new_parts

    # ── Stage 2: Nuclear Raw String Fallback ─────────────────────────────────
    if not modified and _SIGNAL_PREFIX in event_json:
        logger.warning("[SIGNAL] Nuclear fallback triggered! Signal found in raw JSON but not in expected parts.")
        idx = event_json.index(_SIGNAL_PREFIX) + len(_SIGNAL_PREFIX)
        # Try to grab the JSON payload by matching braces
        raw_tail = event_json[idx:].strip()
        if raw_tail.startswith("{"):
            depth = 0
            for i, ch in enumerate(raw_tail):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        signal_payload = raw_tail[:i+1]
                        break
        
        # If we found a signal via fallback, we must 'clean' the event strictly
        # so we don't leak logic to the UI. We'll return an empty event if it's messy.
        if signal_payload:
            modified = True
            # Build a safe dummy event representing the turn's existence
            event["content"] = {"parts": [{"text": ""}]} 

    if modified:
        return signal_payload, json.dumps(event)

    return None, event_json




# ── Main ADK WebSocket Endpoint ───────────────────────────────────────────────

@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    proactivity: bool = False,
    affective_dialog: bool = False,
) -> None:
    """Bidirectional streaming endpoint.

    All client messages go here. The agent lives inside this connection and
    has full conversation context across all turns.
    """
    logger.debug(
        f"WebSocket connection: user_id={user_id}, session_id={session_id}"
    )
    await websocket.accept()

    # ── Determine response modality ───────────────────────────────────────────
    model_name = agent.model
    is_native_audio = "native-audio" in model_name.lower()

    if is_native_audio:
        response_modalities = ["AUDIO"]
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=response_modalities,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
            proactivity=(
                types.ProactivityConfig(proactive_audio=True) if proactivity else None
            ),
            enable_affective_dialog=affective_dialog if affective_dialog else None,
        )
    else:
        response_modalities = ["TEXT"]
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=response_modalities,
            input_audio_transcription=None,
            output_audio_transcription=None,
            session_resumption=types.SessionResumptionConfig(),
        )
        if proactivity or affective_dialog:
            logger.warning(
                f"Proactivity/affective dialog only work with native audio models. "
                f"Current model: {model_name}. Settings ignored."
            )

    # ── Session ───────────────────────────────────────────────────────────────
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    # ── Upstream: client → agent ──────────────────────────────────────────────
    async def upstream_task() -> None:
        logger.debug("upstream_task started")
        while True:
            message = await websocket.receive()

            # Binary frame → audio
            if "bytes" in message:
                audio_data = message["bytes"]
                audio_blob = types.Blob(mime_type="audio/pcm;rate=16000", data=audio_data)
                live_request_queue.send_realtime(audio_blob)

            # Text frame → JSON
            elif "text" in message:
                json_message = json.loads(message["text"])

                if json_message.get("type") == "text":
                    # Plain text message — goes straight to agent
                    content = types.Content(
                        parts=[types.Part(text=json_message["text"])]
                    )
                    live_request_queue.send_content(content)
                    logger.debug(f"Sent text to agent: {json_message['text'][:80]}")

                elif json_message.get("type") == "image":
                    # Uploaded image — wrap as inline_data so before_model_callback
                    # can intercept it and the agent can decide whether to analyze it
                    image_data = base64.b64decode(json_message["data"])
                    mime_type = json_message.get("mimeType", "image/jpeg")
                    prompt_text = json_message.get("prompt", "")

                    parts = [
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=image_data,
                            )
                        )
                    ]
                    if prompt_text:
                        parts.append(types.Part(text=prompt_text))

                    content = types.Content(parts=parts)
                    live_request_queue.send_content(content)
                    logger.debug(
                        f"Sent image to agent: {len(image_data)} bytes, type={mime_type}"
                    )

    # ── Downstream: agent → client ────────────────────────────────────────────
    async def downstream_task() -> None:
        logger.debug("downstream_task started")
        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                event_json = event.model_dump_json(exclude_none=True, by_alias=True)
                # Log the full event at INFO so we can see tool responses in the console
                logger.info(f"[ADK EVENT] {event_json[:500]}")

                # Check for medical analysis signal embedded in tool output
                signal_payload, clean_event_json = _extract_signal(event_json)

                if signal_payload is not None:
                    logger.info(f"[SIGNAL TRIGGER] payload: {signal_payload[:200]}")
                    # Safely parse the JSON payload (may have trailing text)
                    try:
                        payload_dict = json.loads(signal_payload)
                    except json.JSONDecodeError:
                        # Try extracting just the first complete JSON object
                        depth = 0
                        end = 0
                        for i, ch in enumerate(signal_payload):
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                        try:
                            payload_dict = json.loads(signal_payload[:end])
                        except json.JSONDecodeError as exc:
                            logger.error(f"[SIGNAL] Could not parse payload: {exc}")
                            await websocket.send_text(event_json)
                            continue

                    # Forward the trigger to the client so it opens /ws/analyze
                    await websocket.send_text(
                        json.dumps({
                            "medical_stream_trigger": True,
                            "payload": payload_dict,
                        })
                    )
                    # Also forward the cleaned agent event (e.g. "processing…" text)
                    # Only if it still has meaningful content
                    cleaned = json.loads(clean_event_json) if clean_event_json else {}
                    inner_parts = (cleaned.get("content") or {}).get("parts") or []
                    if any(p.get("text") for p in inner_parts):
                        await websocket.send_text(clean_event_json)
                else:
                    await websocket.send_text(event_json)

        except Exception as e:
            # Handle normal closure (1000) or other errors
            if "1000" in str(e):
                logger.info("Normal closure from Gemini (1000).")
            else:
                logger.error(f"Error in downstream_task: {e}", exc_info=True)
                # Optionally send error to client
                try:
                    await websocket.send_json({"error": "Agent connection error"})
                except:
                    pass

        logger.debug("run_live() generator completed")


    # ── Run both tasks concurrently ───────────────────────────────────────────
    try:
        await asyncio.gather(upstream_task(), downstream_task())
    except WebSocketDisconnect:
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        live_request_queue.close()
