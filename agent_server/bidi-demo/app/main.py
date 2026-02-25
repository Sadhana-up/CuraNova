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
    """
    await websocket.accept()
    logger.info("[analyze_proxy] Client connected to /ws/analyze")

    try:
        request_data = await websocket.receive_text()
        logger.info(f"[analyze_proxy] Received request data: {request_data[:200]}")
        request = json.loads(request_data)

        # Allow empty or custom prompts; only fallback if key is missing entirely.
        prompt         = request.get("prompt", "")
        if not prompt:
            # Only if truly empty/None, consider a sensible default OR just leave it empty.
            # User requested removal of automatic "Describe this image" prompt.
            # We will use an empty string or whatever the client sent.
            prompt = "" 
            
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
            logger.error("[analyze_proxy] COLAB_BASE_URL not configured")
            await websocket.send_json({"error": "COLAB_BASE_URL is not configured."})
            await websocket.close()
            return

        ws_base = (
            colab_base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            .rstrip("/")
        )
        logger.info(f"[analyze_proxy] Connecting to Colab backend at: {ws_base}")

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
        
        # Try to connect to Collab
        endpoint_url = endpoint # Store for error message
        try:
             logger.info(f"[analyze_proxy] Attempting connection to {endpoint} with headers: {extra_headers}")
             async with websockets.connect(
                endpoint, 
                extra_headers=extra_headers,
                ping_interval=None,
                max_size=None,
                open_timeout=10, # Add timeout to prevent indefinite hanging
            ) as colab_ws:
                logger.info("[analyze_proxy] Connected to Colab backend!")
                logger.info(f"[analyze_proxy] Sending payload: {json.dumps(payload)[:100]}...")
                await colab_ws.send(json.dumps(payload))
                logger.info("[analyze_proxy] Payload sent, waiting for messages...")
                
                async for message in colab_ws:
                    # Optimize: send raw message to client immediately to reduce latency
                    await websocket.send_text(message)
                    
                    # Quick check for termination conditions without full parse
                    if '"status": "done"' in message or '"status":"done"' in message:
                        break
                    if '"error":' in message:
                        break
                logger.info("[analyze_proxy] Stream finished normally")

        except TypeError:
             # Fallback for newer websockets that might not accept extra_headers as kwarg 
             # (though usually they do specific handling, let's try additional_headers if checking version)
             # But actually, passing it in strict format for modern websockets:
             logger.warning("[analyze_proxy] TypeError connecting with extra_headers, retrying with additional_headers...")
             async with websockets.connect(
                endpoint, 
                additional_headers=extra_headers,
                ping_interval=None,
                max_size=None,
                open_timeout=10,
            ) as colab_ws:
                logger.info("[analyze_proxy] Connected to Colab backend (retry)!")
                await colab_ws.send(json.dumps(payload))
                async for message in colab_ws:
                    # Optimize: send raw message to client immediately to reduce latency
                    await websocket.send_text(message)
                    
                    if '"status": "done"' in message or '"status":"done"' in message:
                        break
                    if '"error":' in message:
                        break
                logger.info("[analyze_proxy] Stream finished normally (retry)")

    except WebSocketDisconnect:
        logger.debug("[analyze_proxy] Client disconnected")
    except Exception as e:
        logger.error(f"[analyze_proxy] Error connecting to {endpoint_url}: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": f"Failed to reach Collab backend: {str(e)}"})
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
                # Found signal in this part, no need to continue parsing this part
                # break is not needed here as we use found_in_this_part flag
            
            if not found_in_this_part and _SIGNAL_PREFIX in text:
                # Fallback: take everything after prefix if structured extraction fails
                prefix_idx = text.index(_SIGNAL_PREFIX)
                raw_payload = text[prefix_idx + len(_SIGNAL_PREFIX):]
                
                signal_payload = raw_payload.strip()
                cleaned_text = text[:prefix_idx].strip()
                if cleaned_text:
                    new_parts.append({**part, "text": cleaned_text})
                else:
                    new_parts.append({**part, "text": " "}) # Preserve empty part to avoid validation errors

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
        session = await session_service.create_session(
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
                    # ──────────────────────────────────────────────────────────
                    # MANUAL MANIPULATION (Bypassing before_model_callback)
                    # ──────────────────────────────────────────────────────────
                    # Since the callback isn't firing reliably in streaming mode,
                    # we manually inject the image into the session state here
                    # and rewrite the message as a text directive.
                    # ──────────────────────────────────────────────────────────
                    
                    image_b64 = json_message["data"]
                    # If prompt is empty, let it be empty. Do not force "Describe this image."
                    prompt_text = json_message.get("prompt", "")
                    logger.info(f"[IMAGE UPLOAD] Received image (len={len(image_b64)}). Manually storing in session state.")
                    
                    # Log first 50 chars of base64 to check for validity/prefix
                    logger.info(f"[IMAGE UPLOAD] Base64 start: {image_b64[:50]}")

                    # update session state
                    if session:
                        # Append to existing images or start new list
                        current_images = session.state.get("uploaded_images_b64", [])
                        if not isinstance(current_images, list):
                            current_images = []
                        
                        current_images.append(image_b64)
                        # Keep max 2
                        current_images = current_images[-2:]
                        
                        session.state["uploaded_images_b64"] = current_images
                        session.state["uploaded_image_count"] = len(current_images)
                        session.state["upload_prompt"] = prompt_text
                        logger.info(f"[IMAGE UPLOAD] Session state updated. Total images: {len(current_images)}")
                        
                        # ──────────────────────────────────────────────────────
                        # Fallback: Save to file system for reliability
                        # ──────────────────────────────────────────────────────
                        try:
                            # Ensure directory exists (with explicit Windows path handling if needed)
                            fallback_dir = Path("d:/curanova/uploaded_images")
                            fallback_dir.mkdir(parents=True, exist_ok=True)
                            
                            fallback_path = fallback_dir / f"{session_id}.json"
                            
                            # Read existing if any
                            existing_data = []
                            if fallback_path.exists():
                                try:
                                    with open(fallback_path, "r", encoding="utf-8") as f:
                                        existing_data = json.load(f)
                                except Exception:
                                    pass
                            
                            if not isinstance(existing_data, list):
                                existing_data = []
                                
                            existing_data.append(image_b64)
                            # Keep max 2
                            existing_data = existing_data[-2:]
                            
                            # Ensure the file is written completely
                            with open(fallback_path, "w", encoding="utf-8") as f:
                                json.dump(existing_data, f)
                                f.flush()
                                os.fsync(f.fileno())
                                
                            logger.info(f"[IMAGE UPLOAD] Saved to fallback file: {fallback_path}")
                        except Exception as e:
                            logger.error(f"[IMAGE UPLOAD] Failed to save fallback file: {e}")

                    else:
                        logger.error("[IMAGE UPLOAD] Session object is None! Cannot store image.")

                    # Construct directive for the agent
                    # Ensure we use session state for count
                    count = len(session.state.get("uploaded_images_b64", [])) if session and session.state else 1
                    
                    # We inject the image data directly into the tool call instructions just in case state persistence fails
                    # We'll use a unique variable name to avoid confusion
                    img_data_snippet = image_b64[:20] + "..." + image_b64[-20:]
                    logger.info(f"DEBUG: Constructing directive with image snippet: {img_data_snippet}")

                    directive = (
                        f"The user sent {count} image(s) with this message: \"{prompt_text}\"\n"
                        f"I have manually injected the image data into your session state under key 'uploaded_images_b64'.\n"
                        f"I have ALSO saved a backup of the image(s) to a file associated with session_id='{session_id}'.\n"
                        
                        f"\nDecide: does this image require a CLINICAL / MEDICAL analysis?\n"
                        f"- If YES → call analyze_medical_image_upload(prompt=\"{prompt_text}\", session_id=\"{session_id}\").\n"
                        f"  (The tool will automatically retrieve images from session state or the fallback file).\n"
                        f"- If NO  → describe the image(s) using your own vision capability and answer normally.\n"
                        f"\nIMPORTANT: Do not try to access state['uploaded_images_b64'] yourself."
                    )

                    # Send as MULTIMODAL content (Text + Image) so the agent can SEE it
                    # We send both the text directive AND the image bytes in the same turn.
                    # This allows the model to "see" the image for general description,
                    # while the directive guides it on whether to use the medical tool.
                    
                    # LOGGING FOR DEBUGGING
                    logger.info(f"DEBUG: Injecting directive + image for upload. Prompt: {prompt_text}")
                    logger.info(f"DEBUG: Session state keys now: {list(session.state.keys()) if session else 'NO SESSION'}")

                    try:
                        # Create the image part
                        image_bytes = base64.b64decode(image_b64)
                        image_part = types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",  # Assuming jpeg from client
                                data=image_bytes
                            )
                        )
                        
                        # Create the text directive part
                        text_part = types.Part(text=directive)
                        
                        # content = types.Content(parts=[types.Part(text=directive)]) -- OLD
                        content = types.Content(parts=[text_part, image_part])
                        
                        # The ADK runner processes items from live_request_queue.
                        # send_content sends a Content object.
                        live_request_queue.send_content(content)
                        logger.info(f"DEBUG: Directive + Image sent to live_request_queue successfully.")
                    except Exception as e:
                        logger.error(f"DEBUG: Failed to send directive/image to queue: {e}")
                    logger.info(f"DEBUG: Session state keys now: {list(session.state.keys()) if session else 'NO SESSION'}")
                    if session and "uploaded_images_b64" in session.state:
                         imgs = session.state["uploaded_images_b64"]
                         logger.info(f"DEBUG: stored image count: {len(imgs)}")
                         if len(imgs) > 0:
                              logger.info(f"DEBUG: First image b64 len: {len(imgs[0])}")

                    try:
                        content = types.Content(parts=[types.Part(text=directive)])
                        # The ADK runner processes items from live_request_queue.
                        # send_content sends a Content object.
                        # live_request_queue.send_content(content) -- This was the old one, we replaced it above
                        pass
                        logger.info(f"DEBUG: Directive sent to live_request_queue successfully (stub).")
                    except Exception as e:
                        logger.error(f"DEBUG: Failed to send directive to queue: {e}")


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
                
                # LOGGING: Write full event to a separate debug file for inspection
                with open("adk_events_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"\n--- EVENT {event.timestamp} ---\n{event_json}\n")

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
        
        # When downstream finishes, close the queue to release the upstream
        live_request_queue.close()
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
