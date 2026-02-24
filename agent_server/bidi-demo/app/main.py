"""FastAPI application demonstrating ADK Bidi-streaming with WebSocket."""

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

# Load environment variables from .env file BEFORE importing agent
load_dotenv(Path(__file__).parent / ".env")

# Import agent after loading environment variables
# pylint: disable=wrong-import-position
from google_search_agent.agent import agent  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Application name constant
APP_NAME = "bidi-demo"

# ========================================
# Phase 1: Application Initialization (once at startup)
# ========================================

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define your session service
session_service = InMemorySessionService()

# Define your runner
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

# ========================================
# HTTP Endpoints
# ========================================


@app.get("/")
async def root():
    """Serve the index.html page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ========================================
# Direct Streaming Proxy for Colab Medical AI
# ========================================


@app.websocket("/ws/analyze")
async def analyze_proxy(websocket: WebSocket) -> None:
    """WebSocket proxy that streams tokens from the Colab medical AI server.

    Handles ALL three scenarios:
      1) Text-only prompt            → routes to /ws/analyze/text
      2) Image URL + prompt          → routes to /ws/analyze/image-url
      3) Image upload (base64/file)  → routes to /ws/analyze/image-base64
      Mixed / multi-image            → routes to /ws/analyze/unified

    Expected JSON payload (all image fields are optional):
    {
      "prompt":       "...",
      "max_new_tokens": 500,
      "image_url":    "https://...",   // optional, first image URL
      "image_url_2":  "https://...",   // optional, second image URL
      "image_b64":    "<base64>",      // optional, uploaded file 1
      "image_b64_2":  "<base64>"       // optional, uploaded file 2
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

        # Build Colab WebSocket base URL
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

        # ── Smart routing ────────────────────────────────────────
        if is_mixed:
            endpoint = f"{ws_base}/ws/analyze/unified"
            payload = {
                "prompt": prompt, "max_new_tokens": max_new_tokens,
                "image_url": image_url, "image_url_2": image_url_2,
                "image_b64": image_b64, "image_b64_2": image_b64_2,
            }
            logger.info("[analyze_proxy] → unified")

        elif has_b64:
            endpoint = f"{ws_base}/ws/analyze/image-base64"
            payload = {
                "image_b64": image_b64,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            logger.info("[analyze_proxy] → image-base64")

        elif has_url:
            endpoint = f"{ws_base}/ws/analyze/image-url"
            payload = {
                "image_url": image_url,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            logger.info("[analyze_proxy] → image-url")

        else:
            endpoint = f"{ws_base}/ws/analyze/text"
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            logger.info("[analyze_proxy] → text")

        # ── Stream from Colab → client ───────────────────────────
        async with websockets.connect(endpoint) as colab_ws:
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


# ========================================
# WebSocket Endpoint
# ========================================


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    proactivity: bool = False,
    affective_dialog: bool = False,
) -> None:
    """WebSocket endpoint for bidirectional streaming with ADK.

    Args:
        websocket: The WebSocket connection
        user_id: User identifier
        session_id: Session identifier
        proactivity: Enable proactive audio (native audio models only)
        affective_dialog: Enable affective dialog (native audio models only)
    """
    logger.debug(
        f"WebSocket connection request: user_id={user_id}, session_id={session_id}, "
        f"proactivity={proactivity}, affective_dialog={affective_dialog}"
    )
    await websocket.accept()
    logger.debug("WebSocket connection accepted")

    # ========================================
    # Phase 2: Session Initialization (once per streaming session)
    # ========================================

    # Automatically determine response modality based on model architecture
    # Native audio models (containing "native-audio" in name)
    # ONLY support AUDIO response modality.
    # Half-cascade models support both TEXT and AUDIO,
    # we default to TEXT for better performance.
    model_name = agent.model
    is_native_audio = "native-audio" in model_name.lower()

    if is_native_audio:
        # Native audio models require AUDIO response modality
        # with audio transcription
        response_modalities = ["AUDIO"]

        # Build RunConfig with optional proactivity and affective dialog
        # These features are only supported on native audio models
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=response_modalities,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
            proactivity=(
                types.ProactivityConfig(proactive_audio=True)
                if proactivity
                else None
            ),
            enable_affective_dialog=affective_dialog
            if affective_dialog
            else None,
        )
        logger.debug(
            f"Native audio model detected: {model_name}, "
            f"using AUDIO response modality, "
            f"proactivity={proactivity}, affective_dialog={affective_dialog}"
        )
    else:
        # Half-cascade models support TEXT response modality
        # for faster performance
        response_modalities = ["TEXT"]
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=response_modalities,
            input_audio_transcription=None,
            output_audio_transcription=None,
            session_resumption=types.SessionResumptionConfig(),
        )
        logger.debug(
            f"Half-cascade model detected: {model_name}, "
            "using TEXT response modality"
        )
        # Warn if user tried to enable native-audio-only features
        if proactivity or affective_dialog:
            logger.warning(
                f"Proactivity and affective dialog are only supported on native "
                f"audio models. Current model: {model_name}. "
                f"These settings will be ignored."
            )
    logger.debug(f"RunConfig created: {run_config}")

    # Get or create session (handles both new sessions and reconnections)
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    # ========================================
    # Phase 3: Active Session (concurrent bidirectional communication)
    # ========================================

    async def upstream_task() -> None:
        """Receives messages from WebSocket and sends to LiveRequestQueue."""
        logger.debug("upstream_task started")
        while True:
            # Receive message from WebSocket (text or binary)
            message = await websocket.receive()

            # Handle binary frames (audio data)
            if "bytes" in message:
                audio_data = message["bytes"]
                logger.debug(
                    f"Received binary audio chunk: {len(audio_data)} bytes"
                )

                audio_blob = types.Blob(
                    mime_type="audio/pcm;rate=16000", data=audio_data
                )
                live_request_queue.send_realtime(audio_blob)

            # Handle text frames (JSON messages)
            elif "text" in message:
                text_data = message["text"]
                logger.debug(f"Received text message: {text_data[:100]}...")

                json_message = json.loads(text_data)

                # Extract text from JSON and send to LiveRequestQueue
                if json_message.get("type") == "text":
                    logger.debug(
                        f"Sending text content: {json_message['text']}"
                    )
                    content = types.Content(
                        parts=[types.Part(text=json_message["text"])]
                    )
                    live_request_queue.send_content(content)

                # Handle image data
                elif json_message.get("type") == "image":
                    logger.debug("Received image data")

                    # Decode base64 image data
                    image_data = base64.b64decode(json_message["data"])
                    mime_type = json_message.get("mimeType", "image/jpeg")

                    logger.debug(
                        f"Sending image: {len(image_data)} bytes, "
                        f"type: {mime_type}"
                    )

                    # Send image as Content so text-mode models can see it
                    content = types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=image_data,
                                )
                            ),
                            types.Part(
                                text="The user has shared an image. Please analyze and describe what you see."
                            ),
                        ]
                    )
                    live_request_queue.send_content(content)

    async def downstream_task() -> None:
        """Receives Events from run_live() and sends to WebSocket."""
        logger.debug("downstream_task started, calling runner.run_live()")
        logger.debug(
            f"Starting run_live with user_id={user_id}, session_id={session_id}"
        )
        async for event in runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            event_json = event.model_dump_json(exclude_none=True, by_alias=True)
            logger.debug(f"[SERVER] Event: {event_json}")
            await websocket.send_text(event_json)
        logger.debug("run_live() generator completed")

    # Run both tasks concurrently
    # Exceptions from either task will propagate and cancel the other task
    try:
        logger.debug(
            "Starting asyncio.gather for upstream and downstream tasks"
        )
        await asyncio.gather(upstream_task(), downstream_task())
        logger.debug("asyncio.gather completed normally")
    except WebSocketDisconnect:
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error in streaming tasks: {e}", exc_info=True)
    finally:
        # ========================================
        # Phase 4: Session Termination
        # ========================================

        # Always close the queue, even if exceptions occurred
        logger.debug("Closing live_request_queue")
        live_request_queue.close()
