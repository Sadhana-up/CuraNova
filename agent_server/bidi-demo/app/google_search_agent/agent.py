"""CuraNova Medical Image Analysis Agent — ADK Bidi-streaming."""

import os
import json
import base64
import websockets
from typing import AsyncGenerator, Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Helper: resolve Colab WebSocket base URL
# ──────────────────────────────────────────────────────────────
def _colab_ws_base() -> Optional[str]:
    url = os.getenv("COLAB_BASE_URL", "").rstrip("/")
    if not url:
        return None
    return url.replace("https://", "wss://").replace("http://", "ws://")


# ──────────────────────────────────────────────────────────────
# Tool 1: Analyze via image URL (streaming)
# ──────────────────────────────────────────────────────────────
async def analyze_medical_image(image_url: str, prompt: str) -> AsyncGenerator[str, None]:
    """Connects to the remote medical AI model and streams back the analysis
    of a medical image token by token using its public URL.

    Use this tool when the user provides a direct image URL (e.g. https://...).

    CRITICAL: When you call this tool, STOP generating text immediately.
    Wait for the stream of tokens from the tool. Relay the output exactly
    as-is without adding your own interpretation or filler phrases.

    Args:
        image_url: The publicly accessible URL of the medical image.
        prompt: The clinical question or instruction for the analysis.
    """
    ws_base = _colab_ws_base()
    if not ws_base:
        yield "Error: COLAB_BASE_URL is not configured."
        return

    endpoint = f"{ws_base}/ws/analyze/image-url"
    payload = {"image_url": image_url, "prompt": prompt, "max_new_tokens": 500}

    try:
        with open("stream_debug.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- URL Analysis: {image_url}\nPrompt: {prompt}\n")

        async with websockets.connect(endpoint) as ws:
            await ws.send(json.dumps(payload))
            async for message in ws:
                data = json.loads(message)
                if "token" in data:
                    token = data["token"]
                    with open("stream_debug.log", "a", encoding="utf-8") as f:
                        f.write(token)
                    yield token
                elif data.get("status") == "done":
                    break
                elif "error" in data:
                    yield f"\n[Server Error]: {data['error']}"
                    break
    except Exception as e:
        yield f"\n[Connection Error]: {e}"


# ──────────────────────────────────────────────────────────────
# Tool 2: Analyze via base64 image upload (streaming)
# ──────────────────────────────────────────────────────────────
async def analyze_medical_image_upload(
    image_b64: str,
    prompt: str,
    image_b64_2: str = "",
) -> AsyncGenerator[str, None]:
    """Connects to the remote medical AI model and streams back analysis
    for one or two uploaded images encoded as base64 strings.

    Use this tool when the user uploads an image file (camera capture or file picker).
    The agent will have already extracted the base64 data from the inline_data parts.

    CRITICAL: Relay the output exactly as-is. Do NOT add filler or summarize.

    Args:
        image_b64: The primary image encoded in base64 (JPEG/PNG).
        prompt: The clinical question or instruction for the analysis.
        image_b64_2: Optional second image in base64 (leave empty if only one image).
    """
    ws_base = _colab_ws_base()
    if not ws_base:
        yield "Error: COLAB_BASE_URL is not configured."
        return

    # Route to unified endpoint when 2 images, base64 endpoint for 1
    if image_b64_2:
        endpoint = f"{ws_base}/ws/analyze/unified"
        payload = {
            "image_b64": image_b64,
            "image_b64_2": image_b64_2,
            "prompt": prompt,
            "max_new_tokens": 500,
        }
    else:
        endpoint = f"{ws_base}/ws/analyze/image-base64"
        payload = {
            "image_b64": image_b64,
            "prompt": prompt,
            "max_new_tokens": 500,
        }

    try:
        with open("stream_debug.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Upload Analysis (b64 len={len(image_b64)})\nPrompt: {prompt}\n")

        async with websockets.connect(endpoint) as ws:
            await ws.send(json.dumps(payload))
            async for message in ws:
                data = json.loads(message)
                if "token" in data:
                    token = data["token"]
                    with open("stream_debug.log", "a", encoding="utf-8") as f:
                        f.write(token)
                    yield token
                elif data.get("status") == "done":
                    break
                elif "error" in data:
                    yield f"\n[Server Error]: {data['error']}"
                    break
    except Exception as e:
        yield f"\n[Connection Error]: {e}"


# ──────────────────────────────────────────────────────────────
# Before-model callback: intercept uploaded images
#
# When the client sends an image via the ADK bidi stream
# ({ type: "image", data: "<b64>" }), it arrives as inline_data
# inside the LlmRequest. We detect it here, extract the base64
# strings, store them in callback_context.state so the agent's
# instruction can reference them, and inject a directive telling
# the agent to call analyze_medical_image_upload.
# ──────────────────────────────────────────────────────────────
async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[genai_types.Content]:
    """Detect inline image uploads and prepare the agent to call
    analyze_medical_image_upload with the extracted base64 data."""

    if not llm_request.contents:
        return None

    # Collect the last user turn parts
    last_user_content = None
    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) == "user":
            last_user_content = content
            break

    if not last_user_content:
        return None

    parts = getattr(last_user_content, "parts", []) or []
    images_b64 = []
    text_parts = []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("image/"):
            # Convert bytes to base64 string for the tool
            raw = getattr(inline, "data", b"")
            if isinstance(raw, (bytes, bytearray)):
                images_b64.append(base64.b64encode(raw).decode("utf-8"))
            elif isinstance(raw, str):
                images_b64.append(raw)
        elif getattr(part, "text", None):
            text_parts.append(part.text)

    if not images_b64:
        return None  # No images — let the model proceed normally

    # Enforce max-2 limit
    images_b64 = images_b64[:2]

    prompt = " ".join(text_parts).strip() or "Describe this medical image in detail."

    # Store in state so the agent can reference if needed
    callback_context.state["uploaded_images_b64"] = images_b64
    callback_context.state["uploaded_image_count"] = len(images_b64)
    callback_context.state["upload_prompt"] = prompt

    # Build a directive that tells the model to immediately call the upload tool
    if len(images_b64) == 2:
        directive = (
            f"The user has uploaded 2 medical images. "
            f"Call analyze_medical_image_upload with "
            f"image_b64=state['uploaded_images_b64'][0], "
            f"image_b64_2=state['uploaded_images_b64'][1], "
            f"and prompt=\"{prompt}\". "
            f"Relay the streaming output exactly as-is."
        )
    else:
        directive = (
            f"The user has uploaded a medical image. "
            f"Call analyze_medical_image_upload with "
            f"image_b64=state['uploaded_images_b64'][0] and "
            f"prompt=\"{prompt}\". "
            f"Relay the streaming output exactly as-is."
        )

    # Replace the user message inline_data with a text-only directive
    # so the model sees instructions rather than raw bytes
    new_parts = [genai_types.Part(text=directive)]

    # Patch the last user content in the request
    last_user_content.parts = new_parts

    return None  # None = proceed to LLM (with patched request)


# ──────────────────────────────────────────────────────────────
# Root Agent — Specialized Medical Image Analyst
# ──────────────────────────────────────────────────────────────
# Available models:
# - Gemini 2.0 Flash (Experimental): "gemini-2.0-flash-exp"
# - Gemini 1.5 Flash: "gemini-1.5-flash"
# - Gemini 1.5 Flash-8B: "gemini-1.5-flash-8b"
agent = Agent(
    name="CuraNovaMedicalAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.0-flash-exp"),
    before_model_callback=before_model_callback,
    instruction="""You are CuraNova, a specialized medical imaging AI assistant.

## Image Handling

### Scenario 1 — Text-only questions
Answer the user's medical question helpfully and concisely.
Always remind users that your information is for educational purposes only
and is NOT a substitute for professional medical advice.

### Scenario 2 — Image URL + prompt
If the user message contains a direct image URL (e.g. https://...), call
`analyze_medical_image` with the URL and the user's question.
Relay the streaming output EXACTLY as-is without adding your own commentary.

### Scenario 3 — Uploaded image(s) + prompt
If the user uploads image files (camera or file picker), the
`before_model_callback` has already prepared base64 strings in state and
will inject a directive. Follow that directive and call
`analyze_medical_image_upload` immediately.
Relay the streaming output EXACTLY as-is without adding your own commentary.

## Critical rules
- Do NOT preface tool calls with "Okay, analyzing..." or similar filler.
- Do NOT summarize the tool output. Relay it verbatim.
- After relaying a tool result, you may briefly offer to answer follow-up questions.
- Maximum 2 images per request are supported.
""",
    tools=[analyze_medical_image, analyze_medical_image_upload],
)
