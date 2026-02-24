"""CuraNova Medical Image Analysis Agent — ADK Bidi-streaming.

Architecture
────────────
ALL messages (text, URL, uploaded image) go through the main ADK WebSocket.
The Gemini agent reads every turn and decides autonomously whether a tool call
is warranted.

When the agent decides to run a medical analysis tool, the tool:
  1. Immediately returns a JSON "signal" string that tells main.py to push a
     special `medical_stream_trigger` event to the Next.js client.
  2. The Next.js client receives that event and opens its own direct WebSocket
     to /ws/analyze, streaming tokens into the green "Medical Analysis" bubble —
     exactly as before, but now fully under agent control.

The agent keeps full conversation context (memory) throughout.
"""

import base64
import json
import os
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# SIGNAL PREFIX — main.py watches for this prefix in tool output
# to know it must forward a medical_stream_trigger event.
# ──────────────────────────────────────────────────────────────
_SIGNAL_PREFIX = "__MEDICAL_STREAM__:"


# ──────────────────────────────────────────────────────────────
# Tool 1: Trigger analysis via public image URL
# ──────────────────────────────────────────────────────────────
def analyze_medical_image(image_url: str, prompt: str) -> str:
    """Triggers the CuraNova medical AI to analyse a medical image given its
    public URL.

    Call this tool whenever the user provides a direct image URL
    (starting with http:// or https://) AND asks for a clinical or medical
    analysis of that image.

    Do NOT call this tool for general image descriptions or non-medical
    questions — answer those directly.

    Args:
        image_url: The publicly accessible URL of the medical image.
        prompt: The clinical question or instruction for the analysis.

    Returns:
        A signal string that the backend uses to stream the analysis to the
        client. After this call, tell the user their request is being
        processed by the medical agent.
    """
    payload = {
        "image_url": image_url,
        "prompt": prompt,
        "max_new_tokens": 500,
    }
    signal = _SIGNAL_PREFIX + json.dumps(payload)
    return signal


# ──────────────────────────────────────────────────────────────
# Tool 2: Trigger analysis for uploaded image(s)
# ──────────────────────────────────────────────────────────────
def analyze_medical_image_upload(
    image_b64: str,
    prompt: str,
    image_b64_2: str = "",
) -> str:
    """Triggers the CuraNova medical AI to analyse one or two uploaded images
    that were encoded as base64 strings.

    Call this tool ONLY when:
      - The user has uploaded an image (camera or file picker), AND
      - The user is asking for a clinical / medical analysis of that image.

    Do NOT call this tool if the user shares a selfie or a non-medical image
    and simply wants a description — answer those directly using your own
    vision capability.

    Args:
        image_b64: Primary image encoded as a base64 string (JPEG/PNG).
        prompt: The clinical question or instruction from the user.
        image_b64_2: Optional second image in base64 (leave empty for one image).

    Returns:
        A signal string that the backend uses to stream the analysis to the
        client. After this call, tell the user their request is being
        processed by the medical agent.
    """
    payload: dict = {
        "image_b64": image_b64,
        "prompt": prompt,
        "max_new_tokens": 500,
    }
    if image_b64_2:
        payload["image_b64_2"] = image_b64_2

    signal = _SIGNAL_PREFIX + json.dumps(payload)
    return signal


# ──────────────────────────────────────────────────────────────
# Before-model callback — intercept uploaded images
#
# When the client sends `{ type: "image", data: "<b64>" }` through the
# bidi stream, main.py wraps it as inline_data on a Content object and
# places it in the LiveRequestQueue. This callback intercepts that Content
# before Gemini sees it, extracts the base64 bytes, stores them in
# callback_context.state, and rewrites the user message as a text-only
# directive so Gemini can decide whether a medical analysis is needed.
# ──────────────────────────────────────────────────────────────
async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[genai_types.Content]:
    """Detect inline image uploads and expose base64 data to the agent via
    a text directive — the agent then decides whether to call a tool."""

    if not llm_request.contents:
        return None

    # Find the last user turn
    last_user_content = None
    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) == "user":
            last_user_content = content
            break

    if not last_user_content:
        return None

    parts = getattr(last_user_content, "parts", []) or []
    images_b64: list[str] = []
    text_parts: list[str] = []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("image/"):
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

    prompt = " ".join(text_parts).strip() or "Describe this image."

    # Store base64 data in session state for the agent to reference
    callback_context.state["uploaded_images_b64"] = images_b64
    callback_context.state["uploaded_image_count"] = len(images_b64)
    callback_context.state["upload_prompt"] = prompt

    # Build indexes into state for the tool-call directive
    if len(images_b64) == 2:
        image_args = (
            f"image_b64=\"{images_b64[0][:30]}...\" "
            f"image_b64_2=\"{images_b64[1][:30]}...\" "
        )
    else:
        image_args = f"image_b64=\"{images_b64[0][:30]}...\" "

    directive = (
        f"The user sent {len(images_b64)} image(s) with this message: \"{prompt}\"\n\n"
        f"The full base64 image data is stored in session state:\n"
        f"  state['uploaded_images_b64'][0] = (primary image, {len(images_b64[0])} chars)\n"
        + (f"  state['uploaded_images_b64'][1] = (second image, {len(images_b64[1])} chars)\n" if len(images_b64) == 2 else "")
        + f"\nDecide: does this image require a CLINICAL / MEDICAL analysis?\n"
        f"- If YES → call analyze_medical_image_upload with the full base64 string(s) from state.\n"
        f"- If NO  → describe the image(s) using your own vision capability and answer normally.\n"
        f"\nIMPORTANT: When calling the tool, pass the COMPLETE base64 string, not a truncated version."
    )

    # Patch the user message to be text-only
    last_user_content.parts = [genai_types.Part(text=directive)]

    # Store actual b64 values so the model can access them mid-prompt
    # (Gemini will see the directive text and must read from state for full data)
    return None  # Proceed to LLM with patched request


# ──────────────────────────────────────────────────────────────
# Root Agent
# ──────────────────────────────────────────────────────────────
agent = Agent(
    name="CuraNovaMedicalAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.0-flash-exp"),
    before_model_callback=before_model_callback,
    instruction="""You are CuraNova, a specialized medical AI assistant with deep expertise in medical imaging and clinical knowledge.

You have full memory of the conversation. Always use previous context when answering.

## Decision Framework

### Text-only questions (no images)
Answer the user's medical or health questions directly, helpfully, and concisely.
Always note that your responses are for educational purposes and NOT a substitute for professional medical advice.

### Image URL in the message (e.g. https://...)
If the user provides a direct image URL AND is asking for a clinical or medical analysis:
→ Call `analyze_medical_image(image_url=<url>, prompt=<user's question>)`
→ The tool will implicitly trigger the medical analysis stream.
→ Do NOT repeat the tool output or the signal.
→ IMMEDIATELY say: "Your request is being processed by our medical imaging agent. The analysis will appear momentarily ✨"
→ Do NOT add any other text, explanation, or filler.

If the user shares an image URL but is NOT asking for medical analysis (e.g. "what does this look like?"):
→ Answer normally without calling any tool.

### Uploaded image(s) (via camera/file picker)
The `before_model_callback` will inject a directive describing the uploaded images and asking you to decide.

Read the directive carefully:
- If the user is asking for CLINICAL / MEDICAL analysis → call `analyze_medical_image_upload` with the FULL base64 string(s) from state
- If the user is NOT asking for medical analysis (selfie, casual question, "what's in this picture?") → describe the image using your own vision capability and answer directly

After calling either analysis tool:
→ The tool will implicitly trigger the medical analysis stream.
→ Do NOT repeat the tool output or the signal.
→ IMMEDIATELY say: "Your request is being processed by our medical imaging agent. The analysis will appear momentarily ✨"
→ Do NOT add any other text, explanation, or filler.

## Critical rules
- Always maintain conversation context and remember previous turns.
- After a tool call, briefly acknowledge and offer to answer follow-up questions.
- Maximum 2 images per request.
- Never truncate or modify the base64 strings when passing them to tools.
""",
    tools=[analyze_medical_image, analyze_medical_image_upload],
)
