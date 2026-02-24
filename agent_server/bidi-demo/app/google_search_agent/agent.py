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
from google.adk.tools.tool_context import ToolContext
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
    prompt: str,
    tool_context: ToolContext,
) -> str:
    """Triggers the CuraNova medical AI to analyse one or two uploaded images.

    Call this tool ONLY when:
      - The user has uploaded an image (and the directive says so), AND
      - The user is asking for a clinical / medical analysis of that image.

    The actual image data is retrieved from session state automatically.

    Args:
        prompt: The clinical question or instruction from the user.

    Returns:
        A signal string for the backend streaming mechanism.
    """
    images_b64 = tool_context.state.get("uploaded_images_b64", [])
    if not images_b64:
        return "ERROR: No uploaded images found in session state."

    payload: dict = {
        "image_b64": images_b64[0],
        "prompt": prompt,
        "max_new_tokens": 500,
    }
    if len(images_b64) > 1:
        payload["image_b64_2"] = images_b64[1]

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
    image_count = len(images_b64)
    
    directive = (
        f"The user sent {image_count} image(s) with this message: \"{prompt}\"\n\n"
        f"Decide: does this image require a CLINICAL / MEDICAL analysis?\n"
        f"- If YES → call analyze_medical_image_upload(prompt=\"{prompt}\").\n"
        f"  (The images are automatically retrieved from session state, no need to pass them).\n"
        f"- If NO  → describe the image(s) using your own vision capability and answer normally.\n"
        f"\nIMPORTANT: Do not try to access state['uploaded_images_b64'] yourself."
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
1. FIRST, call the tool: `analyze_medical_image(image_url=<url>, prompt=<user's question>)`
2. THEN, after the tool has been called, output the text: "Your request is being processed by our medical imaging agent. The analysis will appear momentarily ✨"

Do not describe your plan ("I will call the tool..."). just CALL IT.

### Uploaded image(s) (via camera/file picker)
The `before_model_callback` will inject a directive describing the uploaded images and asking you to decide.

Read the directive carefully:
- If the user is asking for CLINICAL / MEDICAL analysis:
  1. FIRST, call the tool: `analyze_medical_image_upload(prompt=<user's question>)`
  2. THEN, after the tool has been called, output the text: "Your request is being processed by our medical imaging agent. The analysis will appear momentarily ✨"

- If the user is NOT asking for medical analysis (selfie, casual question, "what's in this picture?") → describe the image using your own vision capability and answer directly.

After calling either analysis tool:
- The tool will return a signal string. DO NOT output this signal string to the user.
- JUST say: "Your request is being processed by our medical imaging agent. The analysis will appear momentarily ✨"

## Critical rules
- Always maintain conversation context and remember previous turns.
- After a tool call, briefly acknowledge and offer to answer follow-up questions.
- Maximum 2 images per request.
- Never truncate or modify the base64 strings when passing them to tools.
""",
    tools=[analyze_medical_image, analyze_medical_image_upload],
)
