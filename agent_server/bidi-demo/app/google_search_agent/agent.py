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
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool          # ← NEW
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse   # ← NEW
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

print("LOADING AGENT.PY MODULE...")

# ──────────────────────────────────────────────────────────────
# SIGNAL PREFIX — main.py watches for this prefix in tool output
# to know it must forward a medical_stream_trigger event.
# ──────────────────────────────────────────────────────────────
_SIGNAL_PREFIX = "__MEDICAL_STREAM__:"

# Names of tools that trigger background medical analysis.
# After these run, the LLM's own interpretation is redundant —
# we replace it with a static acknowledgement.
_MEDICAL_TOOL_NAMES = {"analyze_medical_image", "analyze_medical_image_upload"}

# State key used to communicate between the two new callbacks.
_FLAG_KEY = "_medical_tool_called"

def debug_log(msg: str):
    import datetime
    import sys
    try:
        with open("d:\\curanova\\agent_server\\bidi-demo\\app\\google_search_agent\\debug_callback.log", "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg}\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")
    print(msg)
    sys.stdout.flush()

def log_tool_usage(tool_name: str, args: dict):
    import datetime
    try:
        log_path = "d:\\curanova\\agent_server\\bidi-demo\\app\\google_search_agent\\tool_usage.log"
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] Tool: {tool_name} | Args: {json.dumps(args, default=str)}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"TOOL_LOG: {log_entry.strip()}")
    except Exception as e:
        print(f"Failed to log tool usage: {e}")


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
    log_tool_usage("analyze_medical_image", {"image_url": image_url, "prompt": prompt})
    payload = {"image_url": image_url, "prompt": prompt, "max_new_tokens": 500}
    return _SIGNAL_PREFIX + json.dumps(payload)


# ──────────────────────────────────────────────────────────────
# Tool 2: Trigger analysis for uploaded image(s)
# ──────────────────────────────────────────────────────────────
def analyze_medical_image_upload(
    prompt: str,
    tool_context: ToolContext,
    session_id: Optional[str] = None,
) -> str:
    """Triggers the CuraNova medical AI to analyse one or two uploaded images.  

    Call this tool ONLY when:
      - The user has uploaded an image (and the directive says so), AND
      - The user is asking for a clinical / medical analysis of that image.     

    The actual image data is retrieved from session state automatically. 
    If for some reason session state fails, it will attempt to read from a fallback file using session_id.       

    Args:
        prompt: The clinical question or instruction from the user.
        session_id: The session ID provided by the user context (if available).

    Returns:
        A signal string for the backend streaming mechanism.
    """
    debug_log(f"DEBUG: [analyze_medical_image_upload] Called with prompt='{prompt}', session_id='{session_id}'")
    log_tool_usage("analyze_medical_image_upload", {
        "prompt": prompt,
        "session_id": session_id,
        "note": "Image data retrieved from session state or fallback file"
    })

    images_b64 = tool_context.state.get("uploaded_images_b64", [])

    if images_b64:
        debug_log(f"DEBUG: [analyze_medical_image_upload] PRIMARY SUCCESS: Found {len(images_b64)} images in session state memory.")
    else:
        debug_log("DEBUG: [analyze_medical_image_upload] PRIMARY FAIL: Session state empty.")

    if not images_b64 and session_id:
        debug_log(f"DEBUG: [analyze_medical_image_upload] Checking fallback file for session {session_id}...")
        try:
            fallback_path = Path("d:/curanova/uploaded_images") / f"{session_id}.json"
            if fallback_path.exists():
                with open(fallback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        images_b64 = data
                        debug_log(f"DEBUG: [analyze_medical_image_upload] Loaded {len(images_b64)} images from fallback file.")
            else:
                debug_log(f"DEBUG: [analyze_medical_image_upload] Fallback file not found: {fallback_path}")
        except Exception as e:
            debug_log(f"DEBUG: [analyze_medical_image_upload] Error reading fallback file: {e}")

    if not images_b64:
        debug_log("DEBUG: [analyze_medical_image_upload] ERROR: No uploaded images found.")
        return "ERROR: No uploaded images found. Please try uploading again."

    debug_log(f"DEBUG: [analyze_medical_image_upload] SUCCESS! Found {len(images_b64)} images.")
    for idx, img in enumerate(images_b64):
        img_len = len(img) if img else 0
        img_preview = img[:30] + "..." if img_len > 30 else "check data"
        debug_log(f"DEBUG: [analyze_medical_image_upload] Image {idx+1}: Length={img_len} chars | Preview={img_preview}")

    payload: dict = {"image_b64": images_b64[0], "prompt": prompt, "max_new_tokens": 500}
    if len(images_b64) > 1:
        payload["image_b64_2"] = images_b64[1]

    return _SIGNAL_PREFIX + json.dumps(payload)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW CALLBACK 1: before_tool_callback
#
# Fires just before any tool runs.
# When a medical analysis tool is about to execute, we plant a flag
# in session state so the after_model_callback knows to intercept
# the LLM's follow-up interpretation.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def before_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
) -> Optional[Dict]:
    """
    Intercept point: fires before every tool call.

    For medical analysis tools, we set a state flag so the
    after_model_callback can replace the LLM's redundant text
    interpretation with a single static acknowledgement line.

    Returns None always — we never want to skip the tool itself here,
    because main.py needs the signal string the tool returns.
    """
    tool_name = tool.name
    if tool_name in _MEDICAL_TOOL_NAMES:
        debug_log(f"[before_tool_callback] Medical tool '{tool_name}' is about to run — setting flag.")
        tool_context.state[_FLAG_KEY] = True
    return None  # Always let the tool execute normally


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW CALLBACK 2: after_model_callback_handler
#
# Fires after every LLM generation.
# When the flag is present AND the LLM just produced a plain text
# response (i.e. its interpretation of the tool result), we replace
# that text with a single static acknowledgement and clear the flag.
#
# We leave function_call responses untouched — those are the LLM
# deciding TO call a tool, which is fine and needed.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def after_model_callback_handler(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    Intercept point: fires after every LLM generation.

    If a medical tool was just called (flag is set) and the LLM has now
    produced a plain text response (its interpretation of the tool result),
    replace that response with a concise static acknowledgement.

    Function-call responses are left untouched so the tool still executes.
    """
    if not callback_context.state.get(_FLAG_KEY):
        return None  # Not a medical tool turn — do nothing

    content = llm_response.content
    if not content or not content.parts:
        return None

    parts = content.parts

    has_function_call = any(getattr(p, "function_call", None) for p in parts)
    has_text          = any(getattr(p, "text", None) for p in parts)

    if has_function_call:
        # This is the LLM deciding to call the tool — let it through untouched.
        debug_log("[after_model_callback] Function call response — not intercepting.")
        return None

    if has_text:
        # This is the LLM's interpretation AFTER the tool ran — replace it.
        debug_log("[after_model_callback] Replacing LLM interpretation with static acknowledgement.")
        callback_context.state[_FLAG_KEY] = False  # Clear flag

        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(
                    text="Your request is being processed by our medical imaging agent. "
                         "The analysis will appear momentarily ✨"
                )],
            )
        )

    return None


# ──────────────────────────────────────────────────────────────
# Before-model callback — intercept uploaded images
# (unchanged from original)
# ──────────────────────────────────────────────────────────────
async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """Detect inline image uploads and expose base64 data to the agent via
    a text directive — the agent then decides whether to call a tool.
    
    NOTE: As of Feb 2026, image processing has moved to main.py for reliability.
    This callback is kept as a fallback or for debugging purposes.
    """
    debug_log("DEBUG: [before_model_callback] ENTERED")
    try:
        if not llm_request.contents:
            debug_log("DEBUG: [before_model_callback] No contents in llm_request")
            return None

        debug_log(f"DEBUG: [before_model_callback] Contents count: {len(llm_request.contents)}")

        last_user_content = None
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                role = getattr(content, "role", "unknown")
                debug_log(f"DEBUG: [before_model_callback] Checking content with role: {role}")
                if role == "user":
                    last_user_content = content
                    break

        if not last_user_content:
            debug_log("DEBUG: [before_model_callback] No user content found in llm_request.")
            if llm_request.contents:
                debug_log(f"DEBUG: [before_model_callback] Contents available: {len(llm_request.contents)}")
                last_user_content = llm_request.contents[-1]
                debug_log(f"DEBUG: [before_model_callback] Inspecting last content (role={getattr(last_user_content, 'role', 'unknown')})")
            else:
                return None

        parts = getattr(last_user_content, "parts", []) or []
        images_b64: list[str] = []
        text_parts: list[str] = []

        debug_log(f"DEBUG: [before_model_callback] Processing {len(parts)} parts")

        for i, part in enumerate(parts):
            debug_log(f"DEBUG: [before_model_callback] Inspecting part {i}")
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)

            if not inline:
                debug_log(f"DEBUG: [before_model_callback] Part {i} has no direct inline_data attribute")
                if hasattr(part, "to_dict"):
                    d = part.to_dict()
                    debug_log(f"DEBUG: [before_model_callback] Part {i} as dict keys: {d.keys()}")
                    if "inline_data" in d:
                        debug_log(f"DEBUG: [before_model_callback] Found inline_data in dict, likely a serialization issue if getattr failed")

            if inline:
                mime_type = getattr(inline, "mime_type", "") or getattr(inline, "mimeType", "")
                if not mime_type and hasattr(inline, "mime_type"):
                    mime_type = inline.mime_type

                debug_log(f"DEBUG: [before_model_callback] Found inline data with mime_type: {mime_type}")

                if mime_type.startswith("image/"):
                    raw = getattr(inline, "data", b"")
                    if isinstance(raw, (bytes, bytearray)):
                        b64_str = base64.b64encode(raw).decode("utf-8")
                        images_b64.append(b64_str)
                        debug_log(f"DEBUG: [before_model_callback] Encoded image (len={len(b64_str)})")
                    elif isinstance(raw, str):
                        images_b64.append(raw)
                        debug_log(f"DEBUG: [before_model_callback] Found string image data (len={len(raw)})")

            text_content = getattr(part, "text", "")
            if text_content:
                text_parts.append(text_content)

        if not images_b64:
            debug_log(f"DEBUG: [before_model_callback] No images found in parts (Total parts: {len(parts)})")
            return None

        debug_log(f"DEBUG: [before_model_callback] Found {len(images_b64)} images. Storing in state...")
        images_b64 = images_b64[:2]
        prompt = " ".join(text_parts).strip()

        callback_context.state["uploaded_images_b64"] = images_b64
        callback_context.state["uploaded_image_count"] = len(images_b64)
        callback_context.state["upload_prompt"] = prompt

        debug_log(f"DEBUG: [before_model_callback] State updated with {len(images_b64)} images.")

        if "Decide: does this image require a CLINICAL / MEDICAL analysis?" in prompt:
            debug_log("DEBUG: [before_model_callback] Detected existing directive. Letting it pass through with image.")
            return None

        return None

    except Exception as e:
        import traceback
        debug_log(f"ERROR inside before_model_callback: {e}")
        debug_log(traceback.format_exc())
        return None


# ──────────────────────────────────────────────────────────────
# Root Agent
# ──────────────────────────────────────────────────────────────
agent = Agent(
    name="CuraNovaMedicalAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.0-flash-exp"),
    # ── Callbacks ───────────────────────────────────────────
    before_model_callback=before_model_callback,
    before_tool_callback=before_tool_callback,           # ← NEW
    after_model_callback=after_model_callback_handler,   # ← NEW
    # ────────────────────────────────────────────────────────
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