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
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

print("LOADING AGENT.PY MODULE...")

# ──────────────────────────────────────────────────────────────
# SIGNAL PREFIX — main.py watches for this prefix in tool output
# to know it must forward a medical_stream_trigger event.
# ──────────────────────────────────────────────────────────────
_SIGNAL_PREFIX = "__MEDICAL_STREAM__:"

def debug_log(msg: str):
    import datetime
    import sys
    try:
        # Try to write to a file in a known absolute path to avoid CWD issues
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
            
        # Also print to stdout for immediate feedback
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
    # Log usage
    log_tool_usage("analyze_medical_image", {
        "image_url": image_url,
        "prompt": prompt
    })

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
    
    # Log usage (excluding massive image data)
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
        debug_log("DEBUG: [analyze_medical_image_upload] ERROR: No uploaded images found in session state or fallback file.")
        return "ERROR: No uploaded images found. Please try uploading again."
    
    # Detailed logging for verification
    debug_log(f"DEBUG: [analyze_medical_image_upload] SUCCESS! Found {len(images_b64)} images.")
    for idx, img in enumerate(images_b64):
        img_len = len(img) if img else 0
        img_preview = img[:30] + "..." if img_len > 30 else "check data"
        debug_log(f"DEBUG: [analyze_medical_image_upload] Image {idx+1}: Length={img_len} chars | Preview={img_preview}")

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
        
        # Find the last user turn
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
            # Fallback: check if there are any contents at all, maybe the role is missing or different?
            if llm_request.contents:
                 debug_log(f"DEBUG: [before_model_callback] Contents available: {len(llm_request.contents)}")
                 # potentially use the last content if it's the only one and we just sent it?
                 # Check if the last content has no role (sometimes it happens?) or 'model' role but it's actually ours?
                 # Just picking the last content to inspect parts:
                 last_user_content = llm_request.contents[-1]
                 debug_log(f"DEBUG: [before_model_callback] Inspecting last content (role={getattr(last_user_content, 'role', 'unknown')})")
            else:
                 return None

        parts = getattr(last_user_content, "parts", []) or []
        images_b64: list[str] = []
        text_parts: list[str] = []

        debug_log(f"DEBUG: [before_model_callback] Processing {len(parts)} parts")
        # print(f"DEBUG: [before_model_callback] Current state keys: {list(callback_context.state.keys())}")  # 'State' object has no attribute 'keys'

        for i, part in enumerate(parts):
            debug_log(f"DEBUG: [before_model_callback] Inspecting part {i}")
            # Check both snake_case and camelCase attributes
            # Also check deeper structure if part is a dict (which it shouldn't be, but let's be safe)
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            
            if not inline:
                # Check if part itself is a dict-like object or has other attributes
                # Sometimes wrapping might be weird
                debug_log(f"DEBUG: [before_model_callback] Part {i} has no direct inline_data attribute")
                if hasattr(part, "to_dict"):
                    d = part.to_dict()
                    debug_log(f"DEBUG: [before_model_callback] Part {i} as dict keys: {d.keys()}")
                    if "inline_data" in d:
                         # Try to construct it or use it? But we need bytes
                         debug_log(f"DEBUG: [before_model_callback] Found inline_data in dict, likely a serialization issue if getattr failed")

            if inline:
                mime_type = getattr(inline, "mime_type", "") or getattr(inline, "mimeType", "")
                if not mime_type and hasattr(inline, "mime_type"): # Sometimes empty string
                     mime_type = inline.mime_type
                
                debug_log(f"DEBUG: [before_model_callback] Found inline data with mime_type: {mime_type}")
                
                if mime_type.startswith("image/"):
                    raw = getattr(inline, "data", b"")
                    # Ensure data is bytes-like
                    if isinstance(raw, (bytes, bytearray)):
                        b64_str = base64.b64encode(raw).decode("utf-8")
                        images_b64.append(b64_str)
                        debug_log(f"DEBUG: [before_model_callback] Encoded image (len={len(b64_str)})")
                    elif isinstance(raw, str):
                        # If data is already string (base64 possibly?), try to decode then encode or treat as base64
                        # Assuming it's base64 string if it's string type here
                        images_b64.append(raw)
                        debug_log(f"DEBUG: [before_model_callback] Found string image data (len={len(raw)})")
            
            # Check for text content
            text_content = getattr(part, "text", "")
            if text_content:
                text_parts.append(text_content)

        if not images_b64:
            debug_log(f"DEBUG: [before_model_callback] No images found in parts (Total parts: {len(parts)})")
            return None  # No images — let the model proceed normally
        
        debug_log(f"DEBUG: [before_model_callback] Found {len(images_b64)} images. Storing in state...")

        # Enforce max-2 limit
        images_b64 = images_b64[:2]

        prompt = " ".join(text_parts).strip()

        # Check for directive prefix to avoid double-processing (since main.py injects directive)
        # If the 'prompt' starts with "The user sent...", it's likely our own directive coming back around?
        # No, before_model_callback processes what is ABOUT to go to the model.
        # If main.py sends types.Content(parts=[text, image]), this callback will see it.
        
        # KEY CHANGE: Do NOT strip the image from the message if we want the model to see it!
        # The previous logic was explicitly replacing the parts with JUST text directive.
        # We need to allow the image to pass through so the model can see it for "NO" cases.
        
        # Store base64 data in session state for the agent to reference (still needed for tool call)
        callback_context.state["uploaded_images_b64"] = images_b64
        callback_context.state["uploaded_image_count"] = len(images_b64)
        callback_context.state["upload_prompt"] = prompt
        
        debug_log(f"DEBUG: [before_model_callback] State updated with {len(images_b64)} images.")

        # If this is coming from main.py's manual injection, the text part is already the directive.
        # We should detect if the text part ALREADY contains our directive signature.
        if "Decide: does this image require a CLINICAL / MEDICAL analysis?" in prompt:
             debug_log("DEBUG: [before_model_callback] Detected existing directive. Letting it pass through with image.")
             return None # Let the model see the image and the directive!

        # ... (Old logic for patching user message if we were intercepting raw uploads, but main.py handles that now)
        # In fact, since main.py is now doing the heavy lifting, this callback might be redundant or conflicting
        # if it tries to rewrite the message again.
        
        return None
        
        debug_log(f"DEBUG: [before_model_callback] State updated with {len(images_b64)} images.")


        # Build indexes into state for the tool-call directive
        image_count = len(images_b64)
        
        # We REMOVE the patching logic here because we want the image to pass through to the model
        # so it can describe it visually if no tool is called.
        # The directive from main.py is already structured as Text + Image.
        
        if "Decide: does this image require a CLINICAL / MEDICAL analysis?" in prompt:
             debug_log("DEBUG: [before_model_callback] Skipped patching (main.py directive detected)")
             return None 
        
        # If this is a raw client upload not processed by main.py (unlikely now), we might patch it.
        # But generally, we should just let the image through.
        
        debug_log(f"DEBUG: [before_model_callback] Pass-through: Allowing model to see {image_count} images directly.")

        # Store actual b64 values so the model can access them mid-prompt
        # (Gemini will see the directive text and must read from state for full data)
        return None  # Proceed to LLM with patched request
        
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
