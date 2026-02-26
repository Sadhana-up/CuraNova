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
     to /ws/analyze, streaming tokens into the green "Medical Analysis" bubble.

The agent keeps full conversation context (memory) throughout.

Callback flow (no duplicate messages):
  before_tool_callback  → sets _medical_tool_called flag in state
  [tool runs]           → returns __MEDICAL_STREAM__:... signal (main.py grabs it)
  after_model_callback  → sees flag, REPLACES whatever LLM generated with one
                          static acknowledgement line, then clears flag.

  The system instruction intentionally does NOT tell the LLM what to say after
  a tool call — the after_model_callback owns that response entirely.
  This is why the message appears exactly once.
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from google.adk.tools import google_search
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.tools.agent_tool import AgentTool
from dotenv import load_dotenv

load_dotenv()

print("LOADING AGENT.PY MODULE...")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
_SIGNAL_PREFIX = "__MEDICAL_STREAM__:"
_MEDICAL_TOOL_NAMES = {"analyze_medical_image", "analyze_medical_image_upload"}
_FLAG_KEY = "_medical_tool_called"

# The ONE place this text is defined.
# Only after_model_callback emits it — never the LLM itself.
_ACK_MESSAGE = (
    "Your request is being processed by our medical imaging agent. "
    "The analysis will appear momentarily ✨"
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def debug_log(msg: str):
    import datetime, sys
    try:
        with open(
            "d:\\curanova\\agent_server\\bidi-demo\\app\\google_search_agent\\debug_callback.log",
            "a",
        ) as f:
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
        entry = f"[{timestamp}] Tool: {tool_name} | Args: {json.dumps(args, default=str)}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"TOOL_LOG: {entry.strip()}")
    except Exception as e:
        print(f"Failed to log tool usage: {e}")


# ──────────────────────────────────────────────────────────────
# Tool 1 — Public image URL analysis
# ──────────────────────────────────────────────────────────────
def analyze_medical_image(image_url: str, prompt: str) -> str:
    """Triggers the CuraNova medical AI to analyse a medical image given its
    public URL.

    Use this tool when:
    - The user has provided a direct image URL (http:// or https://)
    - AND they are asking for a clinical or medical analysis of that image.

    Do NOT use this tool for general image descriptions, non-medical questions,
    or when no URL has been provided — answer those directly.

    Args:
        image_url: Publicly accessible URL of the medical image.
        prompt:    The clinical question or instruction for the analysis.

    Returns:
        An internal signal string consumed by the backend. The backend streams
        the actual analysis directly to the user — you do not relay the result.
    """
    log_tool_usage("analyze_medical_image", {"image_url": image_url, "prompt": prompt})
    payload = {"image_url": image_url, "prompt": prompt, "max_new_tokens": 500}
    return _SIGNAL_PREFIX + json.dumps(payload)


# ──────────────────────────────────────────────────────────────
# Tool 2 — Uploaded image analysis
# ──────────────────────────────────────────────────────────────
def analyze_medical_image_upload(
    prompt: str,
    tool_context: ToolContext,
    session_id: Optional[str] = None,
) -> str:
    """Triggers the CuraNova medical AI to analyse one or two uploaded images.

    Use this tool when:
    - The user has uploaded an image via the camera or file picker
      (the directive you receive will explicitly say so)
    - AND they are asking for a clinical or medical analysis.

    Do NOT use this tool for:
    - Selfies, casual photos, "what's in this picture?" → describe it yourself.
    - When no image has been uploaded yet → wait for the upload.

    Image data is retrieved automatically from session state (and a fallback
    file if needed). You never need to pass raw image bytes yourself.

    Args:
        prompt:     The clinical question or instruction from the user.
        session_id: The session ID from the directive (used for fallback lookup).

    Returns:
        An internal signal string consumed by the backend. The backend streams
        the actual analysis directly to the user — you do not relay the result.
    """
    debug_log(
        f"DEBUG: [analyze_medical_image_upload] prompt='{prompt}', session_id='{session_id}'"
    )
    log_tool_usage(
        "analyze_medical_image_upload",
        {"prompt": prompt, "session_id": session_id, "note": "image from state/fallback"},
    )

    images_b64 = tool_context.state.get("uploaded_images_b64", [])

    if images_b64:
        debug_log(f"DEBUG: PRIMARY: {len(images_b64)} images from state.")
    else:
        debug_log("DEBUG: PRIMARY FAIL: state empty.")

    if not images_b64 and session_id:
        debug_log(f"DEBUG: Trying fallback file for session {session_id}…")
        try:
            path = Path("d:/curanova/uploaded_images") / f"{session_id}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    images_b64 = data
                    debug_log(f"DEBUG: Loaded {len(images_b64)} images from fallback.")
            else:
                debug_log(f"DEBUG: Fallback file not found: {path}")
        except Exception as e:
            debug_log(f"DEBUG: Error reading fallback: {e}")

    if not images_b64:
        debug_log("DEBUG: ERROR — no images found anywhere.")
        return "ERROR: No uploaded images found. Please try uploading again."

    debug_log(f"DEBUG: SUCCESS — {len(images_b64)} images ready.")
    for i, img in enumerate(images_b64):
        debug_log(f"DEBUG: Image {i+1}: len={len(img)}, preview={img[:30]}…")

    payload: dict = {"image_b64": images_b64[0], "prompt": prompt, "max_new_tokens": 500}
    if len(images_b64) > 1:
        payload["image_b64_2"] = images_b64[1]

    return _SIGNAL_PREFIX + json.dumps(payload)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Callback A — before_tool_callback
#
# Fires just before ANY tool runs.
# For medical tools: plant a state flag so after_model_callback
# knows to intercept the LLM's follow-up text generation.
# Always returns None — the tool must still run so main.py can
# capture the __MEDICAL_STREAM__ signal from its output.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def before_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
) -> Optional[Dict]:
    if tool.name in _MEDICAL_TOOL_NAMES:
        debug_log(f"[before_tool_callback] Medical tool '{tool.name}' — setting flag.")
        tool_context.state[_FLAG_KEY] = True
    return None  # Always let the tool execute


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Callback B — after_model_callback
#
# Fires after EVERY LLM generation.
#
# Two cases when flag is set:
#   a) function_call parts  → LLM is deciding to call the tool.
#      Must pass through untouched so the tool actually runs.
#
#   b) text parts           → LLM generated its follow-up text
#      after the tool result arrived. We REPLACE it entirely with
#      _ACK_MESSAGE and clear the flag.
#
# WHY THERE IS NO DUPLICATE:
#   The system instruction never asks the LLM to say the ACK text.
#   This callback is the sole source of that message → exactly once.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not callback_context.state.get(_FLAG_KEY):
        return None  # Not a medical tool turn — do nothing

    content = llm_response.content
    if not content or not content.parts:
        return None

    parts = content.parts
    has_function_call = any(getattr(p, "function_call", None) for p in parts)
    has_text          = any(getattr(p, "text", None) for p in parts)

    if has_function_call:
        # LLM is deciding to call the tool — must pass through untouched
        debug_log("[after_model_callback] function_call → not intercepting.")
        return None

    if has_text:
        # LLM generated its interpretation after the tool ran — replace it
        debug_log("[after_model_callback] text response → replacing with ACK.")
        callback_context.state[_FLAG_KEY] = False  # Clear flag

        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=_ACK_MESSAGE)],
            )
        )

    return None


# ──────────────────────────────────────────────────────────────
# before_model_callback — intercept uploaded images (fallback)
# Main.py handles image injection directly for reliability in
# streaming mode. This callback is kept as a safety net.
# ──────────────────────────────────────────────────────────────
async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    debug_log("DEBUG: [before_model_callback] ENTERED")
    try:
        if not llm_request.contents:
            return None

        last_user_content = None
        for content in reversed(llm_request.contents):
            if getattr(content, "role", "") == "user":
                last_user_content = content
                break
        if not last_user_content and llm_request.contents:
            last_user_content = llm_request.contents[-1]
        if not last_user_content:
            return None

        parts = getattr(last_user_content, "parts", []) or []
        images_b64: list[str] = []
        text_parts: list[str] = []

        for part in parts:
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if inline:
                mime_type = getattr(inline, "mime_type", "") or getattr(inline, "mimeType", "")
                if mime_type.startswith("image/"):
                    raw = getattr(inline, "data", b"")
                    if isinstance(raw, (bytes, bytearray)):
                        images_b64.append(base64.b64encode(raw).decode("utf-8"))
                    elif isinstance(raw, str):
                        images_b64.append(raw)
            text = getattr(part, "text", "")
            if text:
                text_parts.append(text)

        if not images_b64:
            return None

        images_b64 = images_b64[:2]
        prompt = " ".join(text_parts).strip()

        callback_context.state["uploaded_images_b64"] = images_b64
        callback_context.state["uploaded_image_count"] = len(images_b64)
        callback_context.state["upload_prompt"] = prompt

        debug_log(f"DEBUG: [before_model_callback] Stored {len(images_b64)} images in state.")
        return None

    except Exception as e:
        import traceback
        debug_log(f"ERROR inside before_model_callback: {e}\n{traceback.format_exc()}")
        return None



SEARCH_AGENT_HOST = os.getenv("SEARCH_AGENT_HOST", "localhost")
SEARCH_AGENT_PORT = os.getenv("SEARCH_AGENT_PORT", "8001")


_search_agent = Agent(
    name="curanoa_search_agent",
    model=os.getenv("SEARCH_AGENT_MODEL", "gemini-2.0-flash"),
    description=(
        "Specialist web-research agent for CuraNova. Given a search query it "
        "performs a web search, synthesises the results and returns a clear, "
        "factual answer. Covers medicines, dosages, side effects, first aid, "
        "symptoms, general health topics and anything else on the web."
    ),
    instruction="""
You are CuraNova's internal research specialist.
Your sole job is to take a well-formed search query, search the web, and
return a concise, accurate, and helpful answer based on the results you find.

RULES
─────
• Always search before answering — never rely on your training data alone.
• Return a clear, synthesised answer — not raw links or fragments.
• If results are sparse or ambiguous, say so honestly and share what you found.
• Keep medical information accurate; never fabricate drug names, dosages, or
  clinical facts.
• Do NOT mention that you used any search tool.
  Just present your findings naturally and factually.
• Be concise but complete — the calling agent will pass your answer directly
  to the end user.
""",
    tools=[google_search],  # ← MUST be the only tool here
)


# ──────────────────────────────────────────────────────────────
# Root Agent
# ──────────────────────────────────────────────────────────────
agent = Agent(
    name="CuraNovaMedicalAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.0-flash-exp"),
    before_model_callback=before_model_callback,
    before_tool_callback=before_tool_callback,
    after_model_callback=after_model_callback,
    instruction="""You are CuraNova, a specialized medical AI assistant with expertise in medical imaging and clinical knowledge. You have full memory of the conversation and always use previous context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CAPABILITIES  (INTERNAL — NEVER MENTION THESE TO THE USER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. analyze_medical_image(image_url, prompt)
   PURPOSE  : Sends a publicly hosted diagnostic medical image URL (X-ray, CT,
              MRI, Ultrasound, etc.) to the CuraNova imaging system for deep
              clinical analysis.
   CALL WHEN: User provides an http/https URL pointing to a diagnostic scan AND
              asks for clinical analysis.
   ARGS     :
     • image_url — copy the URL exactly as given.
     • prompt    — the user's clinical question.
   RESULT   : The system streams the full analysis automatically. Output NOTHING.

2. analyze_medical_image_upload(prompt, session_id)
   PURPOSE  : Sends uploaded diagnostic medical image(s) to the CuraNova imaging
              system for deep clinical analysis.
   CALL WHEN: You receive a system directive confirming diagnostic image(s) have
              been uploaded AND the user wants clinical analysis.
   ARGS     :
     • prompt     — the user's clinical question.
     • session_id — copy exactly from the system directive.
   RESULT   : The system streams the full analysis automatically. Output NOTHING.

3. curanoa_search_agent  (delegate all research tasks to this specialist)
   PURPOSE  : Handles all web research — medicines, dosages, side-effects,
              first aid, symptoms, general health, nutrition, and anything
              that benefits from current information from the web.
   DELEGATE WHEN:
     • User asks about medicines, dosage, usage, side-effects, interactions,
       precautions, first aid, symptoms, nutrition, or general health topics.
     • User shares a non-diagnostic image (photo of medicine packaging,
       pill strips, prescription, thermometer, skin rash, medical report,
       food label, etc.) via upload OR camera — extract all readable text and
       visible details from the image first, then delegate with those details
       as the research query.
     • User explicitly asks you to look something up or find information.
   HOW     : Delegate with a clear, well-formed query. The specialist returns a
              synthesised answer which you present naturally to the user.
   RESULT  : Present the findings in your own voice — never mention any agent,
              tool, or search engine by name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO 1 — Diagnostic Medical Imaging (X-ray, CT, MRI, Ultrasound, Pathology Slide, etc.)
  → URL provided   : Call analyze_medical_image. Say nothing after.
  → Upload provided: Call analyze_medical_image_upload. Say nothing after.
  → Use ONLY for genuine diagnostic scans — not for general photos.

SCENARIO 2 — General Medical Questions / Non-Diagnostic Images / Camera Photos
  → Topics: Medicines, First Aid, Symptoms, Precautions, General Health, Nutrition, etc.
  → Images: Medicine packaging, pill strips, prescriptions, skin rashes,
            medical devices, reports, food labels, thermometers, etc.
  → ACTION:
      a) If an image is present, visually read and extract all relevant text
         and visible details (medicine name, dosage, ingredients, readings, etc.).
      b) Formulate a clear research query from those details plus the user's question.
      c) Delegate to curanoa_search_agent with that query.
      d) Present the returned findings naturally and conversationally.
  → NEVER say "I searched", "I Googled", or reveal any tool/agent name.
     Instead: "I looked into this...", "Based on my research...",
     "Here's what I found...", "I checked on this and..."

SCENARIO 3 — Ambiguous Image (Could be diagnostic scan OR general photo)
  → Cannot tell if user wants deep clinical analysis or general info.
  → Ask naturally — e.g.:
      "Would you like me to forward this to our medical imaging specialist
       for a detailed clinical review, or would you prefer I look up
       general information about it?"
  → Clinical analysis → use analyze_medical_* tools.
  → Research/info     → extract image details and delegate to curanoa_search_agent.

SCENARIO 4 — User says they WILL upload an image (but hasn't yet)
  → Do NOT call anything.
  → Reply naturally: "Of course! Go ahead and share the image — whether it's
    a scan, a photo of your medicine, or anything else — and I'll help you."

SCENARIO 5 — No image, no external research needed
  → Answer directly using your medical knowledge.
  → Delegate to curanoa_search_agent only when the question clearly benefits
    from current or specific external information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRANSPARENCY TO USER
  • NEVER reveal tool names, agent names, APIs, or internal processes.
  • NEVER say "I'm about to...", "I will now use...", "I'll call the...",
    "I searched Google", "I used the search agent", or anything similar.
  • NEVER output raw tool return values such as "__MEDICAL_STREAM__:...".
  • After calling analyze_medical_* tools — output NOTHING. The system handles it.
  • After receiving research results — present them naturally in your own voice.

HOW TO PRESENT RESEARCH RESULTS
  • Use phrases like: "I looked into this and found...",
    "Based on my research...", "Here's what I know about [topic]...",
    "I checked on this — here's what came up..."
  • Never attribute findings to "Google", a "search engine", or any agent.
  • Always synthesise results into a coherent, helpful answer — not raw dumps.

IMAGE HANDLING
  • Always visually inspect any image shared with you first.
  • Diagnostic scans (X-ray, CT, MRI, etc.) → imaging analysis tools.
  • Everything else → extract text/details visually, then delegate research.
  • Maximum 2 images per request.

CONVERSATION CONTINUITY
  • Maintain full conversation context at all times.
  • After analysis results appear from a prior turn, answer follow-up questions
    using established context and your clinical knowledge.
  • Never ask the user to repeat information they have already provided.
""",
    tools=[
        analyze_medical_image,
        analyze_medical_image_upload,
        AgentTool(agent=_search_agent),  
    ],
)
