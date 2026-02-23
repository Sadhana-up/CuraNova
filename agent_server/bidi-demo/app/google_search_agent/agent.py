"""CuraNova Medical Image Analysis Agent — ADK Bidi-streaming."""

import os
import json
import websockets
from typing import AsyncGenerator

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Streaming Tool: Connects to the Colab-hosted medical AI model
# via WebSocket and streams back token-by-token analysis.
# ──────────────────────────────────────────────────────────────
async def analyze_medical_image(image_url: str, prompt: str) -> AsyncGenerator[str, None]:
    """Connects to the remote medical AI model and streams back the analysis of a medical image token by token.

    This tool sends the image URL and prompt to a Colab-hosted medical vision model
    over WebSocket. The model analyzes the image (X-ray, CT scan, MRI, medical report, etc.)
    and streams back its findings token by token.

    CRITICAL INSTRUCTION FOR THE MODEL:
    When you call this tool, you must STOP generating text immediately.
    Wait for the stream of tokens to arrive.
    The output of this tool is the ONLY source of truth.
    Do NOT preface this tool call with "Okay, analyzing..." or similar filler.
    Do NOT summarize the output. Relay it exactly as is.

    Args:
        image_url (str): The publicly accessible URL of the medical image to analyze.
        prompt (str): The clinical question or instruction for the analysis (e.g. "Describe this chest X-ray").
    """
    colab_base_url = os.getenv("COLAB_BASE_URL")
    if not colab_base_url:
        yield "Error: COLAB_BASE_URL is not configured. Set it to the ngrok tunnel URL of your Colab server."
        return

    ws_url = colab_base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    endpoint = f"{ws_url}/ws/analyze/image-url"

    payload = {
        "image_url": image_url,
        "prompt": prompt,
        "max_new_tokens": 500,
    }

    try:
        # Debug logging to file
        with open("stream_debug.log", "a", encoding="utf-8") as f:
            f.write(f"\n\n--- New Analysis Request ---\nImage: {image_url}\nPrompt: {prompt}\n")

        async with websockets.connect(endpoint) as ws:
            await ws.send(json.dumps(payload))

            async for message in ws:
                data = json.loads(message)
                if "token" in data:
                    token = data["token"]
                    # Write token to debug file immediately
                    with open("stream_debug.log", "a", encoding="utf-8") as f:
                        f.write(token)
                    yield token
                elif data.get("status") == "done":
                    break
                elif "error" in data:
                    error_msg = f"\n[Server Error]: {data['error']}"
                    with open("stream_debug.log", "a", encoding="utf-8") as f:
                        f.write(error_msg)
                    yield error_msg
                    break
    except Exception as e:
        error_msg = f"\n[Connection Error]: {str(e)}"
        with open("stream_debug.log", "a", encoding="utf-8") as f:
            f.write(error_msg)
        yield error_msg




# ──────────────────────────────────────────────────────────────
# Root Agent — Specialized Medical Image Analyst
# ──────────────────────────────────────────────────────────────
# Available models:
# - Gemini 2.0 Flash (Experimental): "gemini-2.0-flash-exp"
# - Gemini 1.5 Flash: "gemini-1.5-flash"
# - Gemini 1.5 Flash-8B: "gemini-1.5-flash-8b"
agent = Agent(
    name="CuraNovaMedicalAgent",
    # Switching to a stable non-native-audio model to prevent silence-filling hallucinations
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.0-flash-exp"),
    instruction="""You are CuraNova, a friendly medical assistant.

Image analysis requests with URLs are handled automatically by the frontend
streaming proxy — you do NOT need to call any tools for those.

For general medical questions (without an image URL), answer helpfully and
concisely. Always remind users that your information is for educational
purposes and is not a substitute for professional medical advice.

If someone pastes an image URL and you still receive it, call
`analyze_medical_image` with the URL and relay the output exactly as-is
without adding your own interpretation.
""",
    tools=[analyze_medical_image, ],
)
