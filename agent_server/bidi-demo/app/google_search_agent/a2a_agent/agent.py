"""
CuraNova - Google Search Sub-Agent
===================================
This agent is exposed as a standalone A2A microservice on port 8001.
It owns the `google_search` tool exclusively (ADK requires search tools
to live alone in their own agent).  The root CuraNovaMedicalAgent calls
this agent remotely via the A2A protocol whenever web research is needed.

Start this server with:
    uvicorn search_agent.agent:a2a_app --host localhost --port 8001 --reload
"""

import os

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# ──────────────────────────────────────────────────────────────────────────────
# Search Agent definition
# ──────────────────────────────────────────────────────────────────────────────
print("*"*5,os.getenv("SEARCH_AGENT_MODEL", "gemini-3-flash-preview"),"*"*5)
search_agent = Agent(
    name="curanoa_search_agent",
    model=os.getenv("SEARCH_AGENT_MODEL", "gemini-3-flash-preview"),
    description=(
        "A specialist web-research agent for CuraNova. "
        "Given a search query it performs a Google Search, synthesises the "
        "results and returns a clear, factual answer. "
        "Covers medicines, dosages, side effects, first aid, symptoms, "
        "general health topics and any other information that can be found "
        "on the web."
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
• Do NOT mention that you used Google Search or any search tool.
  Just present your findings naturally and factually.
• Be concise but complete — the calling agent will pass your answer directly
  to the end user.
""",
    tools=[google_search],  # google_search MUST be the only tool in this agent
)

# ──────────────────────────────────────────────────────────────────────────────
# Expose as an A2A server
# to_a2a() auto-generates the agent card at /.well-known/agent.json
# ──────────────────────────────────────────────────────────────────────────────

a2a_app = to_a2a(search_agent, port=8001)