"""Google Search Agent definition for ADK Bidi-streaming demo."""

import os
import math

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import agent_tool,google_search
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = LiteLlm(model="groq/llama-3.3-70b-versatile", api_key=groq_api_key)

# Define a tool for the joke agent
def get_default_joke() -> str:
    """Returns a default joke."""
    return "Why do programmers prefer dark mode? Because light attracts bugs!"

# Define a tool for the factorial agent
def calculate_factorial(n: int) -> int:
    """Calculates the factorial of a given number.
    
    Args:
        n (int): The number to calculate the factorial for.
    """
    return math.factorial(n)

# Define the sub-agents
joke_agent = Agent(
    name="JokeAgent",
    model=groq_model,
    description="An agent that tells jokes.",
    instruction="You are a comedian. Use the get_default_joke tool to tell jokes, or come up with your own.",
    tools=[get_default_joke]
)

math_agent = Agent(
    name="MathAgent",
    model=groq_model,
    description="An agent that performs mathematical calculations, specifically factorials.",
    instruction="You are a mathematician. Use the calculate_factorial tool to compute factorials.",
    tools=[calculate_factorial]
)

search_agent = Agent(
    name="SearchAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    description="An agent that searches the web for information.",
    instruction="You are a researcher. Use the google_search tool to find information on the web.",
    tools=[google_search, calculate_factorial]
)

joke_agent_tool = agent_tool.AgentTool(agent=joke_agent)
math_agent_tool = agent_tool.AgentTool(agent=math_agent)

# Define the root orchestrator agent
agent = Agent(
    name="OrchestratorAgent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    instruction="You are a helpful orchestrator assistant. Use the JokeAgent tool for jokes, use the MathAgent tool for factorials, and transfer to SearchAgent for web searches.",
    tools=[joke_agent_tool, math_agent_tool],
    sub_agents=[search_agent]  
)
