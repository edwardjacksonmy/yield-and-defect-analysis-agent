"""
agent/core.py
LangChain ReAct agent setup — wires all 8 tools into an AgentExecutor.
"""

import os
from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from agent.tools.data_ingestion import data_ingestion_tool
from agent.tools.yield_calculator import yield_calculator_tool
from agent.tools.defect_analyzer import defect_analyzer_tool
from agent.tools.spatial_clustering import spatial_clustering_tool
from agent.tools.historical_query import historical_query_tool
from agent.tools.wafer_visualizer import wafer_visualizer_tool
from agent.tools.report_generator import report_generator_tool
from agent.tools.root_cause import root_cause_tool

load_dotenv()

ALL_TOOLS = [
    data_ingestion_tool,
    yield_calculator_tool,
    defect_analyzer_tool,
    spatial_clustering_tool,
    historical_query_tool,
    wafer_visualizer_tool,
    report_generator_tool,
    root_cause_tool,
]

REACT_PROMPT_TEMPLATE = """You are an expert IC manufacturing data scientist specialising in wafer yield analysis \
and defect pattern recognition using the WM-811K defect taxonomy.

You have access to the following tools:
{tools}

TOOL USAGE GUIDELINES:
- When the user message contains a file path: call data_ingestion_tool FIRST with that exact path.
- For ALL other analysis queries (yield, defects, clustering, root cause), call the relevant tool(s) DIRECTLY — they read from the already-loaded session data. Do NOT ask for a file path; just invoke the tool. If the tool returns "No batch loaded", then inform the user a file must be uploaded.
- For yield queries: call yield_calculator_tool with input "current".
- For defect analysis: call defect_analyzer_tool AND spatial_clustering_tool together.
- For comparison queries: call historical_query_tool.
- For any analysis involving visualisation: call wafer_visualizer_tool.
- For root cause questions: call root_cause_tool with 'auto' or a specific defect pattern.
- For comprehensive reports: call report_generator_tool last (after other tools have run).
- For comprehensive/complete analysis requests (covering yield + defects + clustering + root cause): ALWAYS call yield_calculator_tool, defect_analyzer_tool, spatial_clustering_tool, AND root_cause_tool in sequence — do NOT give a Final Answer after only one or two tools.

You MUST follow this EXACT format:

Question: the user question you must answer
Thought: reason about what information you need and which tool to call next
Action: the exact tool name (one of [{tool_names}])
Action Input: the input string for the tool
Observation: the result of the tool call
... (repeat Thought/Action/Action Input/Observation as many times as needed)
Thought: I now have all the information needed to give a complete answer
Final Answer: a comprehensive, well-structured answer using IC manufacturing terminology

Important:
- Be precise and professional — use IC manufacturing terminology (yield, die, lot, wafer, defect pattern).
- When yield is below 80%, explicitly flag it as a critical situation.
- Always summarise findings before giving recommendations.
- Do not guess — use tool outputs as the sole basis for your answers.

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


def _get_llm():
    """Initialise LLM — prefers Anthropic Claude, falls back to OpenAI GPT-4o."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise EnvironmentError(
            "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
        )


def create_agent_executor(memory: ConversationBufferMemory) -> AgentExecutor:
    """
    Create and return a LangChain ReAct AgentExecutor with all 8 tools and conversation memory.

    Args:
        memory: ConversationBufferMemory instance (shared with Streamlit session)

    Returns:
        Configured AgentExecutor ready for invoke()
    """
    llm = _get_llm()

    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "agent_scratchpad", "input", "chat_history"],
        template=REACT_PROMPT_TEMPLATE,
    )

    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25,
        max_execution_time=300,
        return_intermediate_steps=True,
    )

    return executor