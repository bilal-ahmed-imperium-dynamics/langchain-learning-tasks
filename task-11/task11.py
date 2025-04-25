import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import datetime

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-3')))

from task2 import Summarizer
from task3 import Retriever, RetrieverConfig

retriever_config = RetrieverConfig(file_path="task-3/ai_intro.txt")
retriever = Retriever(retriever_config)
summarizer = Summarizer(summary_sentences=3)

@tool
def retrieve_text(query: str) -> str:
    """Retrieves the relevant text."""
    print("\nCalling [retrieve_text] tool:\n\n")
    return retriever.get_retrieved_text(query)

@tool
def summarize_text(text: str) -> str:
    """Summarizes given text."""
    print("\nCalling [summarize_text] tool:\n\n")
    return summarizer.summarize(text)

@tool
def extract_datetime():    
    """Returns current date & time in DD/MM/YYYY HH/MM/SS format."""
    print("\nCalling [extract_datetime] tool:\n\n")
    return datetime.datetime.now()

@tool
def mock_websearch():
    """Performs mock search for recent AI updates."""
    print("\nCalling [mock_websearch] tool:\n\n")
    return "(Mock): In 2025, AI is becoming more integrated into daily life, with tech giants like Microsoft " \
    "enhancing productivity tools through AI agents, and startups like Kira Learning revolutionizing education with " \
    "AI-powered teaching assistants. However, the AI boom faces challenges due to global economic uncertainties and " \
    "escalating U.S.-China trade tensions, leading to disruptions in supply chains and cautious investment " \
    "approaches from major tech companies"

tools = [retrieve_text, summarize_text, extract_datetime, mock_websearch]

agent = create_react_agent(
    summarizer.llm,
    tools
)

response =agent.invoke({'messages':[HumanMessage(content='Summarize this 100-word text about AI and tell me today’s date')]})
print(response['messages'][-1].content)

response = agent.invoke({'messages':[HumanMessage(content='Summarize AI trends and search for recent updates.')]})
print(response['messages'][-1].content)