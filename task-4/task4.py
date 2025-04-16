import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

load_dotenv()

class TextSummarizerTool:
    """A wrapper tool that summarizes any given text using the Summarizer class from Task 2."""

    def __init__(self, summary_sentences=3):
        self.summarizer = Summarizer(summary_sentences=summary_sentences)

        self.tool = Tool(
            name="TextSummarizer",
            func=self.summarizer.summarize,
            description="Useful for summarizing any input text. Input should be a long passage to condense."
        )

    def get_tool(self):
        return self.tool


def get_agent_with_tools(tools):
    """Initializes the zero-shot-react-description agent with Azure LLM and provided tools."""
    llm = AzureChatOpenAI(
        temperature=0,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("ENDPOINT_URL"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME_GPT"),
        api_version=os.getenv("API_VERSION"),
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


if __name__ == "__main__":
    summarization_tool = TextSummarizerTool(summary_sentences=3).get_tool()

    agent = get_agent_with_tools([summarization_tool])

    input_text = (
        "Artificial Intelligence (AI) has greatly impacted healthcare by enabling faster diagnoses, "
        "personalized treatments, and predictive analytics. Machine learning algorithms are used to "
        "detect diseases from medical images and predict patient outcomes. AI-powered tools help reduce "
        "workload for doctors, improve accuracy, and optimize hospital operations."
    )
    print("\nAgent Response: AI in Healthcare")
    agent.invoke(f"Summarize the impact of AI on healthcare: {input_text}")

    print("\nAgent Response: Vague Prompt")
    agent.invoke("Summarize something interesting")
