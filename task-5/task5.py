import os
import sys
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-3')))

from task2 import Summarizer
from task3 import Retriever, RetrieverConfig

retriever_config = RetrieverConfig(file_path="task-3/ai_intro.txt")
retriever = Retriever(retriever_config)
summarizer = Summarizer(summary_sentences=3)

def retrieve_text(query: str) -> str:
    return retriever.get_retrieved_text(query)

def summarize_text(text: str) -> str:
    return summarizer.summarize(text)

def count_words(text: str) -> str:
    count = len(text.split())
    return f"The summary contains {count} words."

def create_agent():
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("ENDPOINT_URL"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME_GPT"),
        api_version=os.getenv("API_VERSION"),
        temperature=0
    )

    tools = [
        Tool(
            name="TextRetriever",
            func=retrieve_text,
            description="Use this to retrieve relevant text from the document based on a query.",
        ),
        Tool(
            name="TextSummarizer",
            func=summarize_text,
            description="Use this to summarize a given text into 3 sentences.",
        ),
        Tool(
            name="WordCounter",
            func=count_words,
            description="Use this to count the number of words in the summary.",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    return agent

if __name__ == "__main__":
    agent = create_agent()

    print("\nTEST 1: Find and summarize AI breakthroughs:\n")
    result1 = agent.invoke("Only find and then finally summarize text about AI breakthroughs from the document.")
    print(result1)

    print("\nTEST 2: Add word count to the summary:\n")
    result2 = agent.invoke("Find and summarize text about AI breakthroughs from the document. Add the word count at the end of the summary.")
    print(result2)
