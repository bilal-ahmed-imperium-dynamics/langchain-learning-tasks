import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# Reuse Summarizer from Task 2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

load_dotenv()


class MultiRetriever:
    def __init__(self, file_path: str, chunk_size=200, chunk_overlap=20):
        loader = TextLoader(file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT_URL"),
            azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
            api_version=os.getenv("API_VERSION"),
        )

        vectorstore = Chroma.from_documents(chunks, embeddings)

        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT_URL"),
            azure_deployment=os.getenv("DEPLOYMENT_NAME_GPT"),
            api_version=os.getenv("API_VERSION"),
            temperature=0.7,
        )

        self.multi_retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=self.llm)
        self.single_retriever = vectorstore.as_retriever()

    def retrieve_multi_query_text(self, query: str):
        docs = self.multi_retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)

    def retrieve_single_query_text(self, query: str):
        docs = self.single_retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)
    
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    file_path = r"C:\Users\dell\langchain-learning-tasks\task-3\ai_intro.txt"  # Ensure this path is correct

    retriever = MultiRetriever(file_path)
    query = "AI advancements"

    print("\n[Generated Queries]\n")
    multi_text = retriever.retrieve_multi_query_text(query)
    print("\n[Multi-Query Retrieved Text]\n")
    print(multi_text)
    print(f"\nMulti-Query Word Count: {len(multi_text.split())}")

    single_text = retriever.retrieve_single_query_text(query)
    print("\n[Single-Query Retrieved Text]\n")
    print(single_text)
    print(f"\nSingle-Query Word Count: {len(single_text.split())}")

    summarizer = Summarizer(summary_sentences=3)
    print("\n[Multi-Query Summary]\n")
    print(summarizer.summarize(multi_text))

    print("\n[Single-Query Summary]\n")
    print(summarizer.summarize(single_text))
