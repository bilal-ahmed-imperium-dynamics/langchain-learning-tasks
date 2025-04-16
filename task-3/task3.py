import os
import sys
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

load_dotenv()


class RetrieverConfig:
    def __init__(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 20):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT_URL"),
            azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
            api_version=os.getenv("API_VERSION"),
        )
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        loader = TextLoader(self.config.file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore.as_retriever()

    def retrieve(self, query: str):
        return self.retriever.invoke(query)

    def get_retrieved_text(self, query: str):
        docs = self.retrieve(query)
        return "\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    config = RetrieverConfig(file_path="task-3/ai_intro.txt")
    retriever = Retriever(config)

    query = "AI milestones"
    retrieved_text = retriever.get_retrieved_text(query)

    print("\nRetrieved Relevant Text:\n")
    print(retrieved_text)

    print(f"\nRetrieved Relevant Text Length: {len(retrieved_text.split())}")

    summarizer = Summarizer(summary_sentences=3)
    summary = summarizer.summarize(retrieved_text)

    print("\nSummary:\n")
    print(summary)
