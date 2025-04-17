import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

# === Config ===
PDF_PATH = r"C:\Users\dell\langchain-learning-tasks\task-7\20242003_IFPMA_NfG_AI-ethics-principles.pdf"
WEB_URL = "https://www.coursera.org/articles/ai-trends"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
SEPARATOR = '\n\n'
QUERY = "AI challenges"

# === Custom Text Splitter Class ===
class Splitter:
    def __init__(self, separator='\n', chunk_size=150, chunk_overlap=30):
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, docs):
        return self.splitter.split_documents(docs)

# === Document Loaders ===
pdf_loader = PyPDFLoader(PDF_PATH)
pdf_docs = pdf_loader.load()
print("PDF loaded successfully.")

web_loader = WebBaseLoader(WEB_URL)
web_docs = web_loader.load()
print("Webpage loaded successfully.")

# === Text Splitting ===
splitter = Splitter(separator=SEPARATOR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
pdf_chunks = splitter.split(pdf_docs)
web_chunks = splitter.split(web_docs)

# === Embedding Model ===
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
    api_version=os.getenv("API_VERSION"),
)

# === In-Memory Vector Store ===
pdf_vectorstore = Chroma.from_documents(pdf_chunks, embeddings)
web_vectorstore = Chroma.from_documents(web_chunks, embeddings)

# === Querying ===
pdf_retriever = pdf_vectorstore.as_retriever()
web_retriever = web_vectorstore.as_retriever()

pdf_results = pdf_retriever.invoke(QUERY)
web_results = web_retriever.invoke(QUERY)

# === Summarization ===
summarizer = Summarizer(summary_sentences=3)

pdf_text = "\n".join(doc.page_content for doc in pdf_results)
web_text = "\n".join(doc.page_content for doc in web_results)

pdf_summary = summarizer.summarize(pdf_text)
web_summary = summarizer.summarize(web_text)

# === Output Results ===
print("\n PDF Summary:\n", pdf_summary)
print("\n Webpage Summary:\n", web_summary)
