import os
import sys
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

load_dotenv()

file_path = r"C:\Users\dell\langchain-learning-tasks\task-3\ai_intro.txt"
with open(file_path, 'r') as file:
    full_text = file.read()

summarizer = Summarizer(summary_sentences=3)
summary = summarizer.summarize(full_text)

qa_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="Answer the following question about the given text concisely:\n\nText: {text}\n\nQuestion: {question}\nAnswer:"
)

llm = summarizer.llm

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

question = "What's the key event mentioned?"
summary_answer = qa_chain.invoke({"text": summary, "question": question})['text']

full_text_answer = qa_chain.invoke({"text": full_text, "question": question})['text']

print("\nSummary text:")
print(summary)
print("\nAnswer from summary:")
print(summary_answer)
print("\nAnswer from full text:")
print(full_text_answer)