import os
import sys
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarizer

summarizer = Summarizer(summary_sentences=1)

conversation1 = ConversationChain(
    llm=summarizer.llm,
    memory=ConversationBufferMemory(k=3),
    verbose=True
)

ml_text = """
    Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. 
    These systems improve their performance as they are exposed to more data over time. Common approaches include 
    supervised learning, unsupervised learning, and reinforcement learning. Applications range from spam filtering to 
    recommendation systems. The field has grown rapidly due to increased computational power and large datasets. 
    Challenges include overfitting, where models perform well on training data but poorly on new data. Researchers 
    continue to develop new algorithms to improve accuracy and efficiency.
"""

dl_text = """
    Deep learning is a specialized form of machine learning that uses artificial neural networks with multiple layers. 
    These deep neural networks can model complex patterns in large datasets. They have achieved remarkable success in 
    areas like computer vision, speech recognition, and natural language processing. Convolutional neural networks 
    excel at image-related tasks, while recurrent networks work well with sequential data. Training deep learning 
    models requires significant computational resources and large amounts of data. Recent advances like transformers 
    have revolutionized fields like language understanding.
"""

print("\nSummary of ML Text (using chain from Task 2): \n")
ml_summary = summarizer.summarize(ml_text)
print(ml_summary)

conversation1.predict(input=f"Remember this machine learning summary: {ml_summary}")

dl_summary_buffer = conversation1.predict(input=f"Summarize this deep learning text in 1 sentence considering the prior machine learning context: {dl_text}")

conversation2 = ConversationChain(
    llm=summarizer.llm,
    memory=ConversationSummaryMemory(llm=summarizer.llm),
    verbose=True
)

conversation2.predict(input=f"Remember this machine learning summary: {ml_summary}")

dl_summary_memory = conversation2.predict(input=f"Summarize this deep learning text in 1 sentence considering the prior machine learning context: {dl_text}")

# Comparison of results
print("\nComparison of Memory Types:")
print("\nBuffer Memory DL Summary:")
print(dl_summary_buffer)
print("\nSummary Memory DL Summary:")
print(dl_summary_memory)