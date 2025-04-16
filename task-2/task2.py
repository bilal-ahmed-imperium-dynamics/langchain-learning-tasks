import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

class AzureOpenAIConfig:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("ENDPOINT_URL")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME_GPT")
        self.api_version = os.getenv("API_VERSION")


class Summarizer:
    def __init__(self, summary_sentences=3):
        config = AzureOpenAIConfig()

        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=f"Summarize the following text in exactly {summary_sentences} sentence(s):\n\n{{text}}"
        )

        self.llm = AzureChatOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            temperature=0.7,
        )

        self.chain = self.prompt | self.llm

    def summarize(self, text: str) -> str:
        return self.chain.invoke({"text": text}).content


if __name__ == "__main__":
    input_text = """
            Artificial intelligence (AI) is transforming industries by automating tasks, improving decision-making, and enabling new innovations. 
            From healthcare diagnostics to self-driving cars, AI applications are expanding rapidly. Machine learning, a subset of AI, allows systems to 
            learn from data without explicit programming. Deep learning, using neural networks, powers advanced applications like image recognition and 
            natural language processing. However, AI also raises ethical concerns, including job displacement and bias in algorithms. Governments and 
            organizations are working on regulations to ensure responsible AI development. Future advancements may include general AI, which could perform 
            any intellectual task a human can. Researchers are also exploring ways to make AI more transparent and explainable. Despite challenges, AI's 
            potential to improve efficiency and solve complex problems makes it a key technology for the future.
        """

    print("\n3 Sentence Summary: ")
    summarizer3 = Summarizer(summary_sentences=3)
    print(summarizer3.summarize(input_text))

    print("\n1 Sentence Summary: ")
    summarizer1 = Summarizer(summary_sentences=1)
    print(summarizer1.summarize(input_text))
