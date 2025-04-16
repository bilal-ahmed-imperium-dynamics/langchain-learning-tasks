from dotenv import load_dotenv
import os

load_dotenv()

# API_VERSION=2024-05-01-preview
# DEPLOYMENT_NAME_GPT=gpt-4o-mini
# DEPLOYMENT_NAME_EMBEDDING=text-embedding-3-small
# ENDPOINT_URL=https://azure-openai-interns.openai.azure.com/
# AZURE_OPENAI_API_KEY=***

# Retrieve and print the variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("API_VERSION")
deployment_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")

print("Azure OpenAI Configuration:")
print(f"API Key: {'*' * len(api_key) if api_key else 'Not set'}")
print(f"Endpoint: {endpoint}")
print(f"Deployment Name: {deployment}")
print(f"Deployment Name Embedding: {deployment_embedding}")
print(f"API Version: {api_version}")