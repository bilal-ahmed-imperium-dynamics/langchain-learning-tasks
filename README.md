# LangChain Learning Tasks

A project to master LangChain through tasks.
\nContributor: Bilal Ahmed

## Task 1: Azure OpenAI Environment Setup

This task configures and validates Azure OpenAI credentials using environment variables.

### Files:

- `task-1/task1.py`: Loads and prints Azure OpenAI settings from `.env`.

### Usage:

1. Create a `.env` file with:

```bash
    - AZURE_OPENAI_API_KEY=your_key_here
    - ENDPOINT_URL=your_endpoint_here
    - DEPLOYMENT_NAME=your_deployment_here
    - API_VERSION=2024-05-01-preview
    - DEPLOYMENT_NAME_EMBEDDING=your_embedding_model
```

2. Run:

```bash
python task-1\task1.py
```

## Task 2: Building a Basic Summarization Chain

This task creates a configurable summarization pipeline using Azure OpenAI and LangChain.

### Files:

- `task-2/task2.py`:
  - `AzureOpenAIConfig`: Loads Azure credentials from environment variables
  - `Summarizer`: Generates 1-3 sentence summaries using a prompt template

### Usage:

1. Ensure your `.env` has Azure OpenAI credentials (same as Task 1).
2. Run:

```bash
python task-2/task2.py
```

3. Output:
   This was the text to be summarized:

```bash
Artificial intelligence (AI) is transforming industries by automating tasks, improving decision-making, and enabling new innovations. From healthcare diagnostics to self-driving cars, AI applications are expanding rapidly. Machine learning, a subset of AI, allows systems to learn from data without explicit programming. Deep learning, using neural networks, powers advanced applications like image recognition and natural language processing. However, AI also raises ethical concerns, including job displacement and bias in algorithms. Governments and organizations are working on regulations to ensure responsible AI development. Future advancements may include general AI, which could perform any intellectual task a human can. Researchers are also exploring ways to make AI more transparent and explainable. Despite challenges, AI's potential to improve efficiency and solve complex problems makes it a key technology for the future.
```

The 3-Sentence Summary:

```bash
Artificial intelligence (AI) is revolutionizing various industries by automating tasks and enhancing decision-making, with applications ranging from healthcare diagnostics to self-driving cars. While AI offers significant benefits, it also raises ethical concerns such as job displacement and algorithmic bias, prompting governments and organizations to seek regulations for responsible development. Looking ahead, advancements like general AI and efforts to improve transparency and explainability could further harness AI's potential to solve complex problems and enhance efficiency.
```

The 1-Sentence Summary:
```bash
Artificial intelligence is revolutionizing various industries through automation and enhanced decision-making, while also presenting ethical challenges that necessitate responsible development and regulation.

```

