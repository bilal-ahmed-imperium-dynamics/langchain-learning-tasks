# LangChain Learning Tasks

A project to master LangChain through tasks.
Contributor: Bilal Ahmed

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

## Task 3: Creating Retrievers with Summarization

This task combines document retrieval with summarization using vector stores.

### Files:

- task-3/task3.py:

  - RetrieverConfig: Sets chunk size/overlap for text splitting

  - Retriever: Builds a FAISS vector store from a text file

- task-3/ai_intro.txt: Sample 500-word text about AI history

### Usage:

1. Add ai_intro.txt with your AI history content.

2. Run:

```bash
python task-3/task3.py
```

3. Output:

- Retrieves text chunks for "AI milestones" query (around half of all text).

```bash
Landmark achievements in the 21st century demonstrated AIâ€™s growing sophisticion. IBMâ€™s Deep Blue made history in 1997 by defeating chess champion Garry Kparov, while Watsonâ€™s victory on Jeopardy! in 2011 showcased its ability to uerstand and respond to natural language. Perhaps the most striking milestone ca in 2016, when Googleâ€™s AlphaGo defeated world champion Lee Sedol in the ancit game of Go, a feat once thought impossible due to the gameâ€™s complexity. The breakthroughs illustrated AIâ€™s ability to master tasks requiring intuition,reativity, and strategic thinkingâ€”qualities once considered uniquely human.
The field officially took shape in 1956 at the Dartmouth Conference, where JohncCarthy and his colleagues coined the term "Artificial Intelligence" and outlin ambitious goals for creating machines that could simulate human thought. The flowing decades saw rapid progress, with early AI systems demonstrating remarkab capabilities. Programs like the Logic Theorist could solve mathematical theore, while ELIZA simulated human conversation, offering a glimpse into natural lanage processing. By the 1980s, expert systems emerged as a dominant approach, ending human expertise into rule-based programs that excelled in specialized task Systems like MYCIN, which assisted in medical diagnosis, and DENDRAL, used forhemical analysis, showcased AIâ€™s potential in real-world applications.
The history of Artificial Intelligence (AI) stretches back to ancient civilizatns, where myths and legends spoke of artificial beings imbued with life and intligence. From the mechanical automatons of Greek mythology to the golems of Jewh folklore, humanity has long dreamed of creating thinking machines. However, t scientific journey of AI truly began in the mid-20th century, when pioneers la the groundwork for what would become one of the most transformative technologi of our time. The modern era of AI was shaped by visionary thinkers like Alan Ting, who in 1950 proposed the idea of machines capable of human-like reasoning  his seminal paper, "Computing Machinery and Intelligence." His conceptual Turi Test became a benchmark for evaluating machine intelligence, setting the stageor future research.
Looking ahead, the pursuit of Artificial General Intelligence (AGI)â€”a machineapable of performing any intellectual task a human canâ€”remains a central goal
While current AI excels in narrow domains, AGI would require systems to generale knowledge across disciplines, reason abstractly, and demonstrate true understding. Researchers are exploring approaches like neuromorphic computing (which mics the brainâ€™s architecture) and reinforcement learning (where AI learns thrgh trial and error) to bridge this gap.
```

- Generates a 3-sentence summary using Task 2's Summarizer.

```bash
The 21st century has seen remarkable advancements in AI, highlighted by IBM's Dp Blue defeating chess champion Garry Kasparov in 1997, Watson winning Jeopardy
is. The ultimate goal of achieving Artificial General Intelligence (AGI) remains a key focus, with researchers investigating innovative methods to enable machines to generalize knowledge and reason like humans.
```

## Task 4: Text Summarizer Agent

This module implements a custom TextSummarizer tool that wraps a summarization chain, integrated into a zero-shot-react-description agent powered by Azure OpenAI. The agent can process both specific and vague summarization requests.

### Features

- Custom TextSummarizer tool for text condensation

- Integration with Azure OpenAI's language model

- Handles both precise and ambiguous summarization requests

- Configurable summary length (default: 3 sentences)

### Usage

1. Provide your Azure OpenAI credentials in .env

2. The agent accepts text input for summarization

3. Example prompts:

   - Specific: "Summarize the impact of AI on healthcare"
   - Vague: "Summarize something interesting"

## Task 05: Retriever + Summarizer Custom Agent

This agent combines document retrieval (Task 3) and text summarization (Task 2) into a single pipeline using a LangChain agent with three tools:

1. **TextRetriever:** Finds relevant text from a document based on a query

2. **TextSummarizer:** Condenses retrieved text into a 3-sentence summary

3. **WordCounter:** Counts words in the summary (used only when explicitly requested)

### Usage

The agent takes two tests. For the first test, the agent accepts natural language prompt, requesting only retrieval and summarization of AI breakthroughs:

1. Basic Retrieval + Summarization:

```bash
agent.invoke("Only find and then finally summarize text about AI breakthroughs from the document.")
```

2. Retrieval + Summarization + Word Count:

```bash
agent.invoke("Find and summarize text about AI breakthroughs from the document. Add the word count at the end of the summary.")
```

### Key Features

- Zero-shot agent dynamically selects tools based on prompt instructions

- Handles both pure summarization and summary+metrics requests

- Verbose mode shows the agent's decision-making process

## Task 6: Memory-Enhanced Summarization Comparison

This script demonstrates how different memory types affect text summarization when processing related topics sequentially.

### Key Features:

- Compares ConversationBufferMemory (stores exact past interactions) vs ConversationSummaryMemory (maintains summarized context)
- Processes machine learning text first, then deep learning text with memory of prior summary
- Uses a custom Summarizer from Task 2 for initial summarization
- Limits buffer memory to last 3 interactions

### Usage

1. Set up Azure OpenAI credentials in .env

2. Run the script to:

- First summarize ML text using basic summarizer
- Then summarize DL text using both memory types
- Compare the outputs

## Task 7: Leveraging Document Loaders

### Overview:

This task demonstrates how to use LangChain to load, process, and analyze documents from multiple sources—a local PDF and a live webpage—then compare summaries retrieved from each.

It does the following:

- Loads a 2-page PDF on AI ethics.

- Scrapes a 300-word article on AI trends from the web.

- Splits each source into 150-character chunks with 30-character overlap.

- Indexes each chunk using Chroma vector stores (in-memory).

- Queries both sources with "AI challenges".

- Summarizes results using a custom summarizer (from Task 2).

- Prints and compares the summaries.

### Output

You’ll see two summaries printed:

1. One based on the PDF

2. One based on the webpage

## Task 8: Customizing with Output Parsers

The task was to enhance the summarization chain from Task 2 by returning a structured JSON output.

### Key Features:

- Integrated StructuredOutputParser to output.

- JSON contains:

  - "summary": The generated summary text.

  - "length": Character count of the summary.

- Prompt includes format instructions based on a defined schema.

### Output:

- **Input:** A 150-word text on AI applications.
- **Output:** returns valid JSON with both fields.

## Task 9: Experimenting with Multi-Query Retrieval

The task is to implement a retriever by using MultiQueryRetriever to generate 3 semantically diverse queries from a single input.

### Key features:

- Replaced standard retriever with MultiQueryRetriever for broader coverage.

- Used it to query "AI advancements" on the ai_intro.txt vector store.

- Compared the depth and detail of summaries between:

- Multi-query retrieved text.

- Single-query (Task 3) retrieved text.

### Outcome:

Multi-query retrieval produced more informative summaries, pulling in richer and more varied context from the vector store as compared to single-query retrieval.
