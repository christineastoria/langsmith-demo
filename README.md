# LangSmith Demo: Museum Description Generator

A comprehensive demonstration of LangSmith's capabilities for LLM observability, prompt engineering, testing, and monitoring. This project showcases how to build, trace, evaluate, and improve an AI application that generates museum labels for artworks using Wikipedia as a knowledge source.

## Project Overview

This demo implements a **Museum Description Generator** that:
- Retrieves context about artworks from Wikipedia
- Uses LLMs to generate scholarly, public-facing museum labels
- Demonstrates LangSmith's full evaluation lifecycle
- Includes both simple function-based and LangGraph-based implementations

## 🏗️ Architecture

The project includes two main implementations:

### 1. Simple Function-Based Approach (`langsmith_demo.ipynb`)
- Uses `@traceable` decorators for observability
- Implements Wikipedia retrieval → LLM generation pipeline
- Demonstrates manual tracing and metadata

### 2. LangGraph-Based Approach (`graph.py`)
- Uses LangGraph for workflow orchestration
- Automatic tracing with LangSmith integration
- Includes SQLite checkpointing for persistence
- Configurable via `langgraph.json`

## 📁 Project Structure

```
langsmith-demo/
├── README.md                    # This file
├── langsmith_demo.ipynb        # Main demo notebook
├── graph.py                    # LangGraph implementation
├── langgraph.json             # LangGraph configuration
├── images/                    # Demo images and diagrams
│   ├── museum-app.jpg
│   ├── museum-app.pdf
│   └── evaluation_lifecycle.png
├── ls-academy/               # Virtual environment
└── readMe.ME                # Original placeholder file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LangSmith account and API key
- OpenAI API key

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd langsmith-demo
   ```

2. **Set up environment variables:**
   ```bash
   export LANGSMITH_API_KEY="your-langsmith-api-key"
   export LANGCHAIN_API_KEY="your-langchain-api-key"  # Same as LangSmith
   export LANGCHAIN_TRACING_V2="true"
   export LANGCHAIN_PROJECT="langsmith-demo"
   ```

3. **Install dependencies:**
   ```bash
   pip install langsmith langchain langchain-openai langchain-community langgraph wikipedia python-dotenv jupyter
   ```

4. **Run the demo:**
   ```bash
   # Option 1: Jupyter Notebook
   jupyter notebook langsmith_demo.ipynb
   
   # Option 2: LangGraph Studio
   langgraph dev
   ```

## 🔍 Key Features Demonstrated

### 1. Observability & Tracing
- **Function Tracing**: Using `@traceable` decorators
- **LangGraph Integration**: Automatic tracing for graph-based workflows
- **Metadata & Filtering**: Rich metadata for run organization
- **Run Types**: Different run types (retriever, llm, chain)

### 2. Prompt Engineering
- **Prompt Hub Integration**: Pulling prompts from LangSmith's prompt hub
- **Template Management**: Versioned prompt templates
- **Prompt Canvas**: Visual prompt development

### 3. Testing & Evaluation
- **Dataset Creation**: Golden dataset with reference outputs
- **Custom Evaluators**: Rule-based evaluation functions
- **LLM-as-Judge**: Automated evaluation using LLMs
- **Annotation Queues**: Human feedback integration

### 4. Monitoring & Improvements
- **Online Evaluations**: Real-time evaluation capabilities
- **Automations**: Automated workflows and webhooks
- **Dashboards**: Prebuilt and custom monitoring dashboards

## 📊 Evaluation Lifecycle

The project demonstrates the complete evaluation lifecycle:

1. **Data Collection**: Creating golden datasets with reference outputs
2. **Evaluation Setup**: Configuring custom and LLM-based evaluators
3. **Experiment Running**: Comparing different prompts and configurations
4. **Analysis**: Reviewing results and identifying improvements
5. **Iteration**: Refining prompts and models based on feedback

## 🎨 Example Usage

### Simple Museum Label Generation

```python
from langsmith import traceable

@traceable(run_type="chain")
def museum_description_generator(artwork_name: str) -> str:
    # Retrieve Wikipedia context
    docs = retrieve_wikipedia(artwork_name)
    
    # Generate museum label
    messages = build_messages(artwork_name, docs)
    response = call_openai(messages)
    
    return response.choices[0].message.content

# Generate a label for "The Birth of Venus"
description = museum_description_generator("The Birth of Venus")
print(description)
```

### LangGraph Workflow

```python
# Using the LangGraph implementation
from graph import graph

result = graph.invoke({
    "question": "The Birth of Venus",
    "messages": []
})

museum_label = result["messages"][-1].content
print(museum_label)
```

## 📈 Evaluation Examples

The project includes comprehensive evaluation examples:

- **Concise Evaluation**: Ensures labels are appropriately sized
- **Quality Checks**: Avoids overused words like "beautiful" or "amazing"
- **Reference Comparison**: Compares against golden dataset
- **A/B Testing**: Compares different prompt versions

## 🔧 Configuration

### LangGraph Configuration (`langgraph.json`)

```json
{
  "graphs": {
    "museum": {
      "entrypoint": "graph.py:graph",
      "title": "Museum Label Graph",
      "description": "Wikipedia → Prompt → LLM"
    }
  },
  "server": {
    "port": 2024
  }
}
```

### Model Configuration

- **Default Model**: `gpt-4o-mini`
- **Temperature**: `0.2` (for consistent outputs)
- **Max Wikipedia Docs**: `2` (for focused context)

## 📚 Learning Resources

This demo covers key LangSmith concepts:

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Evaluation Best Practices](https://docs.smith.langchain.com/evaluate/)
- [Prompt Engineering Guide](https://docs.smith.langchain.com/prompts/)

## Contributing

This is a demonstration project. Feel free to:
- Experiment with different prompts
- Add new evaluation metrics
- Try different LLM providers
- Extend the museum label functionality

## License

This project is for educational and demonstration purposes.

## Support

For questions about LangSmith:
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Community](https://github.com/langchain-ai/langsmith)

---

**Note**: This demo requires valid API keys for LangSmith and OpenAI to function properly. Make sure to set up your environment variables before running the examples.
