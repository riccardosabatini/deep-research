# Deep Research Python

A Python implementation of the Deep Research agent, powered by LangGraph and LangChain. This project allows you to run deep, iterative research tasks with multiple AI models and search providers, featuring a human-in-the-loop workflow for refining research direction.

## Features

*   **LangGraph Architecture**: Stateful, cyclic graph workflow for robust orchestration.
*   **Async & Parallel**: Executes search queries in parallel for speed.
*   **Persistence**: Saves state to a local SQLite database (swappable for Postgres), allowing pause/resume and history inspection.
*   **Human-in-the-Loop**: Interactive review step allows you to add new queries mid-process.
*   **Configurable LLMs**: Support for different models for "Thinking" (Planning) and "Task" (Execution), compatible with OpenAI, DeepSeek, OpenRouter, etc.

## Prerequisites

*   Python 3.10+
*   API Keys for:
    *   LLM Provider (OpenAI, DeepSeek, etc.)
    *   Tavily (for Search)

## Installation

1.  **Clone the repository** (if you haven't already).

2.  **Run the setup script**:
    ```bash
    ./setup.sh
    ```
    This will create a virtual environment and install dependencies.

    *Alternatively, manually:*
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your API keys.

## Configuration

You can configure the AI provider and models in `.env`. The system uses `langchain_openai`, so it supports any OpenAI-compatible provider.

```ini
# API Keys
TAVILY_API_KEY=tvly-...
AI_API_KEY=sk-...

# Provider Settings
AI_PROVIDER=openai
AI_BASE_URL=        # Leave empty for OpenAI, or set for others
THINKING_MODEL=gpt-4o
TASK_MODEL=gpt-4o
```

### Database Configuration

The system supports both SQLite (default) and PostgreSQL for persisting research state and results.

**SQLite (Default):**
```ini
DB_PROVIDER=sqlite
DB_URI=checkpoints.db
```

**PostgreSQL:**
```ini
DB_PROVIDER=postgres
DB_URI=postgresql://user:password@localhost:5432/deepresearch
```

To initialize the database tables manually (optional, as the app does it automatically):
```bash
python init_db.py
```

### Supported Providers

| Provider | `AI_PROVIDER` | `AI_BASE_URL` | Notes |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `openai` | *(Empty)* | Default. |
| **DeepSeek** | `deepseek` | `https://api.deepseek.com` | Excellent for reasoning (`deepseek-reasoner`). |
| **OpenRouter** | `openrouter` | `https://openrouter.ai/api/v1` | Access to Claude, Gemini, Llama, etc. |
| **Ollama** | `ollama` | `http://localhost:11434/v1` | Local models. |
| **Groq** | `groq` | `https://api.groq.com/openai/v1` | Fast inference. |
| **xAI (Grok)** | `xai` | `https://api.x.ai/v1` | |
| **Together AI** | `together` | `https://api.together.xyz/v1` | |

*Note: For providers like Anthropic or Google Vertex directly (not via OpenRouter), you would need to modify `src/nodes.py` to use their specific LangChain classes (`ChatAnthropic`, `ChatVertexAI`).*

## Usage

Run the research agent with a query:

```bash
source venv/bin/activate
python main.py "The future of quantum computing in 2025"
```

### The Workflow
1.  **Plan**: The agent creates a research plan.
2.  **Generate**: It generates a list of initial search queries.
3.  **Search**: It executes all queries in parallel.
4.  **Review (Interactive)**: The system pauses and shows you the results. You can:
    *   Type feedback to generate *more* queries (e.g., "Also look into superconducting qubits").
    *   Press Enter to proceed to the final report.
5.  **Report**: The agent synthesizes a final markdown report.

## Project Structure

*   `main.py`: Entry point and orchestration loop.
*   `src/graph.py`: LangGraph workflow definition.
*   `src/nodes.py`: Core logic nodes (Plan, Search, Write).
*   `src/models.py`: Pydantic data models and State definition.
*   `src/configuration.py`: Configuration management.
*   `src/tools.py`: Search tool wrapper (Tavily).
*   `src/prompts.py`: Prompt templates.
