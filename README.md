# Agent Observer

An observability platform for agentic pipelines. Simulates a multi-agent development loop and tracks **semantic drift** — how far the agent's reasoning has wandered from the original task specification across iterations.

## How it works

At each step of the pipeline, the observer:
- Computes a **drift score** (cosine similarity between the original task embedding and the current agent framing)
- Tracks **context size** in tokens
- Surfaces the agent's **reasoning** at each node

A drift score below `0.7` is flagged as degrading.

## Prerequisites

- [Ollama](https://ollama.com) running locally
- The following models pulled:

```
ollama pull qwen2.5-coder:14b
ollama pull nomic-embed-text
```

- [mise](https://mise.jdx.dev) for Python environment management
- [uv](https://docs.astral.sh/uv/) for dependency management (`brew install uv`)

## Setup

```
mise install
uv sync
```

## Running

```
python main.py
```

By default, runs 4 iterations on a sample task. You can customise both:

```
python main.py --task "Your task description here" --iterations 6
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--task` | Build a REST API endpoint... | The task specification given to the agent |
| `--iterations` | `4` | Number of pipeline iterations to run |

## Example output

```
Agent Observer POC — Agentic Pipeline Observability
Task:       Build a REST API endpoint that validates user input...
Iterations: 4

============================================================
Step  1  |  Node: planner
============================================================
Drift score:  0.921
Context:      312 tokens
Reasoning:
...

============================================================
Summary
============================================================
Steps run:    4
Final drift:  0.743
Drift trend:  -0.178 (degraded)
Peak context: 1,847 tokens
```
