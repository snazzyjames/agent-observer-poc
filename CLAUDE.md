# Agent Observer

## What this is
An observability platform for agentic pipelines. The core problem being solved
is that agentic systems are nondeterministic — the same pipeline given the same
input can take meaningfully different paths, making traditional logging
insufficient for understanding, debugging, or ensuring safe behavior.

## Origin of the problem
This project is informed by direct experience building a multi-agent SDLC
pipeline in a production environment. The primary failure mode observed was
coherence degradation in iterative development loops — agents losing the thread
of the original task specification as context accumulated across iterations,
combined with sparse system prompts that provided insufficient anchoring.

## Current goal
Build a proof of concept using LangGraph that:
- Simulates a multi-agent development loop that exhibits coherence degradation
- Implements an observability layer that intercepts at each node transition
- Surfaces system prompts, context size, and agent reasoning at each step
- Scores semantic drift between the original task specification and current
  agent framing using embeddings

## Tech stack
- LangGraph for the pipeline
- Ollama for local inference (model: agent-coder, based on qwen2.5-coder:14b)
- nomic-embed-text via Ollama for embeddings
- Python with mise for environment management

## What to avoid
- External API calls for inference — keep everything local via Ollama
- Over-engineering before the POC validates the core thesis
