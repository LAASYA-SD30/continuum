# Continuum

A shared, provider-agnostic memory layer for AI agents.

## What it does

Most AI agents build memory in isolation — a chatbot, a scheduling tool, and a coding assistant each maintain separate, disconnected context about the same user. Continuum solves this by giving any number of agents a shared memory pool: one agent writes a fact, and any other agent — regardless of which AI model it uses — can retrieve it.

## Core features

- **Write API** — save facts with source, timestamp, and confidence score
- **Semantic search** — retrieve facts by meaning, not just exact keyword matches, using sentence embeddings
- **Conflict detection** — flags contradicting facts instead of silently duplicating or overwriting them
- **Governance layer** — view, edit, delete, or pin any stored fact
- **Memory expiry** — facts can auto-expire unless explicitly pinned as permanent
- **Multi-agent support** — proven with two independent agents (a chat demo and a scheduling tool) sharing one memory pool through the same API
- **Data export** — full memory downloadable in an open JSON format, avoiding vendor lock-in
- **API key authentication** — each developer's data is isolated and secured

## Tech stack

- **Backend:** Python, FastAPI
- **Database:** Supabase (Postgres + pgvector)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), generated locally, no external API dependency
- **LLM integration:** Groq API (provider-agnostic design — swappable per agent)

## Project structure

- `main.py` — the core API: write, retrieve, search, conflict resolution, governance, export
- `chat.py` — a reference conversational agent demonstrating memory in use
- `scheduler.py` — a second, independent agent proving multi-agent shared memory

## Status

Core memory engine is built and tested end-to-end. In progress: a visual governance dashboard, and production deployment.

## Author

Built by Laasya S D as part of an AI/ML internship project.
