# context-engine

A FastAPI service that takes a user goal, runs a multi-agent **Context Engine** (Planner → Executor → specialist agents), and returns the generated output.
Based on the architecture in *Context Engineering for Multi-Agent Systems* by Denis Rothman (Packt).

## Layout

```
app.py          # FastAPI entry point
config.py       # Settings loaded from .env
controllers/    # HTTP routers (request/response only)
services/       # Context Engine logic (planner, executor, agents, registry)
```

## Run

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY (embeddings) and PINECONE_API_KEY
uv sync
uv run uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs.
