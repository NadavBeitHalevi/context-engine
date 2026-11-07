"""FastAPI entry point for the Context Engine service.

Run locally:
    uv run uvicorn app:app --reload
"""

import logging

from fastapi import FastAPI

from config import get_settings
from controllers.health_controller import router as health_router

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description="Takes a user goal, plans it with the Context Engine, and runs specialist agents to produce the output.",
    version="0.1.0",
)

app.include_router(health_router)
