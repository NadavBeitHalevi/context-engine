from fastapi import APIRouter

from config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", 
            "app": settings.app_name, 
            "environment": settings.environment}
