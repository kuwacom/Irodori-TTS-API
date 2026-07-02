from fastapi import APIRouter

from src.lib.api_error import ErrorCode, api_error
from src.routes.v1.speakers import speakers_router
from src.routes.v1.synthesize import synthesize_router

v1_router = APIRouter()

v1_router.include_router(speakers_router, prefix="/speakers", tags=["speakers"])
v1_router.include_router(synthesize_router, prefix="/synthesize", tags=["synthesize"])


@v1_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def fallback(path: str) -> None:
    del path
    raise api_error(ErrorCode.NOT_FOUND)
