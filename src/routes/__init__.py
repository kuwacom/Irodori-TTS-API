from fastapi import APIRouter

from src.lib.api_error import ErrorCode, api_error
from src.routes.v1 import v1_router

router = APIRouter()

router.include_router(v1_router, prefix="/v1")


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def fallback(path: str) -> None:
    del path
    raise api_error(ErrorCode.FORBIDDEN)
