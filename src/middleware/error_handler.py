from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.lib.api_error import ApiError, ErrorCode, api_error
from src.services.logger import logger


def setup_exception_handlers(app: FastAPI) -> None:
    """API 共通の失敗出口を FastAPI に登録する"""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        del request

        if exc.is_expected:
            logger.warning(f"ApiError: {exc.code} - {exc.message}")
        else:
            logger.exception(f"ApiError: {exc.code}")

        return JSONResponse(status_code=exc.status_code, content=exc.to_response())

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request

        validation_error = api_error(ErrorCode.VALIDATION_ERROR, exc.errors())
        logger.warning(f"ValidationError: {validation_error.details}")
        return JSONResponse(
            status_code=validation_error.status_code,
            content=validation_error.to_response(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del request

        logger.exception("Unexpected error", exc_info=exc)
        error = api_error(ErrorCode.INTERNAL_SERVER_ERROR)
        return JSONResponse(status_code=error.status_code, content=error.to_response())
