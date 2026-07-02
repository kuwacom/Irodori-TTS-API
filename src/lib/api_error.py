from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """API 全体で共有するエラーコード"""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class ApiError(Exception):
    """HTTP レスポンスへ変換するための API 共通例外"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int,
        details: Any | None = None,
        is_expected: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.is_expected = is_expected

    def to_response(self) -> dict[str, Any]:
        """クライアントへ返すエラー形式へ変換する"""
        response: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }

        if self.details is not None:
            response["details"] = self.details

        return response


def api_error(
    code: ErrorCode,
    details: Any | None = None,
    message: str | None = None,
) -> ApiError:
    """エラーコードごとの HTTP status とメッセージを一箇所に集約する"""
    if code == ErrorCode.VALIDATION_ERROR:
        return ApiError(code, message or "Validation error", 400, details)

    if code == ErrorCode.FORBIDDEN:
        return ApiError(code, message or "Forbidden", 403, details)

    if code == ErrorCode.NOT_FOUND:
        return ApiError(code, message or "Not found", 404, details)

    return ApiError(
        ErrorCode.INTERNAL_SERVER_ERROR,
        message or "Internal server error",
        500,
        details,
        is_expected=False,
    )
