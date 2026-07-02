import uuid
from time import perf_counter

from fastapi import Request, Response

from src.services.logger import logger

# リクエストIDを state に保存するためのキー名
REQUEST_ID_KEY = "request_id"


def get_client_ip(request: Request) -> str:
    """proxy 配下でも実 IP を追えるよう代表的な転送ヘッダを優先する"""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    if request.client is None:
        return "-"

    return request.client.host


async def logger_middleware(request: Request, call_next) -> Response:
    """request / response のログを一元化する"""
    start = perf_counter()
    request_path = request.url.path
    connection_host = request.headers.get("host", "-")
    client_ip = get_client_ip(request)

    # リクエストごとに UUID を振り、ログとレスポンスで一貫性を持たせる
    request_id = str(uuid.uuid4())
    setattr(request.state, REQUEST_ID_KEY, request_id)

    request_context = (
        f"{request.method} {request_path} "
        f"req_id={request_id} host={connection_host} client_ip={client_ip}"
    )

    logger.info(f"Incoming request: {request_context}")

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.exception(
            f"Request failed ({elapsed_ms:.2f}ms) {request_context}",
        )
        raise

    elapsed_ms = (perf_counter() - start) * 1000
    logger.info(
        f"Response status: {response.status_code} "
        f"({elapsed_ms:.2f}ms) {request_context}",
    )

    # クライアント側でもリクエストIDを追跡できるようヘッダに付与する
    response.headers["X-Request-Id"] = request_id
    return response
