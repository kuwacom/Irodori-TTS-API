import uvicorn

from src.configs.env import env
from src.services.logger import logger


def main() -> None:
    """アプリケーションサーバーを起動する"""
    logger.info(f"Server is running on: http://{env.host}:{env.port}")
    uvicorn.run(
        "src.app:app",
        host=env.host,
        port=env.port,
        reload=env.reload,
        log_config=None,
    )


if __name__ == "__main__":
    main()
