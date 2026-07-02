import logging

from rich.logging import RichHandler

from src.configs.env import env


def create_logger() -> logging.Logger:
    """アプリケーション全体で共有する logger を生成する"""
    logger = logging.getLogger("fastapi_template")
    logger.setLevel(env.log_level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,
        highlighter=None,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    return logger


logger = create_logger()
