from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.configs.config import APP_NAME, APP_VERSION
from src.configs.env import env
from src.middleware.error_handler import setup_exception_handlers
from src.middleware.logger import logger_middleware
from src.routes import router
from src.services.logger import logger
from src.services.speaker_store import load_speakers
from src.services.tts_runtime import init_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時に speakers.json を読み込む
    load_speakers()

    logger.info("Initializing InferenceRuntime (downloading / loading models)...")
    init_runtime()
    logger.info("InferenceRuntime ready")

    yield


def create_app() -> FastAPI:
    """FastAPI アプリケーションを構成する"""
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=env.cors_policy_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(logger_middleware)

    app.include_router(router)
    setup_exception_handlers(app)

    return app


app = create_app()
