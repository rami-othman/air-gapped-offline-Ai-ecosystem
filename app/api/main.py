import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import WARMUP_ON_STARTUP
from .config import API_DEV_MODE
from .routes.admin import router as admin_router
from .routes.feedback import router as feedback_router
from .routes.health import router as health_router
from .routes.rag import router as rag_router
from .services.rag_service import RAGServiceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local RAG API",
    version="1.0.0",
    description="FastAPI wrapper around the existing local RAG pipeline.",
    debug=API_DEV_MODE,
)

app.include_router(health_router)
app.include_router(rag_router)
app.include_router(feedback_router)
app.include_router(admin_router)


@app.on_event("startup")
def warmup_ollama_on_startup() -> None:
    if not WARMUP_ON_STARTUP:
        logger.info("Ollama startup warm-up skipped; WARMUP_ON_STARTUP=false.")
        return

    try:
        from ..ollama_warmup import warmup_all_models

        logger.info("Starting Ollama startup warm-up.")
        results = warmup_all_models()
        logger.info("Ollama startup warm-up completed. models_warmed=%d", len(results))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama startup warm-up failed; API startup will continue. error=%s", exc)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        logger.info(
            "HTTP request completed. method=%s path=%s status_code=%s duration_sec=%.4f",
            request.method,
            request.url.path,
            status_code,
            elapsed,
        )


@app.exception_handler(RAGServiceError)
async def rag_service_error_handler(_: Request, exc: RAGServiceError) -> JSONResponse:
    logger.warning("RAG service error. code=%s status_code=%d", exc.error_code, exc.status_code)
    content = {
        "status": "error",
        "error": exc.error_code,
        "message": exc.message,
    }
    if exc.details is not None:
        content["details"] = exc.details

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP exception raised. status_code=%d", exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "request_error",
            "message": str(exc.detail),
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error raised.")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid request payload.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error.", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Internal server error.",
        },
    )
