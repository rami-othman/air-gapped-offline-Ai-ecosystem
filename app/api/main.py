from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .routes.health import router as health_router
from .routes.rag import router as rag_router
from .services.rag_service import RAGServiceError

app = FastAPI(
    title="Local RAG API",
    version="1.0.0",
    description="FastAPI wrapper around the existing local RAG pipeline.",
)

app.include_router(health_router)
app.include_router(rag_router)


@app.exception_handler(RAGServiceError)
async def rag_service_error_handler(_: Request, exc: RAGServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
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
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": str(exc) or "Unexpected server error.",
        },
    )
