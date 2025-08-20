#!/usr/bin/env python3
"""
HTTP API Adapter
Alternative interface to core music analysis service

Provides REST API access to music21 functionality when MCP fails.
Completely independent of MCP protocol concerns.
"""

import asyncio
import builtins
import contextlib
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..health_checks import (
    get_health_checker,
    health_check,
    liveness_check,
    readiness_check,
)
from ..rate_limiter import RateLimitStrategy, create_rate_limiter
from ..services import MusicAnalysisService

# Security constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit for file uploads

# Timeout constants
HTTP_REQUEST_TIMEOUT = 60.0  # 60 seconds for HTTP operations
UPLOAD_TIMEOUT = 120.0  # 2 minutes for file uploads


# Request/Response Models
class ImportRequest(BaseModel):
    score_id: str
    source: str
    source_type: str = "corpus"


class AnalysisRequest(BaseModel):
    score_id: str


class HarmonyAnalysisRequest(BaseModel):
    score_id: str
    analysis_type: str = "roman"


class ExportRequest(BaseModel):
    score_id: str
    format: str = "musicxml"


class PatternRequest(BaseModel):
    score_id: str
    pattern_type: str = "melodic"


class HTTPAdapter:
    """
    HTTP/REST API adapter for music analysis service

    Provides web API access to core music analysis functionality.
    Independent of MCP protocol - works when MCP fails.
    """

    def __init__(self):
        """Initialize HTTP adapter with core service"""
        self.core_service = MusicAnalysisService()
        self.app = FastAPI(
            title="Music21 Analysis API",
            description="REST API for music analysis - MCP-independent",
            version="1.0.0",
        )
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup middleware for request tracking and monitoring"""
        import time

        from fastapi import Request

        # Add rate limiting middleware
        rate_limiter = create_rate_limiter(
            requests_per_minute=60,
            requests_per_hour=1000,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )

        @self.app.middleware("http")
        async def apply_rate_limiting(request: Request, call_next):
            """Apply rate limiting to all requests"""
            return await rate_limiter(request, call_next)

        @self.app.middleware("http")
        async def track_requests(request: Request, call_next):
            """Track request metrics for health monitoring"""
            start_time = time.time()

            # Skip health check endpoints to avoid circular dependency
            if request.url.path.startswith("/health"):
                response = await call_next(request)
                return response

            try:
                response = await call_next(request)

                # Record successful request
                response_time_ms = (time.time() - start_time) * 1000
                health_checker = get_health_checker()
                health_checker.record_request(response_time_ms, success=True)

                # Add response headers
                response.headers["X-Response-Time-ms"] = str(response_time_ms)
                return response

            except Exception as e:
                # Record failed request
                response_time_ms = (time.time() - start_time) * 1000
                health_checker = get_health_checker()
                health_checker.record_request(response_time_ms, success=False)
                raise

    async def _with_timeout(self, coro, timeout: float = HTTP_REQUEST_TIMEOUT):
        """
        Wrapper to apply timeout to async operations with graceful error handling

        Args:
            coro: The coroutine to execute
            timeout: Timeout in seconds

        Returns:
            Result of the coroutine

        Raises:
            HTTPException: 504 Gateway Timeout if operation exceeds timeout
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Operation timed out after {timeout} seconds. "
                "The request took too long to process, possibly due to "
                "complex music analysis or high server load.",
            )

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            """API root - health check"""
            return {
                "service": "Music21 Analysis API",
                "status": "healthy",
                "tools_available": len(self.core_service.get_available_tools()),
                "scores_loaded": self.core_service.get_score_count(),
            }

        @self.app.get("/health")
        async def health():
            """Comprehensive health check with detailed diagnostics"""
            return await health_check()

        @self.app.get("/health/ready")
        async def ready():
            """Readiness check - is service ready to handle requests?"""
            result = await readiness_check()
            if not result["ready"]:
                raise HTTPException(status_code=503, detail="Service not ready")
            return result

        @self.app.get("/health/live")
        async def live():
            """Liveness check - is service alive?"""
            result = await liveness_check()
            if not result["alive"]:
                raise HTTPException(status_code=503, detail="Service not alive")
            return result

        # === Core Operations ===

        @self.app.post("/scores/import")
        async def import_score(request: ImportRequest):
            """Import a score from various sources"""
            try:
                result = await self._with_timeout(
                    self.core_service.import_score(
                        request.score_id, request.source, request.source_type
                    )
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/scores/upload")
        async def upload_score(score_id: str, file: UploadFile = File(...)):
            """Upload and import a score file"""
            try:
                # Check file size before processing
                content = await file.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
                    )

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file.filename or "").suffix
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                # Import from temporary file with longer timeout for file uploads
                result = await self._with_timeout(
                    self.core_service.import_score(score_id, tmp_path, "file"),
                    timeout=UPLOAD_TIMEOUT,
                )

                # Cleanup
                os.unlink(tmp_path)

                return JSONResponse(content=result)
            except Exception as e:
                # Cleanup on error
                if "tmp_path" in locals():
                    with contextlib.suppress(builtins.BaseException):
                        os.unlink(tmp_path)
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.get("/scores")
        async def list_scores():
            """List all imported scores"""
            try:
                result = await self.core_service.list_scores()
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/scores/{score_id}")
        async def get_score_info(score_id: str):
            """Get detailed information about a score"""
            try:
                result = await self.core_service.get_score_info(score_id)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

        @self.app.post("/scores/{score_id}/export")
        async def export_score(score_id: str, request: ExportRequest):
            """Export a score to various formats"""
            try:
                result = await self.core_service.export_score(score_id, request.format)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.delete("/scores/{score_id}")
        async def delete_score(score_id: str):
            """Delete a score from storage"""
            try:
                result = await self.core_service.delete_score(score_id)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

        # === Analysis Operations ===

        @self.app.post("/analysis/key")
        async def analyze_key(request: AnalysisRequest):
            """Analyze the key signature of a score"""
            try:
                result = await self._with_timeout(
                    self.core_service.analyze_key(request.score_id)
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/chords")
        async def analyze_chords(request: AnalysisRequest):
            """Analyze chord progressions in a score"""
            try:
                result = await self._with_timeout(
                    self.core_service.analyze_chords(request.score_id)
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/harmony")
        async def analyze_harmony(request: HarmonyAnalysisRequest):
            """Perform harmony analysis"""
            try:
                result = await self._with_timeout(
                    self.core_service.analyze_harmony(
                        request.score_id, request.analysis_type
                    )
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/voice-leading")
        async def analyze_voice_leading(request: AnalysisRequest):
            """Analyze voice leading quality"""
            try:
                result = await self._with_timeout(
                    self.core_service.analyze_voice_leading(request.score_id)
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/patterns")
        async def recognize_patterns(request: PatternRequest):
            """Recognize musical patterns"""
            try:
                result = await self._with_timeout(
                    self.core_service.recognize_patterns(
                        request.score_id, request.pattern_type
                    )
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        # === API Documentation ===

        @self.app.get("/tools")
        async def list_tools():
            """Get list of all available analysis tools"""
            return {
                "tools": self.core_service.get_available_tools(),
                "count": len(self.core_service.get_available_tools()),
                "description": "Music analysis tools available via HTTP API",
            }

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app


def create_http_server():
    """Factory function to create HTTP server"""
    adapter = HTTPAdapter()
    return adapter.get_app()


# For direct execution
if __name__ == "__main__":
    import os

    import uvicorn

    app = create_http_server()

    # Get host and port from environment variables with defaults
    # SECURITY: Default to localhost only. Set MUSIC21_MCP_HOST=0.0.0.0 to bind to all interfaces
    # WARNING: Binding to 0.0.0.0 exposes the service to external network access
    host = os.getenv("MUSIC21_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MUSIC21_MCP_PORT", "8000"))
    display_host = "localhost" if host in ["0.0.0.0", "127.0.0.1"] else host

    print("🎵 Music21 HTTP API Server")
    print("📊 REST API alternative to MCP protocol")
    print(f"🌐 Starting server on http://{display_host}:{port}")
    print(f"📖 API docs: http://{display_host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)  # noqa: S104 # HTTP server needs to bind to all interfaces
