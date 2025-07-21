#!/usr/bin/env python3
"""
HTTP API Adapter
Alternative interface to core music analysis service

Provides REST API access to music21 functionality when MCP fails.
Completely independent of MCP protocol concerns.
"""

import builtins
import contextlib
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..services import MusicAnalysisService


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
        self._setup_routes()

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
            """Detailed health check"""
            return {
                "status": "healthy",
                "service": "Music21 Analysis API",
                "core_service": "MusicAnalysisService",
                "tools": self.core_service.get_available_tools(),
                "scores_count": self.core_service.get_score_count(),
            }

        # === Core Operations ===

        @self.app.post("/scores/import")
        async def import_score(request: ImportRequest):
            """Import a score from various sources"""
            try:
                result = await self.core_service.import_score(
                    request.score_id, request.source, request.source_type
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/scores/upload")
        async def upload_score(score_id: str, file: UploadFile = File(...)):
            """Upload and import a score file"""
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file.filename or "").suffix
                ) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                # Import from temporary file
                result = await self.core_service.import_score(
                    score_id, tmp_path, "file"
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
                result = await self.core_service.analyze_key(request.score_id)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/chords")
        async def analyze_chords(request: AnalysisRequest):
            """Analyze chord progressions in a score"""
            try:
                result = await self.core_service.analyze_chords(request.score_id)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/harmony")
        async def analyze_harmony(request: HarmonyAnalysisRequest):
            """Perform harmony analysis"""
            try:
                result = await self.core_service.analyze_harmony(
                    request.score_id, request.analysis_type
                )
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/voice-leading")
        async def analyze_voice_leading(request: AnalysisRequest):
            """Analyze voice leading quality"""
            try:
                result = await self.core_service.analyze_voice_leading(request.score_id)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/analysis/patterns")
        async def recognize_patterns(request: PatternRequest):
            """Recognize musical patterns"""
            try:
                result = await self.core_service.recognize_patterns(
                    request.score_id, request.pattern_type
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
    import uvicorn

    app = create_http_server()

    print("🎵 Music21 HTTP API Server")
    print("📊 REST API alternative to MCP protocol")
    print("🌐 Starting server on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
