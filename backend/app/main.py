import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import asyncio
import aiofiles
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from typing import Optional, List, Dict, Any
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
from passlib.context import CryptContext

from app.models.database import get_db, create_tables
from app.models.schemas import *
from app.core.config import settings
from app.detection_service import DetectionService

# Setup logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepFake Detection API",
    description="Production-ready deepfake detection system API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
detection_service = DetectionService()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting DeepFake Detection API...")
    
    # Create database tables
    await create_tables()
    
    # Initialize Redis connection
    await detection_service.initialize()
    
    # Initialize detection models
    await detection_service.load_models()
    
    logger.info("API startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    await detection_service.cleanup()
    logger.info("API shutdown completed")


# Health Check Endpoints
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_gen = get_db()
        db = await db_gen.__anext__()
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        await db.close()
        
        # Check Redis connection
        redis_status = await detection_service.check_redis_health()
        
        # Check model status
        model_status = await detection_service.check_model_health()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            database_status="healthy",
            redis_status="healthy" if redis_status else "unhealthy",
            model_status="healthy" if model_status else "unhealthy",
            version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


# Detection Endpoints
@app.post("/api/v1/detect", response_model=DetectionJobResponse)
# Removed rate limiter to fix "Too Many Requests" error
async def create_detection_job(
    file: UploadFile = File(...),
    webcam_frames: Optional[str] = Form(None),
    user_consent: bool = Form(False),
    db: AsyncSession = Depends(get_db)
):
    """Create a new deepfake detection job."""
    try:
        # Save uploaded file temporarily
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Determine file type
        file_type = file.content_type or "unknown"
        
        # Call detection service to create job
        job_data = await detection_service.create_detection_job(
            file_path=str(file_path),
            file_type=file_type,
            user_consent=user_consent
        )
        
        # Convert string status to JobStatus enum
        job_data["status"] = JobStatus(job_data["status"])
        
        return DetectionJobResponse(**job_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection job creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create detection job"
        )


@app.get("/api/v1/result/{job_id}", response_model=DetectionResultResponse)
async def get_detection_result(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get detection result by job ID."""
    try:
        # Get result from detection service
        result = await detection_service.get_detection_result(job_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Detection job not found"
            )
        
        # Convert string status to JobStatus enum
        result["status"] = JobStatus(result["status"])
        
        # Convert to response model
        return DetectionResultResponse(
            job_id=result["job_id"],
            status=result["status"],
            created_at=result["created_at"],
            completed_at=result.get("completed_at"),
            result=DetectionResult(**result["result"]) if result.get("result") else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detection result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detection result"
        )


@app.get("/api/v1/download/{job_id}/report")
async def download_report(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Download detection report in text format."""
    try:
        # Generate report using detection service (always generate text format)
        format = "txt"
        report_path = await detection_service.generate_report(job_id, format, db)
        
        # Check if report file exists
        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )
        
        # Return file response
        return FileResponse(
            path=report_path,
            filename=f"deepfake_report_{job_id}.txt",
            media_type="text/plain"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate download report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )