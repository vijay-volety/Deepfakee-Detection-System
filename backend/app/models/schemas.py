from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    RESNET_LSTM = "resnet_lstm"
    EFFICIENTNET_LSTM = "efficientnet_lstm"
    XCEPTION_LSTM = "xception_lstm"


# Request Models
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RetrainRequest(BaseModel):
    dataset_paths: List[str]
    hyperparams: Dict[str, Any] = {}
    model_type: ModelType = ModelType.RESNET_LSTM


class SetActiveModelRequest(BaseModel):
    model_id: str


# Response Models
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_status: str
    redis_status: str
    model_status: str
    version: str


class DetectionJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    estimated_completion: Optional[datetime] = None


class FrameScore(BaseModel):
    frame_index: int
    timestamp: float
    deepfake_score: float
    confidence: float


class DetectionResult(BaseModel):
    authentic_percentage: float = Field(..., ge=0, le=100)
    deepfake_percentage: float = Field(..., ge=0, le=100)
    overall_confidence: float = Field(..., ge=0, le=1)
    frame_scores: List[FrameScore]
    model_version: str
    processing_time: float
    
    @validator('deepfake_percentage')
    def validate_percentages_sum(cls, v, values):
        if 'authentic_percentage' in values:
            if abs((v + values['authentic_percentage']) - 100.0) > 0.1:
                raise ValueError('Authentic and deepfake percentages must sum to 100')
        return v


class SaliencyMap(BaseModel):
    frame_index: int
    heatmap_url: str
    suspicious_regions: List[Dict[str, float]]


class DetectionResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[DetectionResult] = None
    saliency_maps: List[SaliencyMap] = []
    error_message: Optional[str] = None


class DetectionJobSummary(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None


class AdminStatsResponse(BaseModel):
    total_jobs: int
    jobs_today: int
    successful_jobs: int
    failed_jobs: int
    average_processing_time: float
    active_model: str
    model_accuracy: float
    system_uptime: float


class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    module: str
    job_id: Optional[str] = None


class RetrainJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    estimated_duration: str


class ModelInfo(BaseModel):
    model_id: str
    name: str
    version: str
    type: ModelType
    accuracy: Optional[float] = None
    created_at: datetime
    is_active: bool
    file_size: int
    description: str


class FileValidationResult(BaseModel):
    is_valid: bool
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    
    
class WebcamFrame(BaseModel):
    frame_data: str  # Base64 encoded image
    timestamp: float
    frame_index: int


class BatchDetectionRequest(BaseModel):
    files: List[str]  # File paths or URLs
    batch_name: str
    user_consent: bool = False


class SystemConfig(BaseModel):
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = ["mp4", "mov", "avi", "jpg", "jpeg", "png"]
    max_sequence_length: int = 50
    default_fps: float = 2.0
    enable_gpu: bool = True
    model_cache_size: int = 3