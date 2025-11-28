import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "DeepFake Detection API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./deepfake.db")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = ["mp4", "mov", "avi", "jpg", "jpeg", "png"]
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models")
    INFERENCE_SERVICE_URL: str = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8001")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000"
    ]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Processing
    DEFAULT_SEQUENCE_LENGTH: int = 16
    DEFAULT_FPS: float = 2.0
    MAX_PROCESSING_TIME: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()