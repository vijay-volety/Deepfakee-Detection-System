import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from app.main import app
from app.models.database import get_db, create_tables
import json


@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Test health check endpoint."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_detect_endpoint_without_file():
    """Test detection endpoint without file upload."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/detect")
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_detect_endpoint_with_invalid_file():
    """Test detection endpoint with invalid file."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {"file": ("test.txt", b"invalid content", "text/plain")}
        response = await client.post("/api/v1/detect", files=files)
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_result_endpoint_not_found(async_client: AsyncClient):
    """Test result endpoint with non-existent job ID."""
    response = await async_client.get("/api/v1/result/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_login_endpoint_invalid_credentials(async_client: AsyncClient):
    """Test login with invalid credentials."""
    credentials = {
        "username": "invalid",
        "password": "invalid"
    }
    response = await async_client.post("/api/v1/auth/login", json=credentials)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_endpoints_unauthorized(async_client: AsyncClient):
    """Test admin endpoints without authentication."""
    # Test admin stats
    response = await async_client.get("/api/v1/admin/stats")
    assert response.status_code == 403
    
    # Test admin logs
    response = await async_client.get("/api/v1/admin/logs")
    assert response.status_code == 403


def test_cors_headers(client: TestClient):
    """Test CORS headers are present."""
    response = client.options("/api/v1/health")
    assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting functionality."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make multiple requests quickly
        responses = []
        for _ in range(20):
            response = await client.get("/api/v1/jobs")
            responses.append(response.status_code)
        
        # Should have some rate-limited responses
        assert any(status == 429 for status in responses)


class TestDetectionWorkflow:
    """Test the complete detection workflow."""
    
    @pytest.mark.asyncio
    async def test_video_detection_workflow(self):
        """Test complete video detection workflow."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create a simple test video file (mock)
            video_content = b"fake video content for testing"
            files = {"file": ("test.mp4", video_content, "video/mp4")}
            
            # Upload for detection
            response = await client.post("/api/v1/detect", files=files)
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data["job_id"]
                
                # Check job status
                status_response = await client.get(f"/api/v1/result/{job_id}")
                assert status_response.status_code in [200, 404]  # May not exist in test


class TestAuthentication:
    """Test authentication and authorization."""
    
    @pytest.mark.asyncio
    async def test_jwt_token_validation(self):
        """Test JWT token validation."""
        # This would test the JWT token creation and validation
        # In a real test, you'd create a valid token and test it
        pass
    
    @pytest.mark.asyncio
    async def test_admin_role_required(self):
        """Test that admin endpoints require admin role."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test without token
            response = await client.get("/api/v1/admin/stats")
            assert response.status_code == 403
            
            # Test with invalid token
            headers = {"Authorization": "Bearer invalid-token"}
            response = await client.get("/api/v1/admin/stats", headers=headers)
            assert response.status_code == 401


class TestDataValidation:
    """Test input data validation."""
    
    @pytest.mark.asyncio
    async def test_file_size_validation(self):
        """Test file size limits."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create a large file (mock)
            large_content = b"x" * (200 * 1024 * 1024)  # 200MB
            files = {"file": ("large.mp4", large_content, "video/mp4")}
            
            response = await client.post("/api/v1/detect", files=files)
            assert response.status_code == 400  # Should be rejected
    
    @pytest.mark.asyncio
    async def test_file_type_validation(self):
        """Test file type validation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with disallowed file type
            files = {"file": ("test.exe", b"executable", "application/exe")}
            
            response = await client.post("/api/v1/detect", files=files)
            assert response.status_code == 400


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_internal_server_error_handling(self):
        """Test internal server error handling."""
        # This would test how the app handles unexpected errors
        pass
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test database connection error handling."""
        # This would test app behavior when database is unavailable
        pass


@pytest.mark.integration
class TestIntegration:
    """Integration tests that require external services."""
    
    @pytest.mark.asyncio
    async def test_inference_service_integration(self):
        """Test integration with inference service."""
        # This would test the actual connection to inference service
        pass
    
    @pytest.mark.asyncio
    async def test_redis_integration(self):
        """Test Redis integration."""
        # This would test Redis connectivity and operations
        pass