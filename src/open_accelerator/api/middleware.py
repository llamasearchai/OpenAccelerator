"""
FastAPI middleware for security, rate limiting, and medical compliance.

Provides comprehensive middleware stack for production-ready API deployment
with proper security controls and audit logging.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from contextvars import ContextVar
from uuid import uuid4

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id')
user_context_var: ContextVar[Dict[str, Any]] = ContextVar('user_context')

# Global health metrics
app_health_metrics = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0,
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for API protection."""
    
    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key
        self.security_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks."""
        # Add request ID
        request_id = str(uuid4())
        request_id_var.set(request_id)
        
        # Check API key for protected endpoints
        if self.api_key and request.url.path.startswith('/api/v1/'):
            if request.url.path not in ['/api/v1/health', '/api/v1/docs', '/api/v1/redoc']:
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"error": "Missing or invalid authorization header"},
                        headers={"X-Request-ID": request_id}
                    )
                
                token = auth_header.split(' ', 1)[1]
                if token != self.api_key:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"error": "Invalid API key"},
                        headers={"X-Request-ID": request_id}
                    )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response


class MedicalComplianceMiddleware(BaseHTTPMiddleware):
    """Medical compliance middleware for HIPAA and FDA compliance."""
    
    def __init__(self, app, enable_audit_logging: bool = True):
        super().__init__(app)
        self.enable_audit_logging = enable_audit_logging
        self.audit_logger = logging.getLogger("medical_audit")
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with medical compliance checks."""
        start_time = time.time()
        
        # Extract user context (would come from authentication)
        user_context = {
            "user_id": request.headers.get("X-User-ID", "anonymous"),
            "organization": request.headers.get("X-Organization", "unknown"),
            "role": request.headers.get("X-User-Role", "user")
        }
        user_context_var.set(user_context)
        
        # Get client host safely
        client_host = request.client.host if request.client else "unknown"
        
        # Audit log for medical endpoints
        if request.url.path.startswith('/api/v1/medical/'):
            self.audit_logger.info(
                f"Medical endpoint access: {request.method} {request.url.path} "
                f"by user {user_context['user_id']} from {client_host}"
            )
        
        # Process request
        response = await call_next(request)
        
        # Record processing time for compliance
        processing_time = time.time() - start_time
        
        # Add compliance headers
        response.headers["X-Processing-Time"] = str(processing_time)
        response.headers["X-Compliance-Level"] = "HIPAA"
        
        # Audit log completion
        if self.enable_audit_logging and request.url.path.startswith('/api/v1/medical/'):
            self.audit_logger.info(
                f"Medical endpoint completed: {request.method} {request.url.path} "
                f"status={response.status_code} time={processing_time:.3f}s"
            )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, List[float]] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                timestamp for timestamp in self.request_counts[client_ip]
                if now - timestamp < 60.0  # 1 minute window
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.request_counts[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.request_counts[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(now) + 60)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for API requests and responses."""
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
        self.access_logger = logging.getLogger("api_access")
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Log request
        user_agent = request.headers.get("user-agent", "unknown")
        request_size = request.headers.get("content-length", "0")
        client_host = request.client.host if request.client else "unknown"
        
        self.access_logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_host} (size={request_size}, agent={user_agent})"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        processing_time = time.time() - start_time
        response_size = response.headers.get("content-length", "0")
        
        self.access_logger.info(
            f"Response: {response.status_code} (size={response_size}, time={processing_time:.3f}s)"
        )
        
        return response
    
    async def log_request(self, request: Request):
        """Log request details."""
        client_host = request.client.host if request.client else "unknown"
        self.access_logger.info(
            f"Request: {request.method} {request.url.path} from {client_host}"
        )
    
    async def log_response(self, request: Request, response: Response):
        """Log response details."""
        self.access_logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path}"
        )


class RequestLoggingMiddleware(LoggingMiddleware):
    """Request logging middleware (alias for LoggingMiddleware)."""
    pass


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check middleware for monitoring."""
    
    def __init__(self, app):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track health metrics."""
        app_health_metrics["request_count"] += 1
        
        try:
            response = await call_next(request)
            
            # Track errors
            if response.status_code >= 400:
                app_health_metrics["error_count"] += 1
                
            return response
            
        except Exception as e:
            app_health_metrics["error_count"] += 1
            logger.error(f"Request failed: {e}")
            raise


def get_health_metrics() -> Dict[str, Any]:
    """Get current health metrics."""
    uptime = time.time() - app_health_metrics["start_time"]
    error_rate = app_health_metrics["error_count"] / max(app_health_metrics["request_count"], 1)
    
    return {
        "uptime_seconds": uptime,
        "total_requests": app_health_metrics["request_count"],
        "error_count": app_health_metrics["error_count"],
        "error_rate": error_rate,
        "status": "healthy" if error_rate < 0.1 else "degraded"
    }


def setup_middleware(app: FastAPI, config: Optional[Dict[str, Any]] = None) -> None:
    """Setup all middleware with configuration."""
    if config is None:
        config = {}
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("allowed_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware
    if config.get("trusted_hosts"):
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=config["trusted_hosts"]
        )
    
    # Security middleware
    app.add_middleware(
        SecurityMiddleware,
        api_key=config.get("api_key")
    )
    
    # Medical compliance middleware
    app.add_middleware(
        MedicalComplianceMiddleware,
        enable_audit_logging=config.get("enable_audit_logging", True)
    )
    
    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.get("rate_limit", 100)
    )
    
    # Request logging middleware
    app.add_middleware(
        LoggingMiddleware,
        log_body=config.get("log_request_body", False)
    )
    
    # Health check middleware
    app.add_middleware(HealthCheckMiddleware)
    
    # Store health metrics accessor
    app.state.get_health_metrics = get_health_metrics
    
    logger.info("All middleware configured successfully") 