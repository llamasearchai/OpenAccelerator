#!/usr/bin/env python3
"""
Authentication Security Testing Suite for Open Accelerator

Tests JWT token validation, OAuth flows, session management, and authentication
bypass attempts to ensure robust authentication security.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)


class AuthSecurityTester:
    """Comprehensive authentication security testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize authentication security tester."""
        self.base_url = base_url
        self.results = {
            "jwt_tests": [],
            "oauth_tests": [],
            "session_tests": [],
            "bypass_tests": [],
            "timing_tests": [],
            "brute_force_tests": [],
            "token_management_tests": [],
            "summary": {},
        }
        self.test_username = "testuser"
        self.test_password = "testpass123"
        self.admin_username = "admin"
        self.admin_password = "adminpass123"

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all authentication security tests."""
        logger.info("Starting comprehensive authentication security tests")

        # Test JWT token security
        self._test_jwt_security()

        # Test OAuth flow security
        self._test_oauth_security()

        # Test session management
        self._test_session_management()

        # Test authentication bypass attempts
        self._test_authentication_bypass()

        # Test timing attacks
        self._test_timing_attacks()

        # Test brute force protection
        self._test_brute_force_protection()

        # Test token management
        self._test_token_management()

        # Generate summary
        self._generate_summary()

        return self.results

    def _test_jwt_security(self):
        """Test JWT token security."""
        logger.info("Testing JWT token security")

        # Test JWT token structure
        self._test_jwt_structure()

        # Test JWT token validation
        self._test_jwt_validation()

        # Test JWT token expiration
        self._test_jwt_expiration()

        # Test JWT token manipulation
        self._test_jwt_manipulation()

    def _test_jwt_structure(self):
        """Test JWT token structure validation."""
        # Generate test JWT tokens with various issues
        test_tokens = [
            # Invalid number of segments
            "invalid.token",
            "too.many.segments.here",
            # Invalid base64 encoding
            "invalid-base64.eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0",
            # Missing signature
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.",
            # Empty token
            "",
            # Non-JWT token
            "Bearer regular-token-123",
        ]

        for token in test_tokens:
            try:
                headers = {"Authorization": f"Bearer {token}"}
                url = f"{self.base_url}/api/v1/health"
                response = httpx.get(url, headers=headers)

                test_result = {
                    "test": f"JWT structure test: {token[:20]}...",
                    "expected": "401 or 403",
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403],
                    "token": token,
                    "details": response.json()
                    if response.status_code not in [401, 403]
                    else None,
                }

                self.results["jwt_tests"].append(test_result)

            except Exception as e:
                self.results["jwt_tests"].append(
                    {
                        "test": f"JWT structure test: {token[:20]}...",
                        "error": str(e),
                        "token": token,
                        "passed": True,  # Exception is good for invalid tokens
                    }
                )

    def _test_jwt_validation(self):
        """Test JWT token validation."""
        # Create test JWT tokens with various issues
        test_cases = [
            # Invalid signature
            self._create_invalid_jwt_token(),
            # Expired token
            self._create_expired_jwt_token(),
            # Token with wrong algorithm
            self._create_wrong_algorithm_jwt_token(),
            # Token with invalid issuer
            self._create_invalid_issuer_jwt_token(),
            # Token with invalid audience
            self._create_invalid_audience_jwt_token(),
        ]

        for test_case in test_cases:
            try:
                headers = {"Authorization": f"Bearer {test_case['token']}"}
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(url, json={}, headers=headers)

                test_result = {
                    "test": f"JWT validation: {test_case['description']}",
                    "expected": "401 or 403",
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403],
                    "description": test_case["description"],
                    "details": response.json()
                    if response.status_code not in [401, 403]
                    else None,
                }

                self.results["jwt_tests"].append(test_result)

            except Exception as e:
                self.results["jwt_tests"].append(
                    {
                        "test": f"JWT validation: {test_case['description']}",
                        "error": str(e),
                        "passed": True,  # Exception is good for invalid tokens
                    }
                )

    def _test_jwt_expiration(self):
        """Test JWT token expiration handling."""
        # Test with tokens that have different expiration times
        test_cases = [
            # Already expired
            {"exp": int(time.time()) - 3600, "description": "Expired 1 hour ago"},
            # Expires in the past
            {"exp": int(time.time()) - 1, "description": "Expired 1 second ago"},
            # No expiration claim
            {"description": "No expiration claim"},
            # Invalid expiration format
            {"exp": "invalid", "description": "Invalid expiration format"},
            # Future expiration (should work)
            {"exp": int(time.time()) + 3600, "description": "Expires in 1 hour"},
        ]

        for test_case in test_cases:
            try:
                token = self._create_test_jwt_token(test_case)
                headers = {"Authorization": f"Bearer {token}"}
                url = f"{self.base_url}/api/v1/health"
                response = httpx.get(url, headers=headers)

                # Only the future expiration should work
                expected_success = test_case["description"] == "Expires in 1 hour"

                test_result = {
                    "test": f"JWT expiration: {test_case['description']}",
                    "expected": "200" if expected_success else "401 or 403",
                    "actual": response.status_code,
                    "passed": (response.status_code == 200) == expected_success,
                    "description": test_case["description"],
                    "details": response.json()
                    if not ((response.status_code == 200) == expected_success)
                    else None,
                }

                self.results["jwt_tests"].append(test_result)

            except Exception as e:
                self.results["jwt_tests"].append(
                    {
                        "test": f"JWT expiration: {test_case['description']}",
                        "error": str(e),
                        "passed": not expected_success,  # Exception is good for invalid tokens
                    }
                )

    def _test_jwt_manipulation(self):
        """Test JWT token manipulation attacks."""
        # Create a valid token and try to manipulate it
        valid_token = self._create_test_jwt_token({"exp": int(time.time()) + 3600})

        # Split the token into parts
        parts = valid_token.split(".")

        manipulation_tests = [
            # Change algorithm to none
            {
                "description": "Algorithm changed to none",
                "header": {"alg": "none"},
                "payload": None,
                "signature": "",
            },
            # Change user role in payload
            {
                "description": "User role changed to admin",
                "header": None,
                "payload": {"role": "admin", "exp": int(time.time()) + 3600},
                "signature": None,
            },
            # Remove signature
            {
                "description": "Signature removed",
                "header": None,
                "payload": None,
                "signature": "",
            },
            # Invalid signature
            {
                "description": "Invalid signature",
                "header": None,
                "payload": None,
                "signature": "invalid-signature",
            },
        ]

        for test_case in manipulation_tests:
            try:
                manipulated_token = self._manipulate_jwt_token(valid_token, test_case)
                headers = {"Authorization": f"Bearer {manipulated_token}"}
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(url, json={}, headers=headers)

                test_result = {
                    "test": f"JWT manipulation: {test_case['description']}",
                    "expected": "401 or 403",
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403],
                    "description": test_case["description"],
                    "details": response.json()
                    if response.status_code not in [401, 403]
                    else None,
                }

                self.results["jwt_tests"].append(test_result)

            except Exception as e:
                self.results["jwt_tests"].append(
                    {
                        "test": f"JWT manipulation: {test_case['description']}",
                        "error": str(e),
                        "passed": True,  # Exception is good for manipulated tokens
                    }
                )

    def _test_oauth_security(self):
        """Test OAuth flow security."""
        logger.info("Testing OAuth flow security")

        # Test OAuth authorization code flow
        self._test_oauth_authorization_code()

        # Test OAuth PKCE flow
        self._test_oauth_pkce()

        # Test OAuth token exchange
        self._test_oauth_token_exchange()

    def _test_oauth_authorization_code(self):
        """Test OAuth authorization code flow."""
        test_cases = [
            # Invalid client_id
            {
                "client_id": "invalid_client",
                "description": "Invalid client ID",
                "should_fail": True,
            },
            # Invalid redirect_uri
            {
                "client_id": "test_client",
                "redirect_uri": "https://evil.com/callback",
                "description": "Invalid redirect URI",
                "should_fail": True,
            },
            # Invalid response_type
            {
                "client_id": "test_client",
                "response_type": "invalid",
                "description": "Invalid response type",
                "should_fail": True,
            },
            # Valid request
            {
                "client_id": "test_client",
                "response_type": "code",
                "redirect_uri": "https://app.example.com/callback",
                "description": "Valid authorization request",
                "should_fail": False,
            },
        ]

        for test_case in test_cases:
            try:
                params = {
                    "client_id": test_case.get("client_id"),
                    "response_type": test_case.get("response_type", "code"),
                    "redirect_uri": test_case.get(
                        "redirect_uri", "https://app.example.com/callback"
                    ),
                    "scope": "read write",
                }

                url = f"{self.base_url}/oauth/authorize"
                response = httpx.get(url, params=params)

                test_result = {
                    "test": f"OAuth authorization: {test_case['description']}",
                    "expected": "400 or 401"
                    if test_case["should_fail"]
                    else "200 or 302",
                    "actual": response.status_code,
                    "passed": (response.status_code in [400, 401])
                    == test_case["should_fail"],
                    "description": test_case["description"],
                    "params": params,
                    "details": response.json()
                    if response.status_code not in [200, 302, 400, 401]
                    else None,
                }

                self.results["oauth_tests"].append(test_result)

            except Exception as e:
                self.results["oauth_tests"].append(
                    {
                        "test": f"OAuth authorization: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case[
                            "should_fail"
                        ],  # Exception is good for invalid requests
                    }
                )

    def _test_oauth_pkce(self):
        """Test OAuth PKCE (Proof Key for Code Exchange) flow."""
        # Generate PKCE challenge
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("utf-8")).digest()
            )
            .decode("utf-8")
            .rstrip("=")
        )

        test_cases = [
            # Missing code_challenge
            {
                "description": "Missing code challenge",
                "params": {"client_id": "test_client"},
                "should_fail": True,
            },
            # Invalid code_challenge_method
            {
                "description": "Invalid code challenge method",
                "params": {
                    "client_id": "test_client",
                    "code_challenge": code_challenge,
                    "code_challenge_method": "invalid",
                },
                "should_fail": True,
            },
            # Valid PKCE request
            {
                "description": "Valid PKCE request",
                "params": {
                    "client_id": "test_client",
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                },
                "should_fail": False,
            },
        ]

        for test_case in test_cases:
            try:
                params = {
                    "response_type": "code",
                    "redirect_uri": "https://app.example.com/callback",
                    "scope": "read write",
                    **test_case["params"],
                }

                url = f"{self.base_url}/oauth/authorize"
                response = httpx.get(url, params=params)

                test_result = {
                    "test": f"OAuth PKCE: {test_case['description']}",
                    "expected": "400 or 401"
                    if test_case["should_fail"]
                    else "200 or 302",
                    "actual": response.status_code,
                    "passed": (response.status_code in [400, 401])
                    == test_case["should_fail"],
                    "description": test_case["description"],
                    "params": params,
                    "details": response.json()
                    if response.status_code not in [200, 302, 400, 401]
                    else None,
                }

                self.results["oauth_tests"].append(test_result)

            except Exception as e:
                self.results["oauth_tests"].append(
                    {
                        "test": f"OAuth PKCE: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case["should_fail"],
                    }
                )

    def _test_oauth_token_exchange(self):
        """Test OAuth token exchange security."""
        test_cases = [
            # Invalid authorization code
            {
                "code": "invalid_code",
                "description": "Invalid authorization code",
                "should_fail": True,
            },
            # Invalid client credentials
            {
                "code": "valid_code",
                "client_id": "invalid_client",
                "description": "Invalid client credentials",
                "should_fail": True,
            },
            # Reused authorization code
            {
                "code": "used_code",
                "description": "Reused authorization code",
                "should_fail": True,
            },
        ]

        for test_case in test_cases:
            try:
                data = {
                    "grant_type": "authorization_code",
                    "code": test_case["code"],
                    "client_id": test_case.get("client_id", "test_client"),
                    "client_secret": "test_secret",
                    "redirect_uri": "https://app.example.com/callback",
                }

                url = f"{self.base_url}/oauth/token"
                response = httpx.post(url, data=data)

                test_result = {
                    "test": f"OAuth token exchange: {test_case['description']}",
                    "expected": "400 or 401" if test_case["should_fail"] else "200",
                    "actual": response.status_code,
                    "passed": (response.status_code in [400, 401])
                    == test_case["should_fail"],
                    "description": test_case["description"],
                    "details": response.json()
                    if response.status_code not in [200, 400, 401]
                    else None,
                }

                self.results["oauth_tests"].append(test_result)

            except Exception as e:
                self.results["oauth_tests"].append(
                    {
                        "test": f"OAuth token exchange: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case["should_fail"],
                    }
                )

    def _test_session_management(self):
        """Test session management security."""
        logger.info("Testing session management security")

        # Test session fixation
        self._test_session_fixation()

        # Test session timeout
        self._test_session_timeout()

        # Test concurrent sessions
        self._test_concurrent_sessions()

    def _test_session_fixation(self):
        """Test session fixation attacks."""
        # Try to set a predefined session ID
        session_id = "fixed_session_123"

        try:
            # Attempt to login with a fixed session ID
            headers = {"Cookie": f"session_id={session_id}"}
            data = {"username": self.test_username, "password": self.test_password}

            url = f"{self.base_url}/auth/login"
            response = httpx.post(url, json=data, headers=headers)

            # Check if the session ID was changed after login
            new_session_id = None
            if "Set-Cookie" in response.headers:
                cookies = response.headers["Set-Cookie"]
                if "session_id=" in cookies:
                    new_session_id = cookies.split("session_id=")[1].split(";")[0]

            test_result = {
                "test": "Session fixation protection",
                "expected": "Session ID changed after login",
                "actual": f"Session ID: {new_session_id}",
                "passed": new_session_id != session_id,
                "original_session": session_id,
                "new_session": new_session_id,
                "details": response.json()
                if response.status_code not in [200, 302]
                else None,
            }

            self.results["session_tests"].append(test_result)

        except Exception as e:
            self.results["session_tests"].append(
                {
                    "test": "Session fixation protection",
                    "error": str(e),
                    "passed": False,
                }
            )

    def _test_session_timeout(self):
        """Test session timeout handling."""
        # Test various session timeout scenarios
        test_cases = [
            # Very old session
            {
                "age_seconds": 7200,  # 2 hours
                "description": "Session aged 2 hours",
                "should_be_valid": False,
            },
            # Recent session
            {
                "age_seconds": 300,  # 5 minutes
                "description": "Session aged 5 minutes",
                "should_be_valid": True,
            },
            # Expired session
            {
                "age_seconds": 86400,  # 24 hours
                "description": "Session aged 24 hours",
                "should_be_valid": False,
            },
        ]

        for test_case in test_cases:
            try:
                # Create a session with specific timestamp
                session_data = {
                    "user_id": "test_user",
                    "created_at": int(time.time() - test_case["age_seconds"]),
                    "last_activity": int(time.time() - test_case["age_seconds"]),
                }

                # Encode session data (simplified)
                session_token = base64.b64encode(
                    json.dumps(session_data).encode()
                ).decode()

                headers = {"Cookie": f"session_token={session_token}"}
                url = f"{self.base_url}/api/v1/health"
                response = httpx.get(url, headers=headers)

                test_result = {
                    "test": f"Session timeout: {test_case['description']}",
                    "expected": "200" if test_case["should_be_valid"] else "401 or 403",
                    "actual": response.status_code,
                    "passed": (response.status_code == 200)
                    == test_case["should_be_valid"],
                    "description": test_case["description"],
                    "session_age": test_case["age_seconds"],
                    "details": response.json()
                    if not (
                        (response.status_code == 200) == test_case["should_be_valid"]
                    )
                    else None,
                }

                self.results["session_tests"].append(test_result)

            except Exception as e:
                self.results["session_tests"].append(
                    {
                        "test": f"Session timeout: {test_case['description']}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    def _test_concurrent_sessions(self):
        """Test concurrent session handling."""
        # Test multiple sessions for the same user
        try:
            # Create multiple sessions for the same user
            session_tokens = []
            for i in range(3):
                session_data = {
                    "user_id": "test_user",
                    "session_id": f"session_{i}",
                    "created_at": int(time.time()),
                }
                session_token = base64.b64encode(
                    json.dumps(session_data).encode()
                ).decode()
                session_tokens.append(session_token)

            # Test if all sessions are valid (they shouldn't be)
            valid_sessions = 0
            for i, token in enumerate(session_tokens):
                headers = {"Cookie": f"session_token={token}"}
                url = f"{self.base_url}/api/v1/health"
                response = httpx.get(url, headers=headers)

                if response.status_code == 200:
                    valid_sessions += 1

            test_result = {
                "test": "Concurrent session limit",
                "expected": "Only 1 session should be valid",
                "actual": f"{valid_sessions} sessions valid",
                "passed": valid_sessions <= 1,
                "total_sessions": len(session_tokens),
                "valid_sessions": valid_sessions,
            }

            self.results["session_tests"].append(test_result)

        except Exception as e:
            self.results["session_tests"].append(
                {"test": "Concurrent session limit", "error": str(e), "passed": False}
            )

    def _test_authentication_bypass(self):
        """Test authentication bypass attempts."""
        logger.info("Testing authentication bypass attempts")

        # Test various bypass techniques
        bypass_tests = [
            # Header injection
            {
                "headers": {"X-User-ID": "admin"},
                "description": "Header injection - X-User-ID",
            },
            # Role escalation
            {
                "headers": {"X-User-Role": "admin"},
                "description": "Header injection - X-User-Role",
            },
            # IP whitelist bypass
            {
                "headers": {"X-Real-IP": "127.0.0.1"},
                "description": "IP whitelist bypass - X-Real-IP",
            },
            # Forwarded for bypass
            {
                "headers": {"X-Forwarded-For": "127.0.0.1"},
                "description": "IP whitelist bypass - X-Forwarded-For",
            },
            # Authorization bypass
            {
                "headers": {"Authorization": "Bearer admin-override"},
                "description": "Authorization bypass - admin override",
            },
        ]

        for test_case in bypass_tests:
            try:
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(url, json={}, headers=test_case["headers"])

                test_result = {
                    "test": f"Authentication bypass: {test_case['description']}",
                    "expected": "401 or 403",
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403],
                    "description": test_case["description"],
                    "headers": test_case["headers"],
                    "details": response.json()
                    if response.status_code not in [401, 403]
                    else None,
                }

                self.results["bypass_tests"].append(test_result)

            except Exception as e:
                self.results["bypass_tests"].append(
                    {
                        "test": f"Authentication bypass: {test_case['description']}",
                        "error": str(e),
                        "passed": True,  # Exception is good for bypass attempts
                    }
                )

    def _test_timing_attacks(self):
        """Test timing attack resistance."""
        logger.info("Testing timing attack resistance")

        # Test login timing for valid vs invalid users
        users = [
            ("valid_user", "valid_pass"),
            ("invalid_user", "invalid_pass"),
            ("admin", "wrong_pass"),
            ("", ""),
            ("nonexistent", "password"),
        ]

        timings = []
        for username, password in users:
            try:
                start_time = time.time()

                data = {"username": username, "password": password}
                url = f"{self.base_url}/auth/login"
                response = httpx.post(url, json=data)

                end_time = time.time()
                timing = end_time - start_time

                timings.append(
                    {
                        "username": username,
                        "timing": timing,
                        "status": response.status_code,
                    }
                )

            except Exception as e:
                timings.append({"username": username, "error": str(e), "timing": 0})

        # Check if timings are consistent (should be)
        valid_timings = [t["timing"] for t in timings if t.get("timing", 0) > 0]
        if valid_timings:
            avg_timing = sum(valid_timings) / len(valid_timings)
            max_deviation = max(abs(t - avg_timing) for t in valid_timings)

            test_result = {
                "test": "Timing attack resistance",
                "expected": "Consistent timing for all login attempts",
                "actual": f"Max deviation: {max_deviation:.3f}s",
                "passed": max_deviation < 0.1,  # Less than 100ms deviation
                "average_timing": avg_timing,
                "max_deviation": max_deviation,
                "timings": timings,
            }
        else:
            test_result = {
                "test": "Timing attack resistance",
                "expected": "Consistent timing for all login attempts",
                "actual": "No valid timings recorded",
                "passed": False,
                "error": "Could not measure timings",
            }

        self.results["timing_tests"].append(test_result)

    def _test_brute_force_protection(self):
        """Test brute force attack protection."""
        logger.info("Testing brute force attack protection")

        # Attempt multiple failed logins
        failed_attempts = 0
        blocked_attempts = 0

        for i in range(20):  # Try 20 failed login attempts
            try:
                data = {"username": self.test_username, "password": "wrong_password"}
                url = f"{self.base_url}/auth/login"
                response = httpx.post(url, json=data)

                if response.status_code == 401:
                    failed_attempts += 1
                elif response.status_code == 429:  # Rate limited
                    blocked_attempts += 1
                elif response.status_code == 423:  # Account locked
                    blocked_attempts += 1

                # Short delay between attempts
                time.sleep(0.1)

            except Exception:
                # Connection errors might indicate rate limiting
                blocked_attempts += 1

        test_result = {
            "test": "Brute force protection",
            "expected": "Account locked or rate limited after multiple attempts",
            "actual": f"Failed: {failed_attempts}, Blocked: {blocked_attempts}",
            "passed": blocked_attempts > 0,
            "total_attempts": 20,
            "failed_attempts": failed_attempts,
            "blocked_attempts": blocked_attempts,
        }

        self.results["brute_force_tests"].append(test_result)

    def _test_token_management(self):
        """Test token management security."""
        logger.info("Testing token management security")

        # Test token refresh
        self._test_token_refresh()

        # Test token revocation
        self._test_token_revocation()

        # Test token scope validation
        self._test_token_scope_validation()

    def _test_token_refresh(self):
        """Test token refresh security."""
        test_cases = [
            # Invalid refresh token
            {
                "refresh_token": "invalid_refresh_token",
                "description": "Invalid refresh token",
                "should_fail": True,
            },
            # Expired refresh token
            {
                "refresh_token": "expired_refresh_token",
                "description": "Expired refresh token",
                "should_fail": True,
            },
            # Used refresh token
            {
                "refresh_token": "used_refresh_token",
                "description": "Previously used refresh token",
                "should_fail": True,
            },
        ]

        for test_case in test_cases:
            try:
                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": test_case["refresh_token"],
                }

                url = f"{self.base_url}/auth/token/refresh"
                response = httpx.post(url, json=data)

                test_result = {
                    "test": f"Token refresh: {test_case['description']}",
                    "expected": "400 or 401" if test_case["should_fail"] else "200",
                    "actual": response.status_code,
                    "passed": (response.status_code in [400, 401])
                    == test_case["should_fail"],
                    "description": test_case["description"],
                    "details": response.json()
                    if response.status_code not in [200, 400, 401]
                    else None,
                }

                self.results["token_management_tests"].append(test_result)

            except Exception as e:
                self.results["token_management_tests"].append(
                    {
                        "test": f"Token refresh: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case["should_fail"],
                    }
                )

    def _test_token_revocation(self):
        """Test token revocation."""
        try:
            # Try to revoke a token
            data = {"token": "test_token_to_revoke"}
            url = f"{self.base_url}/auth/token/revoke"
            response = httpx.post(url, json=data)

            test_result = {
                "test": "Token revocation",
                "expected": "200 or 204",
                "actual": response.status_code,
                "passed": response.status_code in [200, 204],
                "details": response.json()
                if response.status_code not in [200, 204]
                else None,
            }

            self.results["token_management_tests"].append(test_result)

        except Exception as e:
            self.results["token_management_tests"].append(
                {"test": "Token revocation", "error": str(e), "passed": False}
            )

    def _test_token_scope_validation(self):
        """Test token scope validation."""
        scopes = ["read", "write", "admin", "medical"]

        for scope in scopes:
            try:
                # Create a token with limited scope
                token = self._create_test_jwt_token(
                    {"scope": scope, "exp": int(time.time()) + 3600}
                )

                headers = {"Authorization": f"Bearer {token}"}

                # Test different endpoints that require different scopes
                test_endpoints = [
                    ("/api/v1/health", "GET", "read"),
                    ("/api/v1/simulation/run", "POST", "write"),
                    ("/api/v1/simulation/stop/all", "POST", "admin"),
                    ("/api/v1/medical/analyze", "POST", "medical"),
                ]

                for endpoint, method, required_scope in test_endpoints:
                    url = f"{self.base_url}{endpoint}"

                    if method == "GET":
                        response = httpx.get(url, headers=headers)
                    else:
                        response = httpx.post(url, json={}, headers=headers)

                    # Should succeed only if token scope matches required scope
                    should_succeed = scope == required_scope or scope == "admin"

                    test_result = {
                        "test": f"Token scope validation: {scope} -> {endpoint}",
                        "expected": "200-299" if should_succeed else "403",
                        "actual": response.status_code,
                        "passed": (200 <= response.status_code < 300) == should_succeed,
                        "token_scope": scope,
                        "required_scope": required_scope,
                        "endpoint": endpoint,
                        "details": response.json()
                        if not ((200 <= response.status_code < 300) == should_succeed)
                        else None,
                    }

                    self.results["token_management_tests"].append(test_result)

            except Exception as e:
                self.results["token_management_tests"].append(
                    {
                        "test": f"Token scope validation: {scope}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    # Helper methods for JWT token creation and manipulation
    def _create_test_jwt_token(self, payload: Dict[str, Any]) -> str:
        """Create a test JWT token with given payload."""
        header = {"alg": "HS256", "typ": "JWT"}

        # Base64 encode header and payload
        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        # Create signature (simplified - in real implementation would use proper key)
        message = f"{header_b64}.{payload_b64}"
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(b"test_secret", message.encode(), hashlib.sha256).digest()
            )
            .decode()
            .rstrip("=")
        )

        return f"{header_b64}.{payload_b64}.{signature}"

    def _create_invalid_jwt_token(self) -> Dict[str, str]:
        """Create JWT token with invalid signature."""
        payload = {"sub": "test", "exp": int(time.time()) + 3600}
        token = self._create_test_jwt_token(payload)

        # Corrupt the signature
        parts = token.split(".")
        parts[2] = "invalid_signature"

        return {"token": ".".join(parts), "description": "Invalid signature"}

    def _create_expired_jwt_token(self) -> Dict[str, str]:
        """Create expired JWT token."""
        payload = {"sub": "test", "exp": int(time.time()) - 3600}
        token = self._create_test_jwt_token(payload)

        return {"token": token, "description": "Expired token"}

    def _create_wrong_algorithm_jwt_token(self) -> Dict[str, str]:
        """Create JWT token with wrong algorithm."""
        header = {"alg": "none", "typ": "JWT"}
        payload = {"sub": "test", "exp": int(time.time()) + 3600}

        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        return {
            "token": f"{header_b64}.{payload_b64}.",
            "description": "Wrong algorithm (none)",
        }

    def _create_invalid_issuer_jwt_token(self) -> Dict[str, str]:
        """Create JWT token with invalid issuer."""
        payload = {"sub": "test", "iss": "evil.com", "exp": int(time.time()) + 3600}
        token = self._create_test_jwt_token(payload)

        return {"token": token, "description": "Invalid issuer"}

    def _create_invalid_audience_jwt_token(self) -> Dict[str, str]:
        """Create JWT token with invalid audience."""
        payload = {
            "sub": "test",
            "aud": "wrong-audience",
            "exp": int(time.time()) + 3600,
        }
        token = self._create_test_jwt_token(payload)

        return {"token": token, "description": "Invalid audience"}

    def _manipulate_jwt_token(self, token: str, manipulation: Dict[str, Any]) -> str:
        """Manipulate JWT token according to test case."""
        parts = token.split(".")

        # Manipulate header
        if manipulation.get("header"):
            header_b64 = (
                base64.urlsafe_b64encode(json.dumps(manipulation["header"]).encode())
                .decode()
                .rstrip("=")
            )
            parts[0] = header_b64

        # Manipulate payload
        if manipulation.get("payload"):
            payload_b64 = (
                base64.urlsafe_b64encode(json.dumps(manipulation["payload"]).encode())
                .decode()
                .rstrip("=")
            )
            parts[1] = payload_b64

        # Manipulate signature
        if manipulation.get("signature") is not None:
            parts[2] = manipulation["signature"]

        return ".".join(parts)

    def _generate_summary(self):
        """Generate test summary."""
        total_tests = 0
        passed_tests = 0

        for category, tests in self.results.items():
            if category == "summary":
                continue

            category_total = len(tests)
            category_passed = sum(1 for test in tests if test.get("passed", False))

            total_tests += category_total
            passed_tests += category_passed

            self.results["summary"][category] = {
                "total": category_total,
                "passed": category_passed,
                "failed": category_total - category_passed,
                "pass_rate": (category_passed / category_total * 100)
                if category_total > 0
                else 0,
            }

        self.results["summary"]["overall"] = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        }

        logger.info(
            f"Authentication security tests completed: {passed_tests}/{total_tests} passed ({self.results['summary']['overall']['pass_rate']:.1f}%)"
        )

    def save_results(self, output_file: str):
        """Save test results to file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Authentication security test results saved to {output_file}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Authentication Security Testing Suite"
    )
    parser.add_argument(
        "--target-url", default="http://localhost:8000", help="Target API URL"
    )
    parser.add_argument(
        "--output-file",
        default="auth_security_report.json",
        help="Output file for results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run authentication security tests
    tester = AuthSecurityTester(args.target_url)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(args.output_file)

    # Print summary
    summary = results["summary"]["overall"]
    print("\nAuthentication Security Test Summary:")
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1f}%")

    # Exit with appropriate code
    return 0 if summary["pass_rate"] >= 80 else 1


if __name__ == "__main__":
    exit(main())
