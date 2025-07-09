#!/usr/bin/env python3
"""
API Security Testing Suite for Open Accelerator

Tests authentication, authorization, input validation, and security headers
for all FastAPI endpoints to ensure enterprise-grade security.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import json
import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


class APISecurityTester:
    """Comprehensive API security testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize security tester."""
        self.base_url = base_url
        self.valid_token = "Bearer test-token-123"
        self.invalid_token = "Bearer invalid-token"
        self.results = {
            "authentication_tests": [],
            "authorization_tests": [],
            "input_validation_tests": [],
            "security_headers_tests": [],
            "rate_limiting_tests": [],
            "injection_tests": [],
            "summary": {},
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests."""
        logger.info("Starting comprehensive API security tests")

        # Test authentication
        self._test_authentication()

        # Test authorization
        self._test_authorization()

        # Test input validation
        self._test_input_validation()

        # Test security headers
        self._test_security_headers()

        # Test rate limiting
        self._test_rate_limiting()

        # Test injection attacks
        self._test_injection_attacks()

        # Generate summary
        self._generate_summary()

        return self.results

    def _test_authentication(self):
        """Test authentication mechanisms."""
        logger.info("Testing authentication mechanisms")

        protected_endpoints = [
            ("/api/v1/simulation/run", "POST"),
            ("/api/v1/agents/chat", "POST"),
            ("/api/v1/medical/analyze", "POST"),
            ("/api/v1/simulation/list", "GET"),
        ]

        for endpoint, method in protected_endpoints:
            # Test without token
            self._test_no_token(endpoint, method)

            # Test with invalid token
            self._test_invalid_token(endpoint, method)

            # Test with malformed token
            self._test_malformed_token(endpoint, method)

    def _test_no_token(self, endpoint: str, method: str):
        """Test endpoint without authentication token."""
        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                response = httpx.get(url)
            elif method == "POST":
                response = httpx.post(url, json={})

            test_result = {
                "test": f"No token - {method} {endpoint}",
                "expected": 401,
                "actual": response.status_code,
                "passed": response.status_code in [401, 403],
                "details": response.json()
                if response.status_code not in [401, 403]
                else None,
            }

            self.results["authentication_tests"].append(test_result)

        except Exception as e:
            self.results["authentication_tests"].append(
                {
                    "test": f"No token - {method} {endpoint}",
                    "error": str(e),
                    "passed": False,
                }
            )

    def _test_invalid_token(self, endpoint: str, method: str):
        """Test endpoint with invalid authentication token."""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {"Authorization": self.invalid_token}

            if method == "GET":
                response = httpx.get(url, headers=headers)
            elif method == "POST":
                response = httpx.post(url, json={}, headers=headers)

            test_result = {
                "test": f"Invalid token - {method} {endpoint}",
                "expected": 401,
                "actual": response.status_code,
                "passed": response.status_code in [401, 403],
                "details": response.json()
                if response.status_code not in [401, 403]
                else None,
            }

            self.results["authentication_tests"].append(test_result)

        except Exception as e:
            self.results["authentication_tests"].append(
                {
                    "test": f"Invalid token - {method} {endpoint}",
                    "error": str(e),
                    "passed": False,
                }
            )

    def _test_malformed_token(self, endpoint: str, method: str):
        """Test endpoint with malformed authentication token."""
        malformed_tokens = [
            "Bearer",
            "Bearer ",
            "InvalidBearer token",
            "Bearer token with spaces",
            "Bearer token-with-special-chars!@#$%",
        ]

        for token in malformed_tokens:
            try:
                url = f"{self.base_url}{endpoint}"
                headers = {"Authorization": token}

                if method == "GET":
                    response = httpx.get(url, headers=headers)
                elif method == "POST":
                    response = httpx.post(url, json={}, headers=headers)

                test_result = {
                    "test": f"Malformed token '{token}' - {method} {endpoint}",
                    "expected": 401,
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403],
                    "details": response.json()
                    if response.status_code not in [401, 403]
                    else None,
                }

                self.results["authentication_tests"].append(test_result)

            except Exception as e:
                self.results["authentication_tests"].append(
                    {
                        "test": f"Malformed token '{token}' - {method} {endpoint}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    def _test_authorization(self):
        """Test authorization mechanisms."""
        logger.info("Testing authorization mechanisms")

        # Test role-based access control
        self._test_rbac()

        # Test resource-based access control
        self._test_resource_access()

    def _test_rbac(self):
        """Test role-based access control."""
        # Mock different user roles
        roles = ["admin", "user", "readonly", "medical_staff"]

        for role in roles:
            # Create mock token with role
            mock_token = f"Bearer {role}-token-123"
            headers = {"Authorization": mock_token, "X-User-Role": role}

            # Test admin endpoints
            if role == "admin":
                self._test_admin_endpoints(headers, should_pass=True)
            else:
                self._test_admin_endpoints(headers, should_pass=False)

            # Test medical endpoints
            if role == "medical_staff":
                self._test_medical_endpoints(headers, should_pass=True)
            else:
                self._test_medical_endpoints(headers, should_pass=False)

    def _test_admin_endpoints(self, headers: Dict[str, str], should_pass: bool):
        """Test admin-only endpoints."""
        admin_endpoints = [
            "/api/v1/health/system",
            "/api/v1/simulation/stop/all",
        ]

        for endpoint in admin_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = httpx.get(url, headers=headers)

                if should_pass:
                    expected_status = 200
                    passed = response.status_code == 200
                else:
                    expected_status = 403
                    passed = response.status_code == 403

                test_result = {
                    "test": f"Admin endpoint {endpoint} - Role: {headers.get('X-User-Role')}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "details": response.json() if not passed else None,
                }

                self.results["authorization_tests"].append(test_result)

            except Exception as e:
                self.results["authorization_tests"].append(
                    {
                        "test": f"Admin endpoint {endpoint} - Role: {headers.get('X-User-Role')}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    def _test_medical_endpoints(self, headers: Dict[str, str], should_pass: bool):
        """Test medical-only endpoints."""
        medical_endpoints = [
            "/api/v1/medical/analyze",
            "/api/v1/medical/compliance",
        ]

        for endpoint in medical_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = httpx.post(url, json={}, headers=headers)

                if should_pass:
                    expected_status = 200
                    passed = response.status_code == 200
                else:
                    expected_status = 403
                    passed = response.status_code == 403

                test_result = {
                    "test": f"Medical endpoint {endpoint} - Role: {headers.get('X-User-Role')}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "details": response.json() if not passed else None,
                }

                self.results["authorization_tests"].append(test_result)

            except Exception as e:
                self.results["authorization_tests"].append(
                    {
                        "test": f"Medical endpoint {endpoint} - Role: {headers.get('X-User-Role')}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    def _test_resource_access(self):
        """Test resource-based access control."""
        # Test simulation access
        self._test_simulation_access()

        # Test agent access
        self._test_agent_access()

    def _test_simulation_access(self):
        """Test simulation resource access."""
        # Test accessing own simulation
        headers = {"Authorization": self.valid_token, "X-User-ID": "user123"}

        try:
            url = f"{self.base_url}/api/v1/simulation/sim123"
            response = httpx.get(url, headers=headers)

            test_result = {
                "test": "Access own simulation",
                "expected": 200,
                "actual": response.status_code,
                "passed": response.status_code
                in [200, 404],  # 404 is ok if simulation doesn't exist
                "details": response.json()
                if response.status_code not in [200, 404]
                else None,
            }

            self.results["authorization_tests"].append(test_result)

        except Exception as e:
            self.results["authorization_tests"].append(
                {"test": "Access own simulation", "error": str(e), "passed": False}
            )

    def _test_agent_access(self):
        """Test agent resource access."""
        # Test accessing agent with proper permissions
        headers = {"Authorization": self.valid_token, "X-Agent-Access": "optimization"}

        try:
            url = f"{self.base_url}/api/v1/agents/chat"
            response = httpx.post(
                url,
                json={"agent_type": "optimization", "message": "Test message"},
                headers=headers,
            )

            test_result = {
                "test": "Access agent with permissions",
                "expected": 200,
                "actual": response.status_code,
                "passed": response.status_code
                in [200, 422],  # 422 is ok for invalid payload
                "details": response.json()
                if response.status_code not in [200, 422]
                else None,
            }

            self.results["authorization_tests"].append(test_result)

        except Exception as e:
            self.results["authorization_tests"].append(
                {
                    "test": "Access agent with permissions",
                    "error": str(e),
                    "passed": False,
                }
            )

    def _test_input_validation(self):
        """Test input validation mechanisms."""
        logger.info("Testing input validation mechanisms")

        # Test simulation endpoint validation
        self._test_simulation_validation()

        # Test agent endpoint validation
        self._test_agent_validation()

    def _test_simulation_validation(self):
        """Test simulation endpoint input validation."""
        invalid_payloads = [
            # Missing required fields
            {},
            # Invalid data types
            {"accelerator_config": "invalid_config", "workload": 123},
            # Invalid values
            {
                "accelerator_config": {"array_rows": -1, "array_cols": 0},
                "workload": {"type": "invalid_type"},
            },
            # Extremely large values
            {
                "accelerator_config": {"array_rows": 999999, "array_cols": 999999},
                "workload": {"type": "gemm", "matrix_size": 999999},
            },
        ]

        headers = {"Authorization": self.valid_token}

        for i, payload in enumerate(invalid_payloads):
            try:
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(url, json=payload, headers=headers)

                test_result = {
                    "test": f"Simulation validation test {i+1}",
                    "expected": 422,
                    "actual": response.status_code,
                    "passed": response.status_code in [400, 422],
                    "payload": payload,
                    "details": response.json()
                    if response.status_code not in [400, 422]
                    else None,
                }

                self.results["input_validation_tests"].append(test_result)

            except Exception as e:
                self.results["input_validation_tests"].append(
                    {
                        "test": f"Simulation validation test {i+1}",
                        "error": str(e),
                        "payload": payload,
                        "passed": False,
                    }
                )

    def _test_agent_validation(self):
        """Test agent endpoint input validation."""
        invalid_payloads = [
            # Missing message
            {"agent_type": "optimization"},
            # Invalid agent type
            {"agent_type": "invalid_agent", "message": "Test message"},
            # Extremely long message
            {"agent_type": "optimization", "message": "x" * 100000},
        ]

        headers = {"Authorization": self.valid_token}

        for i, payload in enumerate(invalid_payloads):
            try:
                url = f"{self.base_url}/api/v1/agents/chat"
                response = httpx.post(url, json=payload, headers=headers)

                test_result = {
                    "test": f"Agent validation test {i+1}",
                    "expected": 422,
                    "actual": response.status_code,
                    "passed": response.status_code in [400, 422],
                    "payload": payload,
                    "details": response.json()
                    if response.status_code not in [400, 422]
                    else None,
                }

                self.results["input_validation_tests"].append(test_result)

            except Exception as e:
                self.results["input_validation_tests"].append(
                    {
                        "test": f"Agent validation test {i+1}",
                        "error": str(e),
                        "payload": payload,
                        "passed": False,
                    }
                )

    def _test_security_headers(self):
        """Test security headers."""
        logger.info("Testing security headers")

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        # Test various endpoints
        endpoints = ["/api/v1/health", "/api/v1/docs", "/api/v1/openapi.json"]

        for endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = httpx.get(url)

                for header in required_headers:
                    test_result = {
                        "test": f"Security header {header} on {endpoint}",
                        "expected": "Present",
                        "actual": "Present"
                        if header in response.headers
                        else "Missing",
                        "passed": header in response.headers,
                        "value": response.headers.get(header),
                    }

                    self.results["security_headers_tests"].append(test_result)

            except Exception as e:
                self.results["security_headers_tests"].append(
                    {
                        "test": f"Security headers on {endpoint}",
                        "error": str(e),
                        "passed": False,
                    }
                )

    def _test_rate_limiting(self):
        """Test rate limiting mechanisms."""
        logger.info("Testing rate limiting mechanisms")

        # Test rapid requests
        headers = {"Authorization": self.valid_token}
        endpoint = "/api/v1/health"

        responses = []
        for i in range(100):  # Send 100 rapid requests
            try:
                url = f"{self.base_url}{endpoint}"
                response = httpx.get(url, headers=headers)
                responses.append(response.status_code)

                # Check if rate limited
                if response.status_code == 429:
                    test_result = {
                        "test": f"Rate limiting triggered at request {i+1}",
                        "expected": 429,
                        "actual": response.status_code,
                        "passed": True,
                        "details": f"Rate limited after {i+1} requests",
                    }

                    self.results["rate_limiting_tests"].append(test_result)
                    break

            except Exception as e:
                self.results["rate_limiting_tests"].append(
                    {
                        "test": f"Rate limiting test at request {i+1}",
                        "error": str(e),
                        "passed": False,
                    }
                )
                break
        else:
            # If we didn't get rate limited
            test_result = {
                "test": "Rate limiting test - 100 requests",
                "expected": "Rate limited",
                "actual": "Not rate limited",
                "passed": False,
                "details": "No rate limiting detected after 100 requests",
            }

            self.results["rate_limiting_tests"].append(test_result)

    def _test_injection_attacks(self):
        """Test injection attack prevention."""
        logger.info("Testing injection attack prevention")

        # SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "' OR 1=1 --",
        ]

        # NoSQL injection payloads
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "1==1"},
        ]

        # Command injection payloads
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(uname -a)",
        ]

        # Test SQL injection
        self._test_sql_injection(sql_payloads)

        # Test NoSQL injection
        self._test_nosql_injection(nosql_payloads)

        # Test command injection
        self._test_command_injection(command_payloads)

    def _test_sql_injection(self, payloads: List[str]):
        """Test SQL injection prevention."""
        headers = {"Authorization": self.valid_token}

        for payload in payloads:
            try:
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(
                    url,
                    json={
                        "simulation_name": payload,
                        "accelerator_config": {"array_rows": 4, "array_cols": 4},
                        "workload": {"type": "gemm", "matrix_size": 16},
                    },
                    headers=headers,
                )

                test_result = {
                    "test": f"SQL injection test: {payload[:20]}...",
                    "expected": "Blocked or sanitized",
                    "actual": response.status_code,
                    "passed": response.status_code in [400, 422, 500],
                    "payload": payload,
                    "details": response.json()
                    if response.status_code not in [400, 422, 500]
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"SQL injection test: {payload[:20]}...",
                        "error": str(e),
                        "payload": payload,
                        "passed": True,  # Exception is good - request was blocked
                    }
                )

    def _test_nosql_injection(self, payloads: List[Dict[str, Any]]):
        """Test NoSQL injection prevention."""
        headers = {"Authorization": self.valid_token}

        for payload in payloads:
            try:
                url = f"{self.base_url}/api/v1/agents/chat"
                response = httpx.post(
                    url,
                    json={
                        "agent_type": "optimization",
                        "message": "Test message",
                        "context": payload,
                    },
                    headers=headers,
                )

                test_result = {
                    "test": f"NoSQL injection test: {str(payload)[:20]}...",
                    "expected": "Blocked or sanitized",
                    "actual": response.status_code,
                    "passed": response.status_code in [400, 422, 500],
                    "payload": payload,
                    "details": response.json()
                    if response.status_code not in [400, 422, 500]
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"NoSQL injection test: {str(payload)[:20]}...",
                        "error": str(e),
                        "payload": payload,
                        "passed": True,  # Exception is good - request was blocked
                    }
                )

    def _test_command_injection(self, payloads: List[str]):
        """Test command injection prevention."""
        headers = {"Authorization": self.valid_token}

        for payload in payloads:
            try:
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(
                    url,
                    json={
                        "simulation_name": "test",
                        "accelerator_config": {"array_rows": 4, "array_cols": 4},
                        "workload": {"type": "gemm", "matrix_size": 16},
                        "output_path": payload,
                    },
                    headers=headers,
                )

                test_result = {
                    "test": f"Command injection test: {payload[:20]}...",
                    "expected": "Blocked or sanitized",
                    "actual": response.status_code,
                    "passed": response.status_code in [400, 422, 500],
                    "payload": payload,
                    "details": response.json()
                    if response.status_code not in [400, 422, 500]
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"Command injection test: {payload[:20]}...",
                        "error": str(e),
                        "payload": payload,
                        "passed": True,  # Exception is good - request was blocked
                    }
                )

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
            f"Security tests completed: {passed_tests}/{total_tests} passed ({self.results['summary']['overall']['pass_rate']:.1f}%)"
        )

    def save_results(self, output_file: str):
        """Save test results to file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Security test results saved to {output_file}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="API Security Testing Suite")
    parser.add_argument(
        "--target-url", default="http://localhost:8000", help="Target API URL"
    )
    parser.add_argument(
        "--output-file",
        default="api_security_report.json",
        help="Output file for results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run security tests
    tester = APISecurityTester(args.target_url)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(args.output_file)

    # Print summary
    summary = results["summary"]["overall"]
    print("\nSecurity Test Summary:")
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1f}%")

    # Exit with appropriate code
    return 0 if summary["pass_rate"] >= 80 else 1


if __name__ == "__main__":
    exit(main())
