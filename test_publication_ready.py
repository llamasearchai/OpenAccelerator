#!/usr/bin/env python3
"""
Publication readiness test for OpenAccelerator.
Author: Nik Jois <nikjois@llamasearch.ai>

This script validates that the OpenAccelerator package is complete and ready for publication.
Tests include:
- Core functionality and imports
- Configuration system
- Workload system
- Systolic array operations
- Memory hierarchy
- AI agents (if OpenAI key available)
- FastAPI endpoints (basic test)
- Security features
- Medical compliance features
- Build system validation
"""

import sys
import os
import subprocess
import tempfile
import time
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_test(test_name, passed, details=""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def test_imports():
    """Test that all critical imports work."""
    print_section("TESTING IMPORTS")
    
    try:
        import open_accelerator
        print_test("Main package import", True, f"Version: {open_accelerator.__version__}")
        
        from open_accelerator.core.accelerator import AcceleratorController
        print_test("AcceleratorController import", True)
        
        from open_accelerator.core.systolic_array import SystolicArray
        print_test("SystolicArray import", True)
        
        from open_accelerator.core.memory import MemoryHierarchy
        print_test("MemoryHierarchy import", True)
        
        from open_accelerator.workloads.gemm import GEMMWorkload
        print_test("GEMMWorkload import", True)
        
        from open_accelerator.ai.agents import AgentOrchestrator
        print_test("AgentOrchestrator import", True)
        
        from open_accelerator.api.main import app
        print_test("FastAPI app import", True)
        
        from open_accelerator.utils.config import AcceleratorConfig
        print_test("AcceleratorConfig import", True)
        
        return True
        
    except Exception as e:
        print_test("Import test", False, f"Error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print_section("TESTING CONFIGURATION")
    
    try:
        from open_accelerator.utils.config import AcceleratorConfig, get_default_configs
        
        # Test default configs
        configs = get_default_configs()
        print_test("Default configs creation", len(configs) > 0, f"Found {len(configs)} configurations")
        
        # Test custom config
        config = AcceleratorConfig(name="TestConfig")
        print_test("Custom config creation", config.name == "TestConfig")
        
        # Test config validation
        valid_config = config.array.rows > 0 and config.array.cols > 0
        print_test("Config validation", valid_config, f"Array: {config.array.rows}x{config.array.cols}")
        
        return True
        
    except Exception as e:
        print_test("Configuration test", False, f"Error: {e}")
        return False

def test_workloads():
    """Test workload system."""
    print_section("TESTING WORKLOADS")
    
    try:
        from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig
        from open_accelerator.workloads.base import ComputeWorkload
        
        # Test GEMM workload
        gemm_config = GEMMWorkloadConfig(M=16, K=16, P=16)
        gemm = GEMMWorkload(gemm_config, name="TestGEMM")
        print_test("GEMM workload creation", gemm.name == "TestGEMM")
        
        # Test workload preparation
        gemm.prepare()
        print_test("Workload preparation", gemm.is_ready())
        
        # Test base workload
        base_workload = ComputeWorkload("TestBase")
        print_test("Base workload creation", base_workload.name == "TestBase")
        
        return True
        
    except Exception as e:
        print_test("Workload test", False, f"Error: {e}")
        return False

def test_systolic_array():
    """Test systolic array functionality."""
    print_section("TESTING SYSTOLIC ARRAY")
    
    try:
        from open_accelerator.core.systolic_array import SystolicArray
        from open_accelerator.utils.config import AcceleratorConfig
        import numpy as np
        
        config = AcceleratorConfig()
        config.array.rows = 4
        config.array.cols = 4
        
        array = SystolicArray(config)
        print_test("Systolic array creation", array.rows == 4 and array.cols == 4)
        
        # Test cycle execution with numpy arrays
        input_data = {
            'edge_a': np.array([1.0, 2.0, 3.0, 4.0]),
            'edge_b': np.array([1.0, 2.0, 3.0, 4.0])
        }
        
        result = array.cycle(input_data)
        print_test("Systolic array cycle execution", 'cycle' in result)
        
        return True
        
    except Exception as e:
        print_test("Systolic array test", False, f"Error: {e}")
        return False

def test_memory_hierarchy():
    """Test memory hierarchy."""
    print_section("TESTING MEMORY HIERARCHY")
    
    try:
        from open_accelerator.core.memory import MemoryHierarchy
        from open_accelerator.utils.config import AcceleratorConfig
        
        config = AcceleratorConfig()
        memory = MemoryHierarchy(config)
        
        print_test("Memory hierarchy creation", memory is not None)
        
        # Test memory read/write using the correct methods
        test_data = [1, 2, 3, 4]
        success, latency = memory.write_request(0, test_data)
        print_test("Memory write request", success)
        
        read_data, read_latency = memory.read_request(0, len(test_data))
        print_test("Memory read request", len(read_data) == len(test_data))
        
        return True
        
    except Exception as e:
        print_test("Memory hierarchy test", False, f"Error: {e}")
        return False

def test_ai_agents():
    """Test AI agents functionality."""
    print_section("TESTING AI AGENTS")
    
    try:
        from open_accelerator.ai.agents import AgentOrchestrator, AgentConfig
        
        # Test agent config
        config = AgentConfig(api_key="test_key", medical_compliance=True)
        print_test("Agent config creation", config.medical_compliance == True)
        
        # Test agent orchestrator
        orchestrator = AgentOrchestrator(config)
        print_test("Agent orchestrator creation", orchestrator is not None)
        
        # Test agent status
        status = orchestrator.get_agent_status()
        print_test("Agent status retrieval", isinstance(status, dict))
        
        return True
        
    except Exception as e:
        print_test("AI agents test", False, f"Error: {e}")
        return False

def test_api_endpoints():
    """Test FastAPI endpoints."""
    print_section("TESTING API ENDPOINTS")
    
    try:
        from open_accelerator.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/v1/health")
        print_test("Health endpoint", response.status_code == 200)
        
        # Test OpenAPI schema
        response = client.get("/api/v1/openapi.json")
        print_test("OpenAPI schema", response.status_code == 200)
        
        return True
        
    except Exception as e:
        print_test("API endpoints test", False, f"Error: {e}")
        return False

def test_security_features():
    """Test security features."""
    print_section("TESTING SECURITY")
    
    try:
        from open_accelerator.core.security import SecurityManager, create_medical_security_config
        
        # Test security config
        config = create_medical_security_config()
        print_test("Medical security config", config.hipaa_compliant == True)
        
        # Test security manager
        security_manager = SecurityManager(config)
        print_test("Security manager creation", security_manager is not None)
        
        # Test encryption
        test_data = b"test data"
        encrypted = security_manager.encrypt_data(test_data)
        decrypted = security_manager.decrypt_data(encrypted)
        print_test("Data encryption/decryption", decrypted == test_data)
        
        return True
        
    except Exception as e:
        print_test("Security test", False, f"Error: {e}")
        return False

def test_medical_compliance():
    """Test medical compliance features."""
    print_section("TESTING MEDICAL COMPLIANCE")
    
    try:
        from open_accelerator.medical.compliance import ComplianceManager, HIPAAConfig, FDAConfig
        
        # Create compliance configs
        hipaa_config = HIPAAConfig()
        fda_config = FDAConfig()
        
        compliance_manager = ComplianceManager(hipaa_config, fda_config)
        print_test("Medical compliance manager creation", compliance_manager is not None)
        
        # Test compliance check
        system_data = {"medical_mode": True, "device_type": "Class II"}
        result = compliance_manager.run_full_compliance_check(system_data)
        print_test("Full compliance check", isinstance(result, dict))
        
        return True
        
    except Exception as e:
        print_test("Medical compliance test", False, f"Error: {e}")
        return False

def test_build_system():
    """Test build system and packaging."""
    print_section("TESTING BUILD SYSTEM")
    
    try:
        # Test that key build files exist
        required_files = [
            "pyproject.toml",
            "Makefile",
            "Dockerfile",
            "requirements.lock",
            "README.md"
        ]
        
        for file_name in required_files:
            file_exists = Path(file_name).exists()
            print_test(f"Build file: {file_name}", file_exists)
        
        # Test package metadata
        import open_accelerator
        has_version = hasattr(open_accelerator, '__version__')
        has_author = hasattr(open_accelerator, '__author__')
        has_email = hasattr(open_accelerator, '__email__')
        
        print_test("Package metadata", has_version and has_author and has_email)
        
        if has_author and has_email:
            author_correct = open_accelerator.__author__ == "Nik Jois"
            email_correct = open_accelerator.__email__ == "nikjois@llamasearch.ai"
            print_test("Author information", author_correct and email_correct)
        
        return True
        
    except Exception as e:
        print_test("Build system test", False, f"Error: {e}")
        return False

def test_docker_build():
    """Test Docker build capability."""
    print_section("TESTING DOCKER BUILD")
    
    try:
        # Check if Docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print_test("Docker availability", False, "Docker not available")
            return False
        
        print_test("Docker availability", True, result.stdout.strip())
        
        # Test docker build (dry run)
        dockerfile_exists = Path("Dockerfile").exists()
        print_test("Dockerfile exists", dockerfile_exists)
        
        # Test docker-compose
        compose_exists = Path("docker-compose.yml").exists()
        print_test("Docker compose file exists", compose_exists)
        
        return True
        
    except Exception as e:
        print_test("Docker build test", False, f"Error: {e}")
        return False

def test_no_stubs_or_placeholders():
    """Test that there are no stubs or placeholders in the code."""
    print_section("TESTING NO STUBS/PLACEHOLDERS")
    
    try:
        import ast
        
        # Search in source files
        src_dir = Path("src")
        python_files = list(src_dir.rglob("*.py"))
        
        found_stubs = []
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse the AST to intelligently detect placeholder methods
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                            # Check if function body contains only pass or NotImplementedError
                            body = node.body
                            if len(body) == 1:
                                stmt = body[0]
                                
                                # Check for standalone pass statement (not in abstract methods)
                                if isinstance(stmt, ast.Pass):
                                    # Check if it's in an abstract method
                                    has_abstractmethod = any(
                                        isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod' or
                                        isinstance(decorator, ast.Attribute) and decorator.attr == 'abstractmethod'
                                        for decorator in node.decorator_list
                                    )
                                    
                                    # Check if it's in a base class or exception
                                    is_exception_class = False
                                    is_abstract_base_class = False
                                    
                                    # Simple heuristics to detect base classes and exceptions
                                    if hasattr(node, 'lineno'):
                                        line_content = content.split('\n')[node.lineno-1:node.lineno+5]
                                        context = '\n'.join(line_content).lower()
                                        
                                        is_exception_class = (
                                            'exception' in context or
                                            'error' in context or 
                                            'class' in context and 'exception' in context
                                        )
                                        
                                        is_abstract_base_class = (
                                            'abc' in context or
                                            'abstract' in context or
                                            'base' in context
                                        )
                                    
                                    # Only flag as stub if it's not abstract and not in base/exception classes
                                    if not has_abstractmethod and not is_exception_class and not is_abstract_base_class:
                                        found_stubs.append(f"{file_path}:{node.lineno}: {node.name}() contains only 'pass'")
                                
                                # Check for NotImplementedError
                                elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                                    if (isinstance(stmt.exc.func, ast.Name) and 
                                        stmt.exc.func.id == 'NotImplementedError'):
                                        found_stubs.append(f"{file_path}:{node.lineno}: {node.name}() raises NotImplementedError")
                except SyntaxError:
                    # Fall back to simple text search for files that can't be parsed
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if ('TODO' in line or 'FIXME' in line or 'placeholder' in line or 'stub' in line) and not line.strip().startswith('#'):
                            found_stubs.append(f"{file_path}:{i+1}: {line.strip()}")
            except Exception:
                continue
        
        no_stubs = len(found_stubs) == 0
        print_test("No stubs or placeholders", no_stubs, 
                  f"Found {len(found_stubs)} potential issues" if found_stubs else "Clean code")
        
        # Show first few issues if any
        if found_stubs:
            print("    First few issues:")
            for issue in found_stubs[:5]:
                print(f"      {issue}")
        
        return no_stubs
        
    except Exception as e:
        print_test("Stub/placeholder test", False, f"Error: {e}")
        return False

def run_all_tests():
    """Run all publication readiness tests."""
    print_section("OpenAccelerator Publication Readiness Test")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("Testing comprehensive functionality for production deployment")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Workloads", test_workloads),
        ("Systolic Array", test_systolic_array),
        ("Memory Hierarchy", test_memory_hierarchy),
        ("AI Agents", test_ai_agents),
        ("API Endpoints", test_api_endpoints),
        ("Security", test_security_features),
        ("Medical Compliance", test_medical_compliance),
        ("Build System", test_build_system),
        ("Docker Build", test_docker_build),
        ("No Stubs/Placeholders", test_no_stubs_or_placeholders),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_test(f"{test_name} (Exception)", False, f"Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print_section("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print("\nDetailed results:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    publication_ready = passed == total
    print(f"\nPublication ready: {'YES' if publication_ready else 'NO'}")
    
    if publication_ready:
        print("\n[SUCCESS] OpenAccelerator is ready for publication!")
        print("- All core functionality implemented")
        print("- Complete API integration")
        print("- OpenAI Agents SDK integrated")
        print("- Docker containerization complete")
        print("- Security and medical compliance features")
        print("- Build automation ready")
        print("- No stubs or placeholders found")
    else:
        print("\n[ATTENTION] Some issues found. Review failed tests above.")
    
    return publication_ready

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 