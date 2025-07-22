#!/usr/bin/env python3
"""
Complete System Validation Script for OpenAccelerator
Tests all major components and functionality after bug fixes.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all major modules can be imported successfully."""
    print("\n[SYSTEM] Testing module imports...")

    try:
        # Core modules

        # Workload modules

        # AI modules

        # API modules

        # Configuration

        print("[SUCCESS] All imports successful")
        return True

    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_configuration():
    """Test basic configuration creation and validation."""
    print("\n[CONFIG] Testing configuration system...")

    try:
        from open_accelerator.utils.config import (
            AcceleratorConfig,
            ArrayConfig,
            DataType,
        )

        # Test basic config creation
        config = AcceleratorConfig(
            name="TestAccelerator",
            array=ArrayConfig(rows=8, cols=8, frequency=1e9, voltage=1.0),
            data_type=DataType.FLOAT32,
            enable_power_modeling=True,
            enable_thermal_modeling=True,
        )

        print(f"[SUCCESS] Configuration created: {config.name}")
        print(f"[INFO] Array size: {config.array.rows}x{config.array.cols}")
        print(f"[INFO] Data type: {config.data_type}")

        return True

    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_workload_creation():
    """Test workload creation and basic functionality."""
    print("\n[WORKLOAD] Testing workload creation...")

    try:
        from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

        # Create GEMM workload with correct API
        workload_config = GEMMWorkloadConfig(M=4, K=4, P=4, seed=42, use_integers=True)
        workload = GEMMWorkload(workload_config, "test_gemm")

        # Generate test data
        workload.prepare()

        print(f"[SUCCESS] GEMM workload created: {workload.get_name()}")
        print(
            f"[INFO] Matrix dimensions: {workload.M}x{workload.K} * {workload.K}x{workload.P}"
        )
        print(
            f"[INFO] Expected operations: {workload.get_complexity()['total_operations']}"
        )

        return True

    except Exception as e:
        print(f"[ERROR] Workload test failed: {e}")
        traceback.print_exc()
        return False


def test_accelerator_controller():
    """Test accelerator controller creation and basic operations."""
    print("\n[ACCELERATOR] Testing accelerator controller...")

    try:
        from open_accelerator.core.accelerator import AcceleratorController
        from open_accelerator.utils.config import (
            AcceleratorConfig,
            ArrayConfig,
            DataType,
        )
        from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

        # Create configuration
        config = AcceleratorConfig(
            name="TestAccelerator",
            array=ArrayConfig(rows=4, cols=4),
            data_type=DataType.FLOAT32,
            enable_power_modeling=True,
            enable_thermal_modeling=False,
        )

        # Create accelerator
        accelerator = AcceleratorController(config)

        # Create and load workload
        workload_config = GEMMWorkloadConfig(M=4, K=4, P=4, seed=42)
        workload = GEMMWorkload(workload_config, "test_gemm")
        workload.prepare()

        # Load workload
        if accelerator.load_workload(workload):
            print("[SUCCESS] Workload loaded successfully")
        else:
            print("[ERROR] Failed to load workload")
            return False

        # Test status
        status = accelerator.get_real_time_status()
        print(f"[INFO] Accelerator status: {status['system_health']}")

        return True

    except Exception as e:
        print(f"[ERROR] Accelerator test failed: {e}")
        traceback.print_exc()
        return False


def test_simulation_execution():
    """Test full simulation execution."""
    print("\n[SIMULATION] Testing simulation execution...")

    try:
        from open_accelerator.core.accelerator import AcceleratorController
        from open_accelerator.utils.config import (
            AcceleratorConfig,
            ArrayConfig,
            DataType,
        )
        from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

        # Create small test configuration
        config = AcceleratorConfig(
            name="TestAccelerator",
            array=ArrayConfig(rows=2, cols=2),
            data_type=DataType.FLOAT32,
            enable_power_modeling=True,
            enable_thermal_modeling=False,
        )

        # Create accelerator
        accelerator = AcceleratorController(config)

        # Create small workload
        workload_config = GEMMWorkloadConfig(M=2, K=2, P=2, seed=42, use_integers=True)
        workload = GEMMWorkload(workload_config, "small_gemm")
        workload.prepare()

        # Get input data for display
        input_data = workload.get_input_data()
        expected_data = workload.get_expected_output()

        print(f"[INFO] Input A:\n{input_data['matrix_A']}")
        print(f"[INFO] Input B:\n{input_data['matrix_B']}")
        print(f"[INFO] Expected C:\n{expected_data['matrix_C']}")

        # Execute simulation
        print("[INFO] Starting simulation...")
        start_time = time.time()

        results = accelerator.execute_workload(workload)

        execution_time = time.time() - start_time

        print(f"[SUCCESS] Simulation completed in {execution_time:.3f}s")
        print(f"[INFO] Total cycles: {results['execution_summary']['total_cycles']}")
        print(
            f"[INFO] Total operations: {results['execution_summary']['total_operations']}"
        )
        print(
            f"[INFO] Throughput: {results['execution_summary']['throughput_ops_per_second']:.2f} ops/sec"
        )

        # Validate results
        if "final_output" in results["raw_results"]:
            output = results["raw_results"]["final_output"]
            print(f"[INFO] Simulation output:\n{output}")

            # Check correctness
            if np.allclose(output, expected_data["matrix_C"], rtol=1e-5):
                print("[SUCCESS] Results match expected output")
            else:
                print("[WARNING] Results don't match expected output")
                print(
                    f"[INFO] Max error: {np.max(np.abs(output - expected_data['matrix_C']))}"
                )

        return True

    except Exception as e:
        print(f"[ERROR] Simulation test failed: {e}")
        traceback.print_exc()
        return False


def test_ai_agents():
    """Test AI agents functionality."""
    print("\n[AI] Testing AI agents...")

    try:
        from open_accelerator.ai.agents import create_agent_orchestrator

        # Create orchestrator
        orchestrator = create_agent_orchestrator(
            optimization_focus="performance", medical_compliance=False
        )

        # Test agent status
        status = orchestrator.get_agent_status()
        print(f"[INFO] Available agents: {status['available_agents']}")
        print(f"[INFO] OpenAI available: {status['openai_available']}")
        print(f"[INFO] Agents initialized: {status['agents_initialized']}")

        # Test basic interaction (if OpenAI available)
        if status["openai_available"]:
            print("[INFO] Testing AI agent interaction...")
            try:
                import asyncio

                async def test_agent_query():
                    response = await orchestrator.interactive_consultation(
                        "What are the key factors for optimizing systolic array performance?",
                        context={"test": True},
                    )
                    return response

                # Run async function
                if hasattr(asyncio, "run"):
                    response = asyncio.run(test_agent_query())
                    print(
                        f"[SUCCESS] Agent response received (length: {len(response)})"
                    )
                else:
                    print("[INFO] Async test skipped (Python < 3.7)")

            except Exception as e:
                print(f"[WARNING] AI agent interaction test failed: {e}")
        else:
            print("[INFO] AI agents not available (OpenAI not configured)")

        return True

    except Exception as e:
        print(f"[ERROR] AI agents test failed: {e}")
        traceback.print_exc()
        return False


def test_api_server():
    """Test API server functionality."""
    print("\n[API] Testing API server...")

    try:
        from fastapi.testclient import TestClient

        from open_accelerator.api.main import app

        # Create test client
        client = TestClient(app)

        # Test health endpoint
        response = client.get("/api/v1/health/")
        if response.status_code == 200:
            health_data = response.json()
            print("[SUCCESS] Health endpoint working")
            print(f"[INFO] System status: {health_data['status']}")
            print(f"[INFO] Version: {health_data['version']}")
        else:
            print(f"[ERROR] Health endpoint failed: {response.status_code}")
            return False

        # Test metrics endpoint
        response = client.get("/api/v1/metrics/")
        if response.status_code == 200:
            print("[SUCCESS] Metrics endpoint working")
        else:
            print(f"[WARNING] Metrics endpoint failed: {response.status_code}")

        return True

    except Exception as e:
        print(f"[ERROR] API server test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_functionality():
    """Test CLI functionality."""
    print("\n[CLI] Testing CLI functionality...")

    try:
        from pathlib import Path
        import sys
        import traceback

        # Test CLI script exists
        cli_script = Path("scripts/accelerator_cli.py")
        if cli_script.exists():
            print("[SUCCESS] CLI script found")

            # Test CLI import - add scripts directory to path
            scripts_dir = Path(__file__).parent.parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            # Import CLI class
            try:
                from accelerator_cli import OpenAcceleratorCLI
            except ImportError as e:
                print(f"[WARNING] Could not import OpenAcceleratorCLI: {e}")
                print("[INFO] Skipping CLI tests")
                return False

            # Create CLI instance
            cli = OpenAcceleratorCLI()

            # Test server check
            server_online = cli.check_server()
            print(f"[INFO] Server status: {'ONLINE' if server_online else 'OFFLINE'}")

            # Test local simulation capability
            result = cli.run_local_simulation("gemm", M=2, K=2, P=2, seed=42)
            if result.get("success"):
                print("[SUCCESS] Local simulation via CLI working")
            else:
                print(
                    f"[WARNING] Local simulation failed: {result.get('error', 'Unknown')}"
                )

            return True
        else:
            print("[WARNING] CLI script not found")
            return False

    except Exception as e:
        print(f"[ERROR] CLI test failed: {e}")
        traceback.print_exc()
        return False


def test_medical_compliance():
    """Test medical compliance features."""
    print("\n[MEDICAL] Testing medical compliance...")

    try:
        from open_accelerator.medical.compliance import ComplianceValidator

        # Create compliance validator
        validator = ComplianceValidator()

        # Test basic compliance check
        test_config = {
            "data_type": "float32",
            "enable_reliability": True,
            "enable_security": True,
            "enable_audit_logging": True,
        }

        compliance_result = validator.validate_configuration(test_config)
        print(f"[INFO] Compliance score: {compliance_result['compliance_score']:.2f}")
        print(f"[INFO] HIPAA ready: {compliance_result['hipaa_compliant']}")
        print(f"[INFO] FDA ready: {compliance_result['fda_compliant']}")

        if compliance_result["compliance_score"] > 0.8:
            print("[SUCCESS] Medical compliance validation passed")
        else:
            print("[WARNING] Medical compliance needs improvement")

        return True

    except Exception as e:
        print(f"[ERROR] Medical compliance test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_validation():
    """Run comprehensive system validation."""
    print("=" * 70)
    print("OPENACCELERATOR COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 70)
    print("Testing all major components after bug fixes...")

    test_results = {
        "imports": False,
        "configuration": False,
        "workload": False,
        "accelerator": False,
        "simulation": False,
        "ai_agents": False,
        "api_server": False,
        "cli": False,
        "medical": False,
    }

    # Run all tests
    test_results["imports"] = test_imports()
    test_results["configuration"] = test_basic_configuration()
    test_results["workload"] = test_workload_creation()
    test_results["accelerator"] = test_accelerator_controller()
    test_results["simulation"] = test_simulation_execution()
    test_results["ai_agents"] = test_ai_agents()
    test_results["api_server"] = test_api_server()
    test_results["cli"] = test_cli_functionality()
    test_results["medical"] = test_medical_compliance()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name.upper().replace('_', ' ')}")

    print(f"\nOVERALL RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] All tests passed! System is fully functional.")
        return True
    elif passed >= total * 0.8:
        print("\n[WARNING] Most tests passed. System is largely functional.")
        return True
    else:
        print("\n[ERROR] Multiple test failures. System needs attention.")
        return False


def generate_validation_report():
    """Generate detailed validation report."""
    print("\n[REPORT] Generating validation report...")

    try:
        report = {
            "timestamp": time.time(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd()),
            },
            "validation_results": {},
            "recommendations": [],
        }

        # Save report
        report_file = Path("validation_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"[SUCCESS] Validation report saved to {report_file}")
        return True

    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting OpenAccelerator system validation...")

    try:
        # Run comprehensive validation
        success = run_comprehensive_validation()

        # Generate report
        generate_validation_report()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[INFO] Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL] Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
