#!/usr/bin/env python3
"""
OpenAccelerator Simple System Validation Script

This script performs essential validation of OpenAccelerator components
to ensure basic functionality and production readiness.

Author: Nik Jois <nikjois@llamasearch.ai>
Date: January 8, 2025
Version: 1.0.0
"""

import sys
import os
import logging
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_basic_imports():
    """Test basic module imports."""
    logger.info("Testing basic imports...")
    
    try:
        import open_accelerator
        logger.info("[SUCCESS] OpenAccelerator module imported successfully")
        
        from open_accelerator.core.accelerator import AcceleratorController
        logger.info("[SUCCESS] AcceleratorController imported successfully")
        
        from open_accelerator.core.systolic_array import SystolicArray
        logger.info("[SUCCESS] SystolicArray imported successfully")
        
        from open_accelerator.workloads.gemm import GEMMWorkload
        logger.info("[SUCCESS] GEMMWorkload imported successfully")
        
        from open_accelerator.utils.config import AcceleratorConfig
        logger.info("[SUCCESS] AcceleratorConfig imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    logger.info("Testing configuration system...")
    
    try:
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig, MemoryConfig
        
        config = AcceleratorConfig(
            array=ArrayConfig(rows=4, cols=4, dataflow="OS"),
            memory=MemoryConfig(size=1024, bandwidth=16),
            medical_mode=True
        )
        
        assert config.array.rows == 4
        assert config.array.cols == 4
        assert config.medical_mode is True
        
        logger.info("[SUCCESS] Configuration system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Configuration test failed: {e}")
        return False

def test_systolic_array():
    """Test systolic array functionality."""
    logger.info("Testing systolic array...")
    
    try:
        from open_accelerator.core.systolic_array import SystolicArray
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig
        
        config = AcceleratorConfig(
            array=ArrayConfig(rows=2, cols=2, dataflow="OS")
        )
        
        array = SystolicArray(config)
        assert array.rows == 2
        assert array.cols == 2
        
        logger.info("[SUCCESS] Systolic array initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Systolic array test failed: {e}")
        return False

def test_gemm_workload():
    """Test GEMM workload functionality."""
    logger.info("Testing GEMM workload...")
    
    try:
        from open_accelerator.workloads.gemm import GEMMWorkload
        from open_accelerator.utils.config import AcceleratorConfig
        
        config = AcceleratorConfig()
        
        # Create small test matrices
        matrix_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        matrix_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        workload = GEMMWorkload(config, matrix_a, matrix_b)
        assert workload.matrix_a.shape == (2, 2)
        assert workload.matrix_b.shape == (2, 2)
        
        logger.info("[SUCCESS] GEMM workload creation successful")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] GEMM workload test failed: {e}")
        return False

def test_accelerator_controller():
    """Test accelerator controller functionality."""
    logger.info("Testing accelerator controller...")
    
    try:
        from open_accelerator.core.accelerator import AcceleratorController
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig
        
        config = AcceleratorConfig(
            array=ArrayConfig(rows=2, cols=2, dataflow="OS")
        )
        
        controller = AcceleratorController(config)
        assert controller.config == config
        assert controller.systolic_array is not None
        
        logger.info("[SUCCESS] Accelerator controller initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Accelerator controller test failed: {e}")
        return False

def test_api_server():
    """Test API server health."""
    logger.info("Testing API server...")
    
    try:
        import requests
        
        # Try to connect to the API server
        response = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"[SUCCESS] API server is healthy: {health_data.get('status', 'unknown')}")
            return True
        else:
            logger.warning(f"[WARNING]  API server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.warning("[WARNING]  API server is not running or not accessible")
        return False
    except Exception as e:
        logger.error(f"[ERROR] API server test failed: {e}")
        return False

def test_basic_simulation():
    """Test basic simulation functionality."""
    logger.info("Testing basic simulation...")
    
    try:
        from open_accelerator.simulation.simulator import Simulator
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig
        from open_accelerator.workloads.gemm import GEMMWorkload
        
        # Create configuration
        config = AcceleratorConfig(
            array=ArrayConfig(rows=2, cols=2, dataflow="OS")
        )
        
        # Create workload
        matrix_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        matrix_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        workload = GEMMWorkload(config, matrix_a, matrix_b)
        
        # Run simulation
        simulator = Simulator(config, workload)
        result = simulator.run()
        
        assert result is not None
        assert "total_cycles" in result
        assert result["total_cycles"] > 0
        
        logger.info("[SUCCESS] Basic simulation successful")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Basic simulation test failed: {e}")
        return False

def test_medical_compliance():
    """Test medical compliance features."""
    logger.info("Testing medical compliance...")
    
    try:
        from open_accelerator.medical.compliance import ComplianceManager, HIPAAConfig, FDAConfig
        
        # Create compliance configurations
        hipaa_config = HIPAAConfig(
            enable_encryption=True,
            audit_all_access=True,
            require_authentication=True
        )
        
        fda_config = FDAConfig(
            device_classification="Class II",
            clinical_validation_required=True,
            risk_management_required=True
        )
        
        # Create compliance manager
        compliance_manager = ComplianceManager(hipaa_config, fda_config)
        
        # Test compliance check
        test_data = {
            "device_id": "TEST-001",
            "software_version": "1.0.0",
            "clinical_validation": True,
            "risk_assessment": "completed"
        }
        
        result = compliance_manager.run_full_compliance_check(test_data)
        assert result is not None
        assert "hipaa_compliance" in result
        assert "fda_compliance" in result
        
        logger.info("[SUCCESS] Medical compliance features working")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Medical compliance test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("OpenAccelerator System Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration System", test_configuration),
        ("Systolic Array", test_systolic_array),
        ("GEMM Workload", test_gemm_workload),
        ("Accelerator Controller", test_accelerator_controller),
        ("API Server", test_api_server),
        ("Basic Simulation", test_basic_simulation),
        ("Medical Compliance", test_medical_compliance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"[ERROR] {test_name} test crashed: {e}")
            failed += 1
    
    total = passed + failed
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        logger.info("[COMPLETE] ALL TESTS PASSED - System is functional!")
        return 0
    else:
        logger.error(f"[ERROR] {failed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 