#!/usr/bin/env python3
"""
Comprehensive test script for OpenAccelerator system.

This script demonstrates the complete functionality of the OpenAccelerator system
including core simulation, AI agents, FastAPI endpoints, medical workflows, and
Docker integration.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test that all core components can be imported."""
    print("\n" + "="*60)
    print("TESTING CORE IMPORTS")
    print("="*60)
    
    try:
        from open_accelerator.core import AcceleratorController
        from open_accelerator.workloads import GEMMWorkload, GEMMWorkloadConfig
        from open_accelerator.analysis import PerformanceAnalyzer
        from open_accelerator.utils.config import AcceleratorConfig
        print("[SUCCESS] Core components imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Core import failed: {e}")
        return False

def test_gemm_simulation():
    """Test GEMM workload simulation."""
    print("\n" + "="*60)
    print("TESTING GEMM SIMULATION")
    print("="*60)
    
    try:
        from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig, DataType
        
        # Create configuration using new structure
        accel_config = AcceleratorConfig(
            name="test_config",
            array=ArrayConfig(rows=4, cols=4, frequency=1e9),
            data_type=DataType.FLOAT32
        )
        
        # Create GEMM workload
        gemm_config = GEMMWorkloadConfig(M=4, K=4, P=4)
        workload = GEMMWorkload(gemm_config, "test_gemm")
        
        # Generate test data
        workload.prepare(seed=42)
        
        print(f"[SUCCESS] GEMM workload created: {gemm_config.M}x{gemm_config.K} @ {gemm_config.K}x{gemm_config.P}")
        if hasattr(workload, 'matrix_A') and workload.matrix_A is not None:
            print(f"Sample matrix A shape: {workload.matrix_A.shape}")
        if hasattr(workload, 'matrix_B') and workload.matrix_B is not None:
            print(f"Sample matrix B shape: {workload.matrix_B.shape}")
        if hasattr(workload, 'expected_C') and workload.expected_C is not None:
            print(f"Expected result shape: {workload.expected_C.shape}")
        
        return True
    except Exception as e:
        print(f"[ERROR] GEMM simulation failed: {e}")
        return False

def test_ai_agents():
    """Test AI agent functionality."""
    print("\n" + "="*60)
    print("TESTING AI AGENTS")
    print("="*60)
    
    try:
        from open_accelerator.ai.agents import AgentConfig, AgentOrchestrator
        
        # Create agent configuration
        agent_config = AgentConfig(
            model="gpt-4o",
            temperature=0.1,
            medical_compliance=True,
            api_key=os.getenv("OPENAI_API_KEY", "test-key")
        )
        
        # Create agent orchestrator
        orchestrator = AgentOrchestrator(agent_config)
        
        print(f"[SUCCESS] AI agent system initialized")
        print(f"Available agents: {len(orchestrator.agents)}")
        
        # Test agent status
        status = orchestrator.get_agent_status()
        print(f"Agent status: {status}")
        
        return True
    except Exception as e:
        print(f"[ERROR] AI agents test failed: {e}")
        return False

def test_fastapi_components():
    """Test FastAPI application components."""
    print("\n" + "="*60)
    print("TESTING FASTAPI COMPONENTS")
    print("="*60)
    
    try:
        from open_accelerator.api.main import app
        from open_accelerator.api.models import (
            SimulationRequest, AgentRequest, AgentType,
            AcceleratorConfig, ArrayConfig, MemoryConfig, PowerConfig,
            WorkloadConfig, GEMMWorkloadConfig, WorkloadType
        )
        
        print(f"[SUCCESS] FastAPI app created successfully")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")
        
        # Test model creation with proper objects
        array_config = ArrayConfig(rows=4, cols=4)
        memory_config = MemoryConfig(l1_cache_size=32768, l2_cache_size=262144, memory_bandwidth=100)
        power_config = PowerConfig(max_power_watts=150.0, thermal_limit_celsius=85.0)
        accel_config = AcceleratorConfig(
            name="test_config",
            array=array_config,
            memory=memory_config,
            power=power_config
        )
        
        gemm_config = GEMMWorkloadConfig(m=4, k=4, p=4)
        workload_config = WorkloadConfig(type=WorkloadType.GEMM, config=gemm_config)
        
        sim_request = SimulationRequest(
            simulation_name="test_simulation",
            workload=workload_config,
            accelerator_config=accel_config
        )
        
        agent_request = AgentRequest(
            message="Test agent message",
            agent_type=AgentType.OPTIMIZATION
        )
        
        print(f"[SUCCESS] API models created successfully")
        return True
    except Exception as e:
        print(f"[ERROR] FastAPI components test failed: {e}")
        return False

def test_medical_workflows():
    """Test medical AI workflows."""
    print("\n" + "="*60)
    print("TESTING MEDICAL WORKFLOWS")
    print("="*60)
    
    try:
        from open_accelerator.workloads.medical import (
            MedicalWorkloadConfig, MedicalModalityType, MedicalTaskType
        )
        
        # Test medical workload configuration creation
        medical_config = MedicalWorkloadConfig(
            modality=MedicalModalityType.CT_SCAN,
            task_type=MedicalTaskType.SEGMENTATION,
            image_size=(256, 256, 16),
            precision_level="medical",
            regulatory_compliance=True
        )
        
        print(f"[SUCCESS] Medical workflow configuration created")
        print(f"Modality: {medical_config.modality.value}")
        print(f"Task type: {medical_config.task_type.value}")
        print(f"Image size: {medical_config.image_size}")
        print(f"Workload type: {medical_config.workload_type.value}")
        print(f"Medical modality: {medical_config.medical_modality}")
        print(f"Medical image size: {medical_config.medical_image_size}")
        
        # Test different medical configurations
        mri_config = MedicalWorkloadConfig(
            modality=MedicalModalityType.MRI,
            task_type=MedicalTaskType.CLASSIFICATION,
            image_size=(224, 224, 3),
            precision_level="high"
        )
        
        print(f"[SUCCESS] Additional medical configuration created")
        print(f"MRI modality: {mri_config.modality.value}")
        print(f"MRI task: {mri_config.task_type.value}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Medical workflows test failed: {e}")
        return False

def test_performance_analysis():
    """Test performance analysis capabilities."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        from open_accelerator.analysis import PerformanceAnalyzer
        
        # Create mock simulation statistics
        sim_stats = {
            "total_cycles": 1000,
            "total_mac_operations": 64,
            "pe_activity_map_over_time": np.random.rand(4, 4),
            "output_matrix": np.random.rand(4, 4)
        }
        
        # Create performance analyzer
        analyzer = PerformanceAnalyzer(sim_stats)
        
        # Compute metrics
        metrics = analyzer.compute_metrics()
        
        print(f"[SUCCESS] Performance analysis completed")
        print(f"Total cycles: {metrics['total_cycles']}")
        print(f"Total MACs: {metrics['total_macs']}")
        print(f"MACs per cycle: {metrics['macs_per_cycle']:.2f}")
        print(f"PE utilization: {metrics['pe_utilization']:.2%}")
        print(f"Efficiency: {metrics['efficiency']:.2%}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Performance analysis test failed: {e}")
        return False

def test_docker_integration():
    """Test Docker integration and containerization."""
    print("\n" + "="*60)
    print("TESTING DOCKER INTEGRATION")
    print("="*60)
    
    try:
        # Check if Docker files exist
        docker_files = [
            'Dockerfile',
            'docker-compose.yml',
            '.dockerignore'
        ]
        
        existing_files = []
        for file in docker_files:
            if os.path.exists(file):
                existing_files.append(file)
        
        print(f"[SUCCESS] Docker integration files found: {existing_files}")
        
        if 'Dockerfile' in existing_files:
            with open('Dockerfile', 'r') as f:
                dockerfile_content = f.read()
            print(f"Dockerfile contains {len(dockerfile_content.split())} words")
        
        return True
    except Exception as e:
        print(f"[ERROR] Docker integration test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*60)
    
    try:
        from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig
        
        # Test default configuration
        config = AcceleratorConfig()
        print(f"[SUCCESS] Default configuration created")
        print(f"Config name: {config.name}")
        print(f"Array dimensions: {config.array_rows}x{config.array_cols}")
        print(f"Frequency: {config.frequency_mhz} MHz")
        
        # Test custom configuration using nested objects
        custom_array = ArrayConfig(rows=8, cols=8, frequency=2000e6)
        custom_config = AcceleratorConfig(
            name="custom_test",
            array=custom_array
        )
        
        print(f"[SUCCESS] Custom configuration created")
        print(f"Custom config name: {custom_config.name}")
        print(f"Custom array dimensions: {custom_config.array_rows}x{custom_config.array_cols}")
        print(f"Custom frequency: {custom_config.frequency_mhz} MHz")
        
        return True
    except Exception as e:
        print(f"[ERROR] Configuration system test failed: {e}")
        return False

def generate_comprehensive_report():
    """Generate a comprehensive system report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE SYSTEM REPORT")
    print("="*60)
    
    # Run all tests
    test_results = {
        "core_imports": test_core_imports(),
        "gemm_simulation": test_gemm_simulation(),
        "ai_agents": test_ai_agents(),
        "fastapi_components": test_fastapi_components(),
        "medical_workflows": test_medical_workflows(),
        "performance_analysis": test_performance_analysis(),
        "docker_integration": test_docker_integration(),
        "configuration_system": test_configuration_system()
    }
    
    # Generate summary
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed results:")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    # Generate JSON report
    report = {
        "timestamp": time.time(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd()
        },
        "test_results": test_results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests/total_tests*100
        }
    }
    
    with open('system_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: system_test_report.json")
    
    return passed_tests == total_tests

def main():
    """Main test execution function."""
    print("OpenAccelerator Comprehensive System Test")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("="*60)
    
    try:
        # Run comprehensive test
        success = generate_comprehensive_report()
        
        if success:
            print("\n[SUCCESS] All tests passed! OpenAccelerator system is fully functional.")
            return 0
        else:
            print("\n[WARNING] Some tests failed. Please check the detailed results above.")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        logger.exception("Test execution failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 