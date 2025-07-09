Quick Start Guide
=================

This guide will get you up and running with OpenAccelerator in minutes.

Prerequisites
-------------

Ensure you have OpenAccelerator installed::

    pip install open-accelerator

Basic Usage
-----------

1. Import and Initialize
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator import Accelerator, WorkloadManager
    from open_accelerator.core import SystemConfig

    # Initialize with default configuration
    accelerator = Accelerator()

    # Or with custom configuration
    config = SystemConfig(
        pe_array_size=(128, 128),
        memory_size="2GB",
        precision="fp16"
    )
    accelerator = Accelerator(config=config)

2. Create and Run Workloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a GEMM workload
    workload = WorkloadManager.create_gemm_workload(
        matrix_a_shape=(1024, 1024),
        matrix_b_shape=(1024, 1024),
        precision="fp16"
    )

    # Run the workload
    result = accelerator.run(workload)

    # Access performance metrics
    print(f"Execution time: {result.execution_time}ms")
    print(f"Throughput: {result.throughput} GOPS")
    print(f"Power consumption: {result.power_consumption}W")

3. API Server Usage
~~~~~~~~~~~~~~~~~~

Start the FastAPI server::

    uvicorn open_accelerator.api.main:app --reload

Example API requests:

.. code-block:: python

    import requests

    # Submit a workload via API
    response = requests.post("http://localhost:8000/workloads/submit", json={
        "type": "gemm",
        "matrix_a_shape": [1024, 1024],
        "matrix_b_shape": [1024, 1024],
        "precision": "fp16"
    })

    workload_id = response.json()["workload_id"]

    # Check workload status
    status = requests.get(f"http://localhost:8000/workloads/{workload_id}/status")
    print(status.json())

Medical AI Workflows
-------------------

For medical applications with HIPAA compliance:

.. code-block:: python

    from open_accelerator.medical import MedicalWorkload, ComplianceManager

    # Enable compliance mode
    compliance = ComplianceManager(hipaa_mode=True)

    # Create medical workload
    medical_workload = MedicalWorkload(
        model_type="medical_imaging",
        data_encryption=True,
        audit_logging=True
    )

    # Run with compliance validation
    result = accelerator.run(medical_workload, compliance=compliance)

OpenAI Agents Integration
------------------------

Integrate with OpenAI Agents SDK:

.. code-block:: python

    from open_accelerator.ai import AgentManager
    from openai import OpenAI

    # Initialize agent manager
    agent_manager = AgentManager(
        openai_client=OpenAI(api_key="your-api-key"),
        accelerator=accelerator
    )

    # Create AI agent for workload optimization
    optimizer_agent = agent_manager.create_agent(
        name="workload_optimizer",
        instructions="Optimize workload parameters for best performance",
        tools=["performance_analyzer", "memory_optimizer"]
    )

    # Let agent optimize workload
    optimized_workload = optimizer_agent.optimize(workload)
    result = accelerator.run(optimized_workload)

Performance Monitoring
---------------------

Monitor performance in real-time:

.. code-block:: python

    from open_accelerator.benchmarks import PerformanceMonitor

    # Create performance monitor
    monitor = PerformanceMonitor(accelerator)

    # Start monitoring
    monitor.start()

    # Run workload with monitoring
    result = accelerator.run(workload)

    # Get detailed metrics
    metrics = monitor.get_metrics()
    print(f"Memory utilization: {metrics.memory_utilization}%")
    print(f"PE array efficiency: {metrics.pe_efficiency}%")

Docker Quick Start
-----------------

Run OpenAccelerator in Docker:

.. code-block:: bash

    # Pull and run the container
    docker run -p 8000:8000 nikjois/openaccelerator:latest

    # Or use docker-compose
    git clone https://github.com/nikjois/OpenAccelerator.git
    cd OpenAccelerator
    docker-compose up -d

Configuration Examples
--------------------

System Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.core import SystemConfig

    # High-performance configuration
    high_perf_config = SystemConfig(
        pe_array_size=(512, 512),
        memory_hierarchy={
            "l1_cache": "128KB",
            "l2_cache": "2MB",
            "main_memory": "8GB"
        },
        precision="fp16",
        power_management=True
    )

    # Low-power configuration
    low_power_config = SystemConfig(
        pe_array_size=(128, 128),
        memory_hierarchy={
            "l1_cache": "32KB",
            "l2_cache": "512KB",
            "main_memory": "2GB"
        },
        precision="int8",
        power_management=True,
        power_budget=50  # watts
    )

Workload Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # CNN workload for image processing
    cnn_workload = WorkloadManager.create_cnn_workload(
        input_shape=(224, 224, 3),
        model_architecture="resnet50",
        batch_size=32,
        precision="fp16"
    )

    # Transformer workload for NLP
    transformer_workload = WorkloadManager.create_transformer_workload(
        sequence_length=512,
        hidden_size=768,
        num_heads=12,
        batch_size=16,
        precision="fp16"
    )

Testing and Validation
---------------------

Run comprehensive tests:

.. code-block:: bash

    # Run all tests
    pytest tests/ -v

    # Run specific test categories
    pytest tests/test_accelerator.py -v
    pytest tests/test_medical.py -v
    pytest tests/test_benchmarks.py -v

    # Run performance tests
    pytest tests/test_performance.py -v --benchmark-only

Next Steps
----------

1. **Read the User Guide**: For detailed usage instructions
2. **Explore Examples**: Check the ``examples/`` directory
3. **Review API Documentation**: For complete API reference
4. **Medical Applications**: See the medical guide for compliance requirements
5. **Performance Tuning**: Read benchmarking guide for optimization tips

Common Use Cases
---------------

* **AI/ML Research**: Simulate different accelerator architectures
* **Medical AI**: HIPAA-compliant medical imaging and diagnostics
* **Performance Analysis**: Benchmark and optimize workloads
* **System Design**: Evaluate accelerator configurations
* **API Development**: Build applications with REST API integration

Support
-------

* Documentation: https://openaccelerator.readthedocs.io/
* Issues: https://github.com/nikjois/OpenAccelerator/issues
* Email: nikjois@llamasearch.ai
