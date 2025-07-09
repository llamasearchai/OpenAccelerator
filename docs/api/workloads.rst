Workloads API
=============

This module provides workload management and execution capabilities for OpenAccelerator.

Base Workload
-------------

.. automodule:: open_accelerator.workloads.base
   :members:
   :undoc-members:
   :show-inheritance:

Base Workload Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.base.BaseWorkload
   :members:
   :undoc-members:
   :show-inheritance:

Workload Manager
~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.base.WorkloadManager
   :members:
   :undoc-members:
   :show-inheritance:

GEMM Workloads
--------------

.. automodule:: open_accelerator.workloads.gemm
   :members:
   :undoc-members:
   :show-inheritance:

GEMM Workload Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.gemm.GEMMWorkload
   :members:
   :undoc-members:
   :show-inheritance:

Matrix Operations
~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.gemm.MatrixOperations
   :members:
   :undoc-members:
   :show-inheritance:

Medical Workloads
-----------------

.. automodule:: open_accelerator.workloads.medical
   :members:
   :undoc-members:
   :show-inheritance:

Medical Workload Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.medical.MedicalWorkload
   :members:
   :undoc-members:
   :show-inheritance:

Medical Imaging
~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.medical.MedicalImagingWorkload
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
~~~~~~~~~~~

.. autoclass:: open_accelerator.workloads.medical.DiagnosticsWorkload
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Creating GEMM Workloads
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.workloads import WorkloadManager

    # Create basic GEMM workload
    gemm_workload = WorkloadManager.create_gemm_workload(
        matrix_a_shape=(1024, 1024),
        matrix_b_shape=(1024, 1024),
        precision="fp16"
    )

    # Create batched GEMM workload
    batched_gemm = WorkloadManager.create_batched_gemm_workload(
        batch_size=32,
        matrix_a_shape=(512, 512),
        matrix_b_shape=(512, 512),
        precision="fp16"
    )

    # Custom GEMM configuration
    custom_gemm = GEMMWorkload(
        matrix_a_shape=(2048, 2048),
        matrix_b_shape=(2048, 2048),
        precision="fp32",
        algorithm="winograd",
        optimization_level=3
    )

Creating CNN Workloads
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create CNN workload for image classification
    cnn_workload = WorkloadManager.create_cnn_workload(
        model_type="resnet50",
        input_shape=(224, 224, 3),
        batch_size=32,
        precision="fp16"
    )

    # Create CNN workload for object detection
    detection_workload = WorkloadManager.create_cnn_workload(
        model_type="yolov5",
        input_shape=(640, 640, 3),
        batch_size=16,
        precision="fp16"
    )

Creating Transformer Workloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create transformer workload for NLP
    transformer_workload = WorkloadManager.create_transformer_workload(
        model_type="bert_base",
        sequence_length=512,
        batch_size=16,
        precision="fp16"
    )

    # Create GPT workload for text generation
    gpt_workload = WorkloadManager.create_transformer_workload(
        model_type="gpt2",
        sequence_length=1024,
        batch_size=8,
        precision="fp16"
    )

Creating Medical Workloads
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.workloads.medical import MedicalWorkload

    # Create medical imaging workload
    medical_imaging = MedicalWorkload(
        workload_type="medical_imaging",
        model_type="unet",
        input_shape=(512, 512, 1),
        batch_size=8,
        precision="fp16",
        hipaa_compliant=True,
        encryption_enabled=True
    )

    # Create diagnostics workload
    diagnostics = MedicalWorkload(
        workload_type="diagnostics",
        model_type="diagnostic_classifier",
        input_shape=(224, 224, 3),
        batch_size=16,
        precision="fp16",
        audit_logging=True
    )

Workload Configuration
----------------------

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configure workload for performance
    performance_config = {
        "optimization_level": 3,
        "use_tensor_cores": True,
        "memory_layout": "nhwc",
        "batch_size": 32,
        "precision": "fp16"
    }

    workload = WorkloadManager.create_workload(
        workload_type="gemm",
        config=performance_config
    )

Power Optimization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configure workload for power efficiency
    power_config = {
        "optimization_level": 1,
        "power_mode": "low_power",
        "precision": "int8",
        "batch_size": 16,
        "frequency_scaling": True
    }

    workload = WorkloadManager.create_workload(
        workload_type="cnn",
        config=power_config
    )

Medical Compliance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configure workload for medical compliance
    medical_config = {
        "hipaa_compliant": True,
        "data_encryption": True,
        "audit_logging": True,
        "secure_processing": True,
        "compliance_validation": True
    }

    workload = WorkloadManager.create_medical_workload(
        workload_type="medical_imaging",
        config=medical_config
    )

Workload Execution
------------------

Running Workloads
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator import Accelerator

    # Initialize accelerator
    accelerator = Accelerator()

    # Run workload
    result = accelerator.run(workload)

    # Access results
    print(f"Execution time: {result.execution_time}ms")
    print(f"Throughput: {result.throughput} GOPS")
    print(f"Accuracy: {result.accuracy}")
    print(f"Power consumption: {result.power_consumption}W")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process multiple workloads
    workloads = [workload1, workload2, workload3]

    results = []
    for workload in workloads:
        result = accelerator.run(workload)
        results.append(result)

    # Parallel execution
    results = accelerator.run_parallel(workloads)

Asynchronous Execution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio

    async def run_workload_async(workload):
        result = await accelerator.run_async(workload)
        return result

    # Run multiple workloads asynchronously
    async def run_multiple_workloads():
        tasks = [run_workload_async(w) for w in workloads]
        results = await asyncio.gather(*tasks)
        return results

Monitoring and Profiling
------------------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.monitoring import WorkloadMonitor

    # Create monitor
    monitor = WorkloadMonitor(accelerator)

    # Start monitoring
    monitor.start()

    # Run workload with monitoring
    result = accelerator.run(workload)

    # Get monitoring data
    metrics = monitor.get_metrics()
    print(f"Peak memory usage: {metrics.peak_memory_usage}MB")
    print(f"Average utilization: {metrics.average_utilization}%")

Profiling
~~~~~~~~~

.. code-block:: python

    from open_accelerator.profiling import WorkloadProfiler

    # Create profiler
    profiler = WorkloadProfiler(accelerator)

    # Profile workload execution
    profile_data = profiler.profile(workload)

    # Generate profiling report
    profiler.generate_report(profile_data, output_file="profile_report.html")

Error Handling
--------------

Workload Exceptions
~~~~~~~~~~~~~~~~~~

.. autoexception:: open_accelerator.workloads.base.WorkloadError
.. autoexception:: open_accelerator.workloads.gemm.GEMMError
.. autoexception:: open_accelerator.workloads.medical.MedicalWorkloadError

Exception Handling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.workloads.base import WorkloadError

    try:
        result = accelerator.run(workload)
    except WorkloadError as e:
        print(f"Workload execution failed: {e}")
        # Handle error appropriately

    # Retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = accelerator.run(workload)
            break
        except WorkloadError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")

Best Practices
--------------

Workload Design
~~~~~~~~~~~~~~~

1. **Batch Size**: Choose optimal batch size for your hardware
2. **Precision**: Use appropriate precision for your accuracy requirements
3. **Memory Layout**: Consider memory access patterns
4. **Optimization Level**: Balance compilation time with performance
5. **Resource Allocation**: Pre-allocate resources when possible

Performance Tuning
~~~~~~~~~~~~~~~~~~

1. **Profiling**: Always profile your workloads before optimization
2. **Memory Management**: Minimize memory allocations during execution
3. **Parallelization**: Use parallel execution for independent workloads
4. **Caching**: Cache frequently used models and data
5. **Monitoring**: Monitor performance metrics in production

Medical Compliance
~~~~~~~~~~~~~~~~~~

1. **Data Encryption**: Always encrypt sensitive medical data
2. **Audit Logging**: Enable comprehensive audit logging
3. **Access Control**: Implement proper access controls
4. **Compliance Validation**: Regularly validate compliance
5. **Security Updates**: Keep security components updated
