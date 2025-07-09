Getting Started Tutorial
========================

This tutorial will guide you through the basics of OpenAccelerator, from installation to running your first workload.

Prerequisites
-------------

Before starting, ensure you have:

* Python 3.8 or higher
* pip package manager
* Basic knowledge of Python programming
* (Optional) Docker for containerized deployment

Step 1: Installation
--------------------

Install OpenAccelerator using pip:

.. code-block:: bash

    pip install open-accelerator

For development installation:

.. code-block:: bash

    git clone https://github.com/llamasearch/OpenAccelerator.git
    cd OpenAccelerator
    pip install -e ".[dev]"

Step 2: Basic Configuration
---------------------------

Create a basic configuration file:

.. code-block:: python

    # config.py
    from open_accelerator.core import SystemConfig

    # Basic configuration
    config = SystemConfig(
        pe_array_size=(128, 128),
        memory_hierarchy={
            "l1_cache": "64KB",
            "l2_cache": "1MB",
            "main_memory": "4GB"
        },
        precision="fp16",
        power_management=True
    )

Step 3: Initialize the Accelerator
----------------------------------

.. code-block:: python

    from open_accelerator import Accelerator
    from open_accelerator.core import SystemConfig

    # Initialize with default configuration
    accelerator = Accelerator()

    # Or with custom configuration
    accelerator = Accelerator(config=config)

    # Check accelerator status
    status = accelerator.get_status()
    print(f"Accelerator ready: {status.ready}")
    print(f"Configuration: {status.config}")

Step 4: Create Your First Workload
----------------------------------

Let's create a simple GEMM (General Matrix Multiplication) workload:

.. code-block:: python

    from open_accelerator.workloads import WorkloadManager

    # Create a GEMM workload
    workload = WorkloadManager.create_gemm_workload(
        matrix_a_shape=(1024, 1024),
        matrix_b_shape=(1024, 1024),
        precision="fp16"
    )

    print(f"Workload created: {workload.name}")
    print(f"Matrix A shape: {workload.matrix_a_shape}")
    print(f"Matrix B shape: {workload.matrix_b_shape}")
    print(f"Precision: {workload.precision}")

Step 5: Run the Workload
------------------------

.. code-block:: python

    # Run the workload
    result = accelerator.run(workload)

    # Display results
    print(f"Execution time: {result.execution_time}ms")
    print(f"Throughput: {result.throughput} GOPS")
    print(f"Power consumption: {result.power_consumption}W")
    print(f"Memory usage: {result.memory_usage}MB")

Step 6: Performance Analysis
----------------------------

.. code-block:: python

    from open_accelerator.analysis import PerformanceAnalyzer

    # Create performance analyzer
    analyzer = PerformanceAnalyzer(accelerator)

    # Analyze the result
    analysis = analyzer.analyze(result)

    print(f"Performance score: {analysis.performance_score}")
    print(f"Efficiency: {analysis.efficiency}%")
    print(f"Bottlenecks: {analysis.bottlenecks}")
    print(f"Recommendations: {analysis.recommendations}")

Step 7: Working with Different Workload Types
---------------------------------------------

CNN Workload
~~~~~~~~~~~~

.. code-block:: python

    # Create CNN workload for image classification
    cnn_workload = WorkloadManager.create_cnn_workload(
        model_type="resnet50",
        input_shape=(224, 224, 3),
        batch_size=32,
        precision="fp16"
    )

    cnn_result = accelerator.run(cnn_workload)
    print(f"CNN inference time: {cnn_result.execution_time}ms")

Transformer Workload
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create transformer workload for NLP
    transformer_workload = WorkloadManager.create_transformer_workload(
        model_type="bert_base",
        sequence_length=512,
        batch_size=16,
        precision="fp16"
    )

    transformer_result = accelerator.run(transformer_workload)
    print(f"Transformer inference time: {transformer_result.execution_time}ms")

Step 8: Configuration Optimization
----------------------------------

.. code-block:: python

    # Try different configurations
    configurations = [
        SystemConfig(pe_array_size=(64, 64), precision="fp32"),
        SystemConfig(pe_array_size=(128, 128), precision="fp16"),
        SystemConfig(pe_array_size=(256, 256), precision="int8")
    ]

    best_config = None
    best_performance = 0

    for config in configurations:
        accelerator = Accelerator(config=config)
        result = accelerator.run(workload)

        if result.throughput > best_performance:
            best_performance = result.throughput
            best_config = config

    print(f"Best configuration: {best_config}")
    print(f"Best throughput: {best_performance} GOPS")

Step 9: Monitoring and Logging
------------------------------

.. code-block:: python

    import logging
    from open_accelerator.monitoring import SystemMonitor

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create system monitor
    monitor = SystemMonitor(accelerator)

    # Start monitoring
    monitor.start()

    # Run workload with monitoring
    logger.info("Starting workload execution")
    result = accelerator.run(workload)
    logger.info(f"Workload completed in {result.execution_time}ms")

    # Get monitoring data
    metrics = monitor.get_metrics()
    logger.info(f"Peak memory usage: {metrics.peak_memory_usage}MB")
    logger.info(f"Average power consumption: {metrics.average_power}W")

Step 10: Error Handling
-----------------------

.. code-block:: python

    from open_accelerator.core.accelerator import AcceleratorError
    from open_accelerator.workloads.base import WorkloadError

    try:
        # Attempt to run workload
        result = accelerator.run(workload)
        print(f"Success: {result.execution_time}ms")

    except WorkloadError as e:
        print(f"Workload error: {e}")
        # Handle workload-specific error

    except AcceleratorError as e:
        print(f"Accelerator error: {e}")
        # Handle accelerator-specific error

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle unexpected error

Complete Example Script
-----------------------

Here's a complete example that puts everything together:

.. code-block:: python

    #!/usr/bin/env python3
    """
    OpenAccelerator Getting Started Example

    This script demonstrates basic OpenAccelerator usage.

    Author: Nik Jois <nikjois@llamasearch.ai>
    """

    import logging
    from open_accelerator import Accelerator
    from open_accelerator.core import SystemConfig
    from open_accelerator.workloads import WorkloadManager
    from open_accelerator.analysis import PerformanceAnalyzer
    from open_accelerator.monitoring import SystemMonitor

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def main():
        """Main function demonstrating OpenAccelerator usage."""

        # Step 1: Create configuration
        config = SystemConfig(
            pe_array_size=(128, 128),
            memory_hierarchy={
                "l1_cache": "64KB",
                "l2_cache": "1MB",
                "main_memory": "4GB"
            },
            precision="fp16",
            power_management=True
        )

        # Step 2: Initialize accelerator
        accelerator = Accelerator(config=config)
        logger.info("Accelerator initialized")

        # Step 3: Create workload
        workload = WorkloadManager.create_gemm_workload(
            matrix_a_shape=(1024, 1024),
            matrix_b_shape=(1024, 1024),
            precision="fp16"
        )
        logger.info(f"Workload created: {workload.name}")

        # Step 4: Setup monitoring
        monitor = SystemMonitor(accelerator)
        monitor.start()
        logger.info("Monitoring started")

        # Step 5: Run workload
        try:
            result = accelerator.run(workload)
            logger.info(f"Workload completed successfully")

            # Step 6: Display results
            print(f"\nResults:")
            print(f"  Execution time: {result.execution_time}ms")
            print(f"  Throughput: {result.throughput} GOPS")
            print(f"  Power consumption: {result.power_consumption}W")
            print(f"  Memory usage: {result.memory_usage}MB")

            # Step 7: Performance analysis
            analyzer = PerformanceAnalyzer(accelerator)
            analysis = analyzer.analyze(result)

            print(f"\nPerformance Analysis:")
            print(f"  Performance score: {analysis.performance_score}")
            print(f"  Efficiency: {analysis.efficiency}%")
            print(f"  Bottlenecks: {', '.join(analysis.bottlenecks)}")

            # Step 8: Monitoring results
            metrics = monitor.get_metrics()
            print(f"\nMonitoring Results:")
            print(f"  Peak memory usage: {metrics.peak_memory_usage}MB")
            print(f"  Average power: {metrics.average_power}W")
            print(f"  Utilization: {metrics.utilization}%")

        except Exception as e:
            logger.error(f"Error running workload: {e}")
            return 1

        logger.info("Tutorial completed successfully")
        return 0

    if __name__ == "__main__":
        exit(main())

Running the Example
-------------------

Save the complete example as ``getting_started.py`` and run it:

.. code-block:: bash

    python getting_started.py

Expected output:

.. code-block:: text

    INFO:__main__:Accelerator initialized
    INFO:__main__:Workload created: GEMM_1024x1024_fp16
    INFO:__main__:Monitoring started
    INFO:__main__:Workload completed successfully

    Results:
      Execution time: 12.5ms
      Throughput: 8500 GOPS
      Power consumption: 75W
      Memory usage: 256MB

    Performance Analysis:
      Performance score: 85.2
      Efficiency: 92%
      Bottlenecks: memory_bandwidth

    Monitoring Results:
      Peak memory usage: 512MB
      Average power: 68W
      Utilization: 89%

    INFO:__main__:Tutorial completed successfully

Next Steps
----------

Now that you've completed the basic tutorial, you can:

1. **Explore Advanced Features**: Try different workload types and configurations
2. **Read the API Documentation**: Learn about advanced features and customization
3. **Try Medical Workflows**: Explore the medical AI capabilities
4. **Experiment with Agents**: Use OpenAI Agents for intelligent optimization
5. **Build Applications**: Create applications using the FastAPI integration

Common Issues and Solutions
--------------------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

    export PYTHONPATH="${PYTHONPATH}:/path/to/OpenAccelerator/src"

Memory Issues
~~~~~~~~~~~~~

If you run out of memory:

.. code-block:: python

    # Reduce batch size or matrix size
    smaller_workload = WorkloadManager.create_gemm_workload(
        matrix_a_shape=(512, 512),
        matrix_b_shape=(512, 512),
        precision="fp16"
    )

Performance Issues
~~~~~~~~~~~~~~~~~

If performance is poor:

.. code-block:: python

    # Try different precision
    config = SystemConfig(
        pe_array_size=(256, 256),
        precision="int8"  # Lower precision, higher performance
    )

Getting Help
------------

If you need help:

* Check the API documentation for detailed reference
* Visit the GitHub repository for issues and discussions
* Contact support at nikjois@llamasearch.ai
* Join the community forums for user discussions

Congratulations! You've completed the OpenAccelerator getting started tutorial. You now have the foundation to explore more advanced features and build sophisticated accelerator applications.
