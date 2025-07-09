Accelerator API
===============

This module provides the core accelerator functionality for OpenAccelerator.

Core Accelerator
-----------------

.. automodule:: open_accelerator.core.accelerator
   :members:
   :undoc-members:
   :show-inheritance:

Accelerator Class
~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.accelerator.Accelerator
   :members:
   :undoc-members:
   :show-inheritance:

Memory Management
-----------------

.. automodule:: open_accelerator.core.memory
   :members:
   :undoc-members:
   :show-inheritance:

Memory Hierarchy
~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.memory.MemoryHierarchy
   :members:
   :undoc-members:
   :show-inheritance:

Processing Elements
-------------------

.. automodule:: open_accelerator.core.pe
   :members:
   :undoc-members:
   :show-inheritance:

PE Array
~~~~~~~~

.. autoclass:: open_accelerator.core.pe.PEArray
   :members:
   :undoc-members:
   :show-inheritance:

Processing Element
~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.pe.ProcessingElement
   :members:
   :undoc-members:
   :show-inheritance:

Systolic Array
--------------

.. automodule:: open_accelerator.core.systolic_array
   :members:
   :undoc-members:
   :show-inheritance:

Systolic Array Class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.systolic_array.SystolicArray
   :members:
   :undoc-members:
   :show-inheritance:

Power Management
----------------

.. automodule:: open_accelerator.core.power_management
   :members:
   :undoc-members:
   :show-inheritance:

Power Manager
~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.power_management.PowerManager
   :members:
   :undoc-members:
   :show-inheritance:

Power Monitor
~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.power.PowerMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Security
--------

.. automodule:: open_accelerator.core.security
   :members:
   :undoc-members:
   :show-inheritance:

Security Manager
~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.security.SecurityManager
   :members:
   :undoc-members:
   :show-inheritance:

Reliability
-----------

.. automodule:: open_accelerator.core.reliability
   :members:
   :undoc-members:
   :show-inheritance:

Reliability Manager
~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.core.reliability.ReliabilityManager
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Accelerator Usage
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.core import Accelerator, SystemConfig
    
    # Initialize accelerator with default configuration
    accelerator = Accelerator()
    
    # Custom configuration
    config = SystemConfig(
        pe_array_size=(256, 256),
        memory_size="4GB",
        precision="fp16"
    )
    accelerator = Accelerator(config=config)
    
    # Get accelerator status
    status = accelerator.get_status()
    print(f"Accelerator ready: {status.ready}")
    print(f"Utilization: {status.utilization}%")

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.core.memory import MemoryHierarchy
    
    # Create memory hierarchy
    memory = MemoryHierarchy(
        l1_cache_size="128KB",
        l2_cache_size="2MB",
        main_memory_size="8GB"
    )
    
    # Allocate memory
    buffer = memory.allocate(size="1MB", type="weight")
    
    # Memory statistics
    stats = memory.get_statistics()
    print(f"Memory utilization: {stats.utilization}%")

Power Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.core.power_management import PowerManager
    
    # Initialize power manager
    power_mgr = PowerManager(
        power_budget=100,  # watts
        thermal_limit=85   # celsius
    )
    
    # Monitor power consumption
    power_stats = power_mgr.get_power_stats()
    print(f"Current power: {power_stats.current_power}W")
    print(f"Temperature: {power_stats.temperature}°C")
    
    # Set power mode
    power_mgr.set_power_mode("high_performance")

Security Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.core.security import SecurityManager
    
    # Initialize security manager
    security = SecurityManager(
        encryption_enabled=True,
        secure_boot=True
    )
    
    # Encrypt data
    encrypted_data = security.encrypt(data, key="user_key")
    
    # Validate integrity
    is_valid = security.validate_integrity(data, checksum)

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. autoexception:: open_accelerator.core.accelerator.AcceleratorError
.. autoexception:: open_accelerator.core.memory.MemoryError
.. autoexception:: open_accelerator.core.power.PowerError
.. autoexception:: open_accelerator.core.security.SecurityError

Performance Considerations
-------------------------

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Memory Allocation**: Pre-allocate buffers to reduce overhead
2. **Batch Processing**: Use larger batch sizes for better efficiency
3. **Precision**: Use appropriate precision (fp16/int8) for your workload
4. **Power Management**: Configure power modes based on requirements
5. **Parallel Processing**: Leverage multiple PE arrays for parallel execution

Benchmarking
~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.benchmarks import AcceleratorBenchmark
    
    # Create benchmark
    benchmark = AcceleratorBenchmark(accelerator)
    
    # Run performance tests
    results = benchmark.run_performance_tests()
    
    # Generate report
    benchmark.generate_report(results, output_file="performance_report.html") 