Benchmarking Guide
==================

OpenAccelerator includes benchmarking tools to measure performance.

Overview
--------

Benchmark tests evaluate:

* Cycle counts
* MAC operations per cycle
* Energy consumption

Running Benchmarks
------------------

.. code-block:: bash

   pytest tests/ --benchmark-only

Interpreting Results
--------------------

* **total_cycles** – lower is better
* **macs_per_cycle** – higher is better
* **energy_consumed** – lower is better

Visualizations
--------------

Use `open_accelerator.visualization` to generate charts.
