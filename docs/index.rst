OpenAccelerator Documentation
==========================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

OpenAccelerator is a high-performance accelerator simulation framework designed for AI/ML workloads with specialized support for medical applications. Built with FastAPI integration, OpenAI Agents SDK support, and comprehensive testing automation.

Author: Nik Jois <nikjois@llamasearch.ai>

Features
--------

* **High-Performance Simulation**: Systolic array and memory hierarchy modeling
* **Medical AI Integration**: HIPAA-compliant medical workload processing
* **FastAPI REST API**: Complete API endpoints for accelerator management
* **OpenAI Agents SDK**: Seamless AI agent integration
* **Comprehensive Testing**: Automated testing with 90% coverage requirement
* **Docker Support**: Full containerization with docker-compose
* **Performance Monitoring**: Real-time benchmarking and optimization
* **Security First**: Comprehensive security scanning and compliance validation

Quick Start
-----------

Installation::

    pip install open-accelerator

Basic Usage::

    from open_accelerator import Accelerator, WorkloadManager
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create and run workload
    workload = WorkloadManager.create_gemm_workload(
        matrix_size=1024,
        precision="fp16"
    )
    
    result = accelerator.run(workload)
    print(f"Performance: {result.performance_metrics}")

API Server::

    # Start FastAPI server
    uvicorn open_accelerator.api.main:app --reload
    
    # Access API documentation at http://localhost:8000/docs

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   medical_guide
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   testing
   benchmarking
   docker_guide

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/medical_workflows
   examples/performance_optimization
   examples/agent_integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/accelerator
   api/workloads
   api/medical
   api/agents
   api/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/getting_started
   tutorials/medical_ai_workflows
   tutorials/custom_workloads
   tutorials/performance_tuning
   tutorials/agent_development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 