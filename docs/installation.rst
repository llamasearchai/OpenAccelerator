Installation Guide
==================

System Requirements
-------------------

* Python 3.8 or higher
* pip package manager
* Git (for development installation)
* Docker (optional, for containerized deployment)

Supported Platforms
-------------------

* Linux (Ubuntu 18.04+, CentOS 7+)
* macOS (10.14+)
* Windows 10+ (with WSL2 recommended)

Installation Methods
--------------------

PyPI Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable version from PyPI::

    pip install open-accelerator

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development or latest features::

    git clone https://github.com/llamasearch/OpenAccelerator.git
    cd OpenAccelerator
    pip install -e ".[dev]"

This installs OpenAccelerator in development mode with additional dependencies for testing, linting, and documentation.

Docker Installation
~~~~~~~~~~~~~~~~~~~

Using Docker Compose::

    git clone https://github.com/llamasearch/OpenAccelerator.git
    cd OpenAccelerator
    docker-compose up -d

Or using Docker directly::

    docker run -p 8000:8000 nikjois/openaccelerator:latest

Dependency Groups
----------------

The project supports multiple dependency groups:

* ``core``: Essential dependencies for basic functionality
* ``dev``: Development tools (pytest, black, ruff, mypy)
* ``docs``: Documentation generation (Sphinx, themes)
* ``benchmark``: Performance testing tools
* ``security``: Security scanning and compliance tools
* ``medical``: Medical AI specific dependencies

Install specific groups::

    pip install "open-accelerator[dev,docs]"

Virtual Environment Setup
-------------------------

It's recommended to use a virtual environment::

    python -m venv openaccel-env
    source openaccel-env/bin/activate  # On Windows: openaccel-env\Scripts\activate
    pip install open-accelerator

Configuration
-------------

Create a configuration file::

    mkdir ~/.openaccelerator
    cat > ~/.openaccelerator/config.yaml << EOF
    accelerator:
      type: "systolic_array"
      dimensions: [256, 256]
      memory_hierarchy:
        l1_cache: 64KB
        l2_cache: 1MB
        main_memory: 4GB

    medical:
      compliance_mode: true
      hipaa_logging: true
      encryption_enabled: true

    api:
      host: "localhost"
      port: 8000
      debug: false
    EOF

Environment Variables
~~~~~~~~~~~~~~~~~~~~

Set environment variables for configuration::

    export OPENACCEL_CONFIG_PATH=~/.openaccelerator/config.yaml
    export OPENACCEL_LOG_LEVEL=INFO
    export OPENACCEL_MEDICAL_MODE=true

Verification
-----------

Verify installation::

    python -c "import open_accelerator; print(open_accelerator.__version__)"

    # Test basic functionality
    python -c "
    from open_accelerator import Accelerator
    acc = Accelerator()
    print('OpenAccelerator installed successfully!')
    "

Run the test suite::

    pytest tests/ -v

Start the API server::

    uvicorn open_accelerator.api.main:app --reload

Access the API documentation at http://localhost:8000/docs

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Import Error**: Ensure Python path is correctly set::

    export PYTHONPATH="${PYTHONPATH}:/path/to/OpenAccelerator/src"

**Permission Errors**: Use virtual environment or user installation::

    pip install --user open-accelerator

**Docker Issues**: Ensure Docker daemon is running::

    systemctl start docker  # Linux
    # or
    open -a Docker  # macOS

**Medical Compliance**: For medical workloads, ensure compliance mode::

    export OPENACCEL_MEDICAL_MODE=true
    export OPENACCEL_HIPAA_LOGGING=true

Getting Help
-----------

* Documentation: https://openaccelerator.readthedocs.io/
* Issues: https://github.com/llamasearch/OpenAccelerator/issues
* Email: nikjois@llamasearch.ai

For medical applications, please review the medical guide for additional compliance requirements.
