Testing Guide
=============

This guide describes the testing strategy for OpenAccelerator.

Test Suite
----------

* **Unit tests** in `tests/` verify individual components.
* **Integration tests** ensure modules work together.
* **Benchmark tests** (optional) measure performance regressions.

Running Tests
-------------

.. code-block:: bash

   pytest -n auto -v

Coverage
--------

We enforce 90% coverage via `pytest-cov` and CI.

Static Analysis
---------------

* `ruff` for linting
* `mypy` for type checking

CI Pipeline
-----------

All tests run automatically on GitHub Actions across Python versions 3.8–3.11.
