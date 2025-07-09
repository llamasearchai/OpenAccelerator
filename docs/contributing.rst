Contributing Guide
==================

Thank you for considering contributing to OpenAccelerator!

How to Contribute
-----------------

1. Fork the repository and create your branch.
2. Write tests and ensure `pytest` passes.
3. Run linters: `ruff`, `black`, `isort`, and `mypy`.
4. Submit a pull request describing your changes.

Code Style
----------

We follow **PEP 8** with the following tools:

* `ruff` for linting
* `black` for formatting
* `isort` for import ordering
* `mypy` for static typing

Documentation
-------------

All public APIs must include docstrings and the documentation must build without warnings using `make docs`.

Testing
-------

* Write unit tests in the `tests/` directory.
* Keep coverage ≥ 90%.

Issue Reporting
---------------

Please include reproduction steps and environment details.
