Docker Guide
===========

This guide explains how to build and run OpenAccelerator using Docker.

Prerequisites
-------------

* Docker 20.10+
* (Optional) Docker Compose 1.29+

Building the Image
------------------

.. code-block:: bash

   docker build -t openaccelerator:latest .

Running the Container
---------------------

.. code-block:: bash

   docker run -p 8000:8000 openaccelerator:latest

Using Docker Compose
--------------------

.. code-block:: bash

   docker-compose up -d

The FastAPI server will be available at http://localhost:8000.

Development Workflow
--------------------

Mount source code for live reload:

.. code-block:: bash

   docker run -it -p 8000:8000 -v $PWD:/code openaccelerator:latest
