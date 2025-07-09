Custom Workloads Tutorial
=========================

In this tutorial you learn how to implement a new workload and execute it.

Steps
-----

1. Subclass :class:`open_accelerator.workloads.base.MLWorkload`.
2. Implement the `prepare`, `get_input_data`, `get_expected_output`, and `validate_output` methods.
3. Register the workload with your simulation script.
4. Run the simulator and analyse results.
