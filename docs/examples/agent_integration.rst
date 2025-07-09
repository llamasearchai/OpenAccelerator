OpenAI Agents Integration Example
=================================

This example shows how to use the `AgentManager` to optimise workloads.

.. code-block:: python

    from open_accelerator.ai.agents import AgentManager
    from openai import OpenAI
    from open_accelerator import Accelerator, WorkloadManager

    accelerator = Accelerator()
    workload = WorkloadManager.create_gemm_workload(matrix_size=1024)

    client = OpenAI(api_key="sk-***")
    manager = AgentManager(openai_client=client, accelerator=accelerator)

    optimizer = manager.create_agent(
        name="optimizer",
        agent_type="workload_optimization",
        instructions="Optimise workload parameters",
        tools=["performance_analyzer", "memory_optimizer"]
    )

    optimised = optimizer.optimize_workload(workload)
    result = accelerator.run(optimised)
    print(result.performance_metrics)
