Agents API
==========

This module provides OpenAI Agents SDK integration for OpenAccelerator, enabling AI-powered optimization, monitoring, and decision-making capabilities.

Agent Management
----------------

.. automodule:: open_accelerator.ai.agents
   :members:
   :undoc-members:
   :show-inheritance:

Agent Manager
~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.agents.AgentManager
   :members:
   :undoc-members:
   :show-inheritance:

Base Agent
~~~~~~~~~~

.. autoclass:: open_accelerator.ai.agents.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:

Workload Optimization Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.agents.WorkloadOptimizationAgent
   :members:
   :undoc-members:
   :show-inheritance:

Performance Monitoring Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.agents.PerformanceMonitoringAgent
   :members:
   :undoc-members:
   :show-inheritance:

Medical AI Agent
~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.agents.MedicalAIAgent
   :members:
   :undoc-members:
   :show-inheritance:

Compound AI Systems
-------------------

.. automodule:: open_accelerator.ai.compound_ai
   :members:
   :undoc-members:
   :show-inheritance:

Compound AI Manager
~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.compound_ai.CompoundAIManager
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Agent System
~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.compound_ai.MultiAgentSystem
   :members:
   :undoc-members:
   :show-inheritance:

Reasoning Chains
----------------

.. automodule:: open_accelerator.ai.reasoning_chains
   :members:
   :undoc-members:
   :show-inheritance:

Reasoning Chain Manager
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.reasoning_chains.ReasoningChainManager
   :members:
   :undoc-members:
   :show-inheritance:

Chain of Thought
~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.reasoning_chains.ChainOfThought
   :members:
   :undoc-members:
   :show-inheritance:

Tree of Thought
~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.reasoning_chains.TreeOfThought
   :members:
   :undoc-members:
   :show-inheritance:

Model Registry
--------------

.. automodule:: open_accelerator.ai.model_registry
   :members:
   :undoc-members:
   :show-inheritance:

Model Registry Manager
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.model_registry.ModelRegistryManager
   :members:
   :undoc-members:
   :show-inheritance:

Model Metadata
~~~~~~~~~~~~~~

.. autoclass:: open_accelerator.ai.model_registry.ModelMetadata
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Agent Setup
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.agents import AgentManager
    from openai import OpenAI

    # Initialize OpenAI client
    openai_client = OpenAI(api_key="your-api-key")

    # Create agent manager
    agent_manager = AgentManager(
        openai_client=openai_client,
        accelerator=accelerator
    )

    # Create a workload optimization agent
    optimizer_agent = agent_manager.create_agent(
        name="workload_optimizer",
        agent_type="workload_optimization",
        instructions="Optimize workload parameters for best performance on the accelerator",
        tools=["performance_analyzer", "memory_optimizer", "power_optimizer"]
    )

Workload Optimization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.agents import WorkloadOptimizationAgent

    # Create workload optimization agent
    optimizer = WorkloadOptimizationAgent(
        openai_client=openai_client,
        accelerator=accelerator,
        optimization_strategy="performance_first"
    )

    # Optimize workload parameters
    original_workload = WorkloadManager.create_gemm_workload(
        matrix_a_shape=(1024, 1024),
        matrix_b_shape=(1024, 1024),
        precision="fp32"
    )

    optimized_workload = optimizer.optimize_workload(
        workload=original_workload,
        optimization_goals=["minimize_latency", "maximize_throughput"]
    )

    print(f"Original precision: {original_workload.precision}")
    print(f"Optimized precision: {optimized_workload.precision}")
    print(f"Optimization recommendations: {optimizer.get_recommendations()}")

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.agents import PerformanceMonitoringAgent

    # Create performance monitoring agent
    monitor_agent = PerformanceMonitoringAgent(
        openai_client=openai_client,
        accelerator=accelerator,
        monitoring_interval=1.0  # seconds
    )

    # Start monitoring
    monitor_agent.start_monitoring()

    # Run workload with AI monitoring
    result = accelerator.run(workload)

    # Get AI-generated performance insights
    insights = monitor_agent.get_performance_insights()
    print(f"Performance insights: {insights.summary}")
    print(f"Optimization suggestions: {insights.suggestions}")
    print(f"Anomalies detected: {insights.anomalies}")

Medical AI Agent
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.agents import MedicalAIAgent
    from open_accelerator.medical.compliance import ComplianceManager

    # Create compliance manager
    compliance_manager = ComplianceManager(hipaa_enabled=True)

    # Create medical AI agent
    medical_agent = MedicalAIAgent(
        openai_client=openai_client,
        accelerator=accelerator,
        compliance_manager=compliance_manager,
        medical_knowledge_base="medical_kb_v2.1"
    )

    # Perform medical diagnosis
    diagnosis = medical_agent.diagnose(
        patient_data={
            "age": 45,
            "gender": "female",
            "symptoms": ["chest_pain", "shortness_of_breath"],
            "medical_history": ["hypertension", "diabetes"]
        },
        imaging_data="chest_xray.dicom"
    )

    print(f"Primary diagnosis: {diagnosis.primary_diagnosis}")
    print(f"Confidence: {diagnosis.confidence}")
    print(f"Differential diagnoses: {diagnosis.differential_diagnoses}")
    print(f"Recommendations: {diagnosis.recommendations}")

Compound AI Systems
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.compound_ai import CompoundAIManager

    # Create compound AI manager
    compound_ai = CompoundAIManager(
        openai_client=openai_client,
        accelerator=accelerator
    )

    # Create multi-agent system
    multi_agent_system = compound_ai.create_multi_agent_system(
        agents=[
            {
                "name": "data_preprocessor",
                "type": "data_processing",
                "instructions": "Preprocess and validate input data"
            },
            {
                "name": "model_selector",
                "type": "model_selection",
                "instructions": "Select optimal model for the workload"
            },
            {
                "name": "performance_optimizer",
                "type": "optimization",
                "instructions": "Optimize performance parameters"
            },
            {
                "name": "result_validator",
                "type": "validation",
                "instructions": "Validate and post-process results"
            }
        ]
    )

    # Execute compound AI workflow
    result = multi_agent_system.execute_workflow(
        input_data=workload_data,
        workflow_type="end_to_end_optimization"
    )

Reasoning Chains
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.reasoning_chains import ReasoningChainManager

    # Create reasoning chain manager
    reasoning_manager = ReasoningChainManager(
        openai_client=openai_client,
        accelerator=accelerator
    )

    # Create chain of thought reasoning
    cot_chain = reasoning_manager.create_chain_of_thought(
        problem="Optimize memory allocation for large matrix multiplication",
        context={
            "matrix_size": (4096, 4096),
            "available_memory": "8GB",
            "precision": "fp16"
        }
    )

    # Execute reasoning chain
    reasoning_result = cot_chain.execute()
    print(f"Reasoning steps: {reasoning_result.steps}")
    print(f"Final solution: {reasoning_result.solution}")
    print(f"Confidence: {reasoning_result.confidence}")

Agent Tools and Functions
-------------------------

Performance Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.tools import PerformanceAnalyzerTool

    # Create performance analyzer tool
    perf_tool = PerformanceAnalyzerTool(accelerator=accelerator)

    # Tool function for agents
    def analyze_performance(workload_params: dict) -> dict:
        return perf_tool.analyze(workload_params)

    # Register tool with agent
    agent_manager.register_tool(
        name="performance_analyzer",
        function=analyze_performance,
        description="Analyze workload performance metrics"
    )

Memory Optimization Tools
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.tools import MemoryOptimizerTool

    # Create memory optimizer tool
    memory_tool = MemoryOptimizerTool(accelerator=accelerator)

    # Tool function for agents
    def optimize_memory(workload_params: dict) -> dict:
        return memory_tool.optimize(workload_params)

    # Register tool with agent
    agent_manager.register_tool(
        name="memory_optimizer",
        function=optimize_memory,
        description="Optimize memory allocation and usage"
    )

Power Management Tools
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.tools import PowerOptimizerTool

    # Create power optimizer tool
    power_tool = PowerOptimizerTool(accelerator=accelerator)

    # Tool function for agents
    def optimize_power(workload_params: dict) -> dict:
        return power_tool.optimize(workload_params)

    # Register tool with agent
    agent_manager.register_tool(
        name="power_optimizer",
        function=optimize_power,
        description="Optimize power consumption and thermal management"
    )

Model Registry Integration
-------------------------

Model Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.model_registry import ModelRegistryManager

    # Create model registry manager
    model_registry = ModelRegistryManager(
        storage_backend="s3",
        metadata_store="postgresql"
    )

    # Register a model
    model_metadata = model_registry.register_model(
        name="gemm_optimizer_v2.1",
        model_type="optimization",
        framework="pytorch",
        version="2.1.0",
        performance_metrics={
            "accuracy": 0.95,
            "latency": 12.5,
            "throughput": 850
        }
    )

    # Use model with agent
    agent = agent_manager.create_agent(
        name="model_based_optimizer",
        agent_type="optimization",
        model_id=model_metadata.id
    )

Model Versioning
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create new model version
    new_version = model_registry.create_version(
        model_id="gemm_optimizer_v2.1",
        version="2.2.0",
        changes=["improved_accuracy", "reduced_latency"],
        performance_metrics={
            "accuracy": 0.97,
            "latency": 10.8,
            "throughput": 920
        }
    )

    # Compare model versions
    comparison = model_registry.compare_versions(
        model_id="gemm_optimizer_v2.1",
        version_a="2.1.0",
        version_b="2.2.0"
    )

    print(f"Performance improvement: {comparison.improvement_summary}")

Advanced Features
-----------------

Multimodal Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.multimodal_processor import MultimodalProcessor

    # Create multimodal processor
    multimodal = MultimodalProcessor(
        openai_client=openai_client,
        accelerator=accelerator
    )

    # Process multimodal medical data
    result = multimodal.process_medical_data(
        text_data="Patient presents with chest pain and shortness of breath",
        image_data="chest_xray.dicom",
        structured_data={
            "vitals": {"heart_rate": 95, "blood_pressure": "140/90"},
            "labs": {"troponin": 0.8, "bnp": 450}
        }
    )

    print(f"Multimodal analysis: {result.analysis}")
    print(f"Confidence: {result.confidence}")

Adaptive Learning
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.adaptive_learning import AdaptiveLearningManager

    # Create adaptive learning manager
    adaptive_manager = AdaptiveLearningManager(
        openai_client=openai_client,
        accelerator=accelerator
    )

    # Enable adaptive learning for agent
    adaptive_agent = adaptive_manager.create_adaptive_agent(
        base_agent=optimizer_agent,
        learning_strategy="reinforcement_learning",
        feedback_mechanism="performance_metrics"
    )

    # Agent learns from performance feedback
    for workload in workload_batch:
        result = accelerator.run(workload)
        adaptive_agent.learn_from_feedback(
            workload=workload,
            result=result,
            feedback_score=result.performance_score
        )

Error Handling
--------------

Agent Exceptions
~~~~~~~~~~~~~~~

.. autoexception:: open_accelerator.ai.agents.AgentError
.. autoexception:: open_accelerator.ai.compound_ai.CompoundAIError
.. autoexception:: open_accelerator.ai.reasoning_chains.ReasoningError
.. autoexception:: open_accelerator.ai.model_registry.ModelRegistryError

Exception Handling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_accelerator.ai.agents import AgentError

    try:
        result = optimizer_agent.optimize_workload(workload)
    except AgentError as e:
        print(f"Agent error: {e}")
        # Handle agent-specific error
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle general error

    # Retry with fallback
    try:
        result = optimizer_agent.optimize_workload(workload)
    except AgentError:
        # Fallback to manual optimization
        result = manual_optimizer.optimize(workload)

Configuration
-------------

Agent Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    agent_config = {
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2000,
        "timeout": 30.0,
        "retry_attempts": 3,
        "tools": ["performance_analyzer", "memory_optimizer"],
        "instructions": "Optimize workload parameters for best performance"
    }

    agent = agent_manager.create_agent(
        name="configured_optimizer",
        config=agent_config
    )

OpenAI Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from openai import OpenAI

    # Configure OpenAI client
    openai_client = OpenAI(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        timeout=30.0,
        max_retries=3
    )

    # Use with agent manager
    agent_manager = AgentManager(
        openai_client=openai_client,
        accelerator=accelerator,
        default_model="gpt-4"
    )

Best Practices
--------------

Agent Development
~~~~~~~~~~~~~~~~

1. **Clear Instructions**: Provide clear, specific instructions to agents
2. **Tool Selection**: Choose appropriate tools for each agent's role
3. **Error Handling**: Implement robust error handling and fallbacks
4. **Performance Monitoring**: Monitor agent performance and costs
5. **Security**: Implement proper authentication and authorization

Multi-Agent Coordination
~~~~~~~~~~~~~~~~~~~~~~~

1. **Communication**: Define clear communication protocols between agents
2. **Task Distribution**: Distribute tasks based on agent capabilities
3. **Conflict Resolution**: Implement mechanisms for resolving conflicts
4. **Scalability**: Design for horizontal scaling of agent systems
5. **Monitoring**: Monitor multi-agent system performance

Medical AI Safety
~~~~~~~~~~~~~~~~

1. **Compliance**: Ensure all medical AI agents comply with regulations
2. **Validation**: Validate medical AI outputs with clinical experts
3. **Transparency**: Provide explainable AI for medical decisions
4. **Bias Testing**: Regularly test for bias in medical AI systems
5. **Audit Trail**: Maintain comprehensive audit trails for medical AI

Integration Examples
-------------------

FastAPI Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fastapi import FastAPI, HTTPException
    from open_accelerator.ai.agents import AgentManager

    app = FastAPI()

    @app.post("/agents/optimize")
    async def optimize_workload(request: OptimizationRequest):
        try:
            result = optimizer_agent.optimize_workload(request.workload)
            return {"optimized_workload": result, "success": True}
        except AgentError as e:
            raise HTTPException(status_code=500, detail=str(e))

WebSocket Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fastapi import WebSocket

    @app.websocket("/agents/monitor")
    async def monitor_performance(websocket: WebSocket):
        await websocket.accept()
        monitor_agent.start_monitoring()

        while True:
            insights = monitor_agent.get_real_time_insights()
            await websocket.send_json(insights)
            await asyncio.sleep(1.0)

Streaming Responses
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fastapi.responses import StreamingResponse

    @app.get("/agents/reasoning/stream")
    async def stream_reasoning(problem: str):
        def generate_reasoning():
            reasoning_chain = reasoning_manager.create_chain_of_thought(problem)
            for step in reasoning_chain.execute_streaming():
                yield f"data: {step}\n\n"

        return StreamingResponse(generate_reasoning(), media_type="text/plain")
