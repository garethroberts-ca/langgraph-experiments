#!/usr/bin/env python3
"""
LangGraph Advanced Example 33: Subgraph Composition
====================================================

This example demonstrates LangGraph's subgraph capabilities for modular design:
- Nested graphs (parent -> child -> grandchild)
- Shared state schemas between parent and child
- Different schemas with state transformation
- Multi-agent systems using subgraphs
- Streaming from subgraphs

Subgraphs enable building complex, modular systems where each component
can be developed, tested, and maintained independently.

Author: LangGraph Examples
"""

import operator
from typing import Annotated, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# =============================================================================
# Helper Utilities
# =============================================================================

def print_banner(title: str) -> None:
    """Print a formatted section banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def log(message: str, indent: int = 0) -> None:
    """Print a log message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}→ {message}")


# Reducer function for list concatenation
def reduce_list(left: list | None, right: list | None) -> list:
    """Merge two lists, handling None values."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


# =============================================================================
# Demo 1: Basic Subgraph with Shared State
# =============================================================================

def demo_basic_subgraph():
    """
    Basic subgraph where parent and child share state schema.
    
    Demonstrates:
    - Adding compiled subgraph as a node
    - Shared state keys between graphs
    - Simple parent-child communication
    """
    print_banner("Demo 1: Basic Subgraph with Shared State")
    
    # Shared state between parent and child
    class SharedState(TypedDict):
        message: str
        steps: Annotated[list[str], reduce_list]
    
    # Child subgraph
    def child_step_1(state: SharedState) -> dict:
        log("Child: Step 1 - Processing", indent=1)
        return {"steps": ["child_step_1"]}
    
    def child_step_2(state: SharedState) -> dict:
        log("Child: Step 2 - Transforming", indent=1)
        return {
            "message": state["message"].upper(),
            "steps": ["child_step_2"]
        }
    
    child_builder = StateGraph(SharedState)
    child_builder.add_node("child_1", child_step_1)
    child_builder.add_node("child_2", child_step_2)
    child_builder.add_edge(START, "child_1")
    child_builder.add_edge("child_1", "child_2")
    child_builder.add_edge("child_2", END)
    child_subgraph = child_builder.compile()
    
    # Parent graph
    def parent_start(state: SharedState) -> dict:
        log("Parent: Starting workflow")
        return {"steps": ["parent_start"]}
    
    def parent_end(state: SharedState) -> dict:
        log("Parent: Completing workflow")
        return {
            "message": f"Final: {state['message']}",
            "steps": ["parent_end"]
        }
    
    parent_builder = StateGraph(SharedState)
    parent_builder.add_node("start", parent_start)
    parent_builder.add_node("child", child_subgraph)  # Add subgraph as node
    parent_builder.add_node("end", parent_end)
    parent_builder.add_edge(START, "start")
    parent_builder.add_edge("start", "child")
    parent_builder.add_edge("child", "end")
    parent_builder.add_edge("end", END)
    
    graph = parent_builder.compile()
    
    # Execute
    result = graph.invoke({
        "message": "hello world",
        "steps": []
    })
    
    log(f"\nFinal message: {result['message']}")
    log(f"Execution path: {result['steps']}")


# =============================================================================
# Demo 2: Subgraph with Different Schema
# =============================================================================

def demo_different_schema():
    """
    Subgraph with different schema requiring state transformation.
    
    Demonstrates:
    - Different state schemas
    - Input/output transformation
    - Isolating subgraph concerns
    """
    print_banner("Demo 2: Subgraph with Different Schema")
    
    # Parent state
    class ParentState(TypedDict):
        user_input: str
        processed_result: str
        history: Annotated[list[str], reduce_list]
    
    # Child state (different schema)
    class ChildState(TypedDict):
        data: str
        transformed: str
    
    # Child subgraph
    def child_transform(state: ChildState) -> dict:
        log(f"Child: Transforming '{state['data']}'", indent=1)
        return {"transformed": f"[{state['data'].upper()}]"}
    
    child_builder = StateGraph(ChildState)
    child_builder.add_node("transform", child_transform)
    child_builder.add_edge(START, "transform")
    child_builder.add_edge("transform", END)
    child_subgraph = child_builder.compile()
    
    # Parent wrapper function to transform state
    def call_child_subgraph(state: ParentState) -> dict:
        """Invoke child with state transformation."""
        log("Parent: Calling child subgraph")
        
        # Transform parent state to child state
        child_input = {"data": state["user_input"], "transformed": ""}
        
        # Invoke child
        child_output = child_subgraph.invoke(child_input)
        
        # Transform child output back to parent state
        return {
            "processed_result": child_output["transformed"],
            "history": ["child_invoked"]
        }
    
    def parent_finalize(state: ParentState) -> dict:
        log("Parent: Finalizing")
        return {"history": ["finalized"]}
    
    # Parent graph
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("process", call_child_subgraph)
    parent_builder.add_node("finalize", parent_finalize)
    parent_builder.add_edge(START, "process")
    parent_builder.add_edge("process", "finalize")
    parent_builder.add_edge("finalize", END)
    
    graph = parent_builder.compile()
    
    result = graph.invoke({
        "user_input": "important data",
        "processed_result": "",
        "history": []
    })
    
    log(f"\nProcessed result: {result['processed_result']}")
    log(f"History: {result['history']}")


# =============================================================================
# Demo 3: Multi-Level Nested Subgraphs
# =============================================================================

def demo_nested_subgraphs():
    """
    Multi-level nesting: Parent -> Child -> Grandchild.
    
    Demonstrates:
    - Deep nesting of graphs
    - State propagation through levels
    - Hierarchical workflow composition
    """
    print_banner("Demo 3: Multi-Level Nested Subgraphs")
    
    # Grandchild state and graph
    class GrandchildState(TypedDict):
        value: str
        level: str
    
    def grandchild_process(state: GrandchildState) -> dict:
        log("Grandchild: Processing", indent=2)
        return {
            "value": f"Grandchild({state['value']})",
            "level": "grandchild"
        }
    
    grandchild_builder = StateGraph(GrandchildState)
    grandchild_builder.add_node("process", grandchild_process)
    grandchild_builder.add_edge(START, "process")
    grandchild_builder.add_edge("process", END)
    grandchild_graph = grandchild_builder.compile()
    
    # Child state and graph
    class ChildState(TypedDict):
        value: str
        level: str
    
    def child_pre(state: ChildState) -> dict:
        log("Child: Pre-processing", indent=1)
        return {"value": f"Child_Pre({state['value']})"}
    
    def call_grandchild(state: ChildState) -> dict:
        log("Child: Invoking grandchild", indent=1)
        gc_input = {"value": state["value"], "level": "child"}
        gc_output = grandchild_graph.invoke(gc_input)
        return {"value": gc_output["value"], "level": gc_output["level"]}
    
    def child_post(state: ChildState) -> dict:
        log("Child: Post-processing", indent=1)
        return {"value": f"Child_Post({state['value']})"}
    
    child_builder = StateGraph(ChildState)
    child_builder.add_node("pre", child_pre)
    child_builder.add_node("grandchild", call_grandchild)
    child_builder.add_node("post", child_post)
    child_builder.add_edge(START, "pre")
    child_builder.add_edge("pre", "grandchild")
    child_builder.add_edge("grandchild", "post")
    child_builder.add_edge("post", END)
    child_graph = child_builder.compile()
    
    # Parent state and graph
    class ParentState(TypedDict):
        value: str
        level: str
    
    def parent_init(state: ParentState) -> dict:
        log("Parent: Initializing")
        return {"value": f"Parent_Init({state['value']})"}
    
    def call_child(state: ParentState) -> dict:
        log("Parent: Invoking child")
        child_input = {"value": state["value"], "level": "parent"}
        child_output = child_graph.invoke(child_input)
        return {"value": child_output["value"], "level": child_output["level"]}
    
    def parent_finalize(state: ParentState) -> dict:
        log("Parent: Finalizing")
        return {"value": f"Parent_Final({state['value']})"}
    
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("init", parent_init)
    parent_builder.add_node("child", call_child)
    parent_builder.add_node("finalize", parent_finalize)
    parent_builder.add_edge(START, "init")
    parent_builder.add_edge("init", "child")
    parent_builder.add_edge("child", "finalize")
    parent_builder.add_edge("finalize", END)
    
    graph = parent_builder.compile()
    
    result = graph.invoke({"value": "input", "level": "start"})
    
    log(f"\nFinal value: {result['value']}")
    log(f"Last level: {result['level']}")


# =============================================================================
# Demo 4: Multi-Agent System with Subgraphs
# =============================================================================

def demo_multi_agent_system():
    """
    Multi-agent system where each agent is a subgraph.
    
    Demonstrates:
    - Agents as independent subgraphs
    - Agent communication via shared state
    - Coordinator pattern
    """
    print_banner("Demo 4: Multi-Agent System with Subgraphs")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Research Agent Subgraph
    class ResearchAgentState(TypedDict):
        query: str
        research_findings: str
    
    def research_agent_work(state: ResearchAgentState) -> dict:
        log("Research Agent: Gathering information", indent=1)
        response = llm.invoke(
            f"As a research agent, provide 2-3 key facts about: {state['query']}"
        )
        return {"research_findings": response.content}
    
    research_builder = StateGraph(ResearchAgentState)
    research_builder.add_node("research", research_agent_work)
    research_builder.add_edge(START, "research")
    research_builder.add_edge("research", END)
    research_agent = research_builder.compile()
    
    # Writing Agent Subgraph
    class WritingAgentState(TypedDict):
        topic: str
        research: str
        draft: str
    
    def writing_agent_work(state: WritingAgentState) -> dict:
        log("Writing Agent: Creating draft", indent=1)
        response = llm.invoke(
            f"As a writing agent, write a brief paragraph about '{state['topic']}' "
            f"using these facts: {state['research']}"
        )
        return {"draft": response.content}
    
    writing_builder = StateGraph(WritingAgentState)
    writing_builder.add_node("write", writing_agent_work)
    writing_builder.add_edge(START, "write")
    writing_builder.add_edge("write", END)
    writing_agent = writing_builder.compile()
    
    # Coordinator/Parent Graph
    class CoordinatorState(TypedDict):
        user_topic: str
        research_findings: str
        final_output: str
        agents_used: Annotated[list[str], reduce_list]
    
    def call_research_agent(state: CoordinatorState) -> dict:
        log("Coordinator: Delegating to Research Agent")
        result = research_agent.invoke({
            "query": state["user_topic"],
            "research_findings": ""
        })
        return {
            "research_findings": result["research_findings"],
            "agents_used": ["research_agent"]
        }
    
    def call_writing_agent(state: CoordinatorState) -> dict:
        log("Coordinator: Delegating to Writing Agent")
        result = writing_agent.invoke({
            "topic": state["user_topic"],
            "research": state["research_findings"],
            "draft": ""
        })
        return {
            "final_output": result["draft"],
            "agents_used": ["writing_agent"]
        }
    
    coord_builder = StateGraph(CoordinatorState)
    coord_builder.add_node("research", call_research_agent)
    coord_builder.add_node("writing", call_writing_agent)
    coord_builder.add_edge(START, "research")
    coord_builder.add_edge("research", "writing")
    coord_builder.add_edge("writing", END)
    
    coordinator = coord_builder.compile()
    
    result = coordinator.invoke({
        "user_topic": "renewable energy",
        "research_findings": "",
        "final_output": "",
        "agents_used": []
    })
    
    log(f"\nResearch findings: {result['research_findings'][:200]}...")
    log(f"\nFinal output: {result['final_output'][:200]}...")
    log(f"Agents used: {result['agents_used']}")


# =============================================================================
# Demo 5: Subgraph with Private State
# =============================================================================

def demo_private_state():
    """
    Subgraph with private state not visible to parent.
    
    Demonstrates:
    - Private internal state
    - Exposing only necessary output
    - Encapsulation of complex logic
    """
    print_banner("Demo 5: Subgraph with Private State")
    
    # Subgraph with extended private state
    class PrivateWorkerState(TypedDict):
        input_value: int
        # Private state keys
        internal_counter: int
        intermediate_results: list[int]
        debug_log: list[str]
        # Output key
        output_value: int
    
    def worker_init(state: PrivateWorkerState) -> dict:
        return {
            "internal_counter": 0,
            "intermediate_results": [],
            "debug_log": ["Initialized"]
        }
    
    def worker_process(state: PrivateWorkerState) -> dict:
        counter = state["internal_counter"] + 1
        intermediate = state["input_value"] * counter
        return {
            "internal_counter": counter,
            "intermediate_results": state["intermediate_results"] + [intermediate],
            "debug_log": state["debug_log"] + [f"Step {counter}: {intermediate}"]
        }
    
    def should_continue(state: PrivateWorkerState) -> str:
        if state["internal_counter"] < 3:
            return "process"
        return "finalize"
    
    def worker_finalize(state: PrivateWorkerState) -> dict:
        total = sum(state["intermediate_results"])
        log(f"  Worker internal debug: {state['debug_log']}", indent=1)
        return {"output_value": total}
    
    worker_builder = StateGraph(PrivateWorkerState)
    worker_builder.add_node("init", worker_init)
    worker_builder.add_node("process", worker_process)
    worker_builder.add_node("finalize", worker_finalize)
    worker_builder.add_edge(START, "init")
    worker_builder.add_edge("init", "process")
    worker_builder.add_conditional_edges("process", should_continue)
    worker_builder.add_edge("finalize", END)
    
    worker_graph = worker_builder.compile()
    
    # Parent with minimal state
    class ParentSimpleState(TypedDict):
        number: int
        result: int
    
    def call_worker(state: ParentSimpleState) -> dict:
        log("Parent: Calling worker subgraph")
        # Only pass input, receive output - internal state is hidden
        worker_output = worker_graph.invoke({
            "input_value": state["number"],
            "internal_counter": 0,
            "intermediate_results": [],
            "debug_log": [],
            "output_value": 0
        })
        # Parent only sees the output
        return {"result": worker_output["output_value"]}
    
    parent_builder = StateGraph(ParentSimpleState)
    parent_builder.add_node("worker", call_worker)
    parent_builder.add_edge(START, "worker")
    parent_builder.add_edge("worker", END)
    
    graph = parent_builder.compile()
    
    result = graph.invoke({"number": 5, "result": 0})
    log(f"\nInput: 5, Result: {result['result']}")
    log("(Parent doesn't see internal counter, intermediate results, or debug log)")


# =============================================================================
# Demo 6: Parallel Subgraphs
# =============================================================================

def demo_parallel_subgraphs():
    """
    Multiple subgraphs running in parallel.
    
    Demonstrates:
    - Concurrent subgraph execution
    - Aggregating parallel results
    - Independent processing paths
    """
    print_banner("Demo 6: Parallel Subgraphs")
    
    # Subgraph A: Number processing
    class NumberState(TypedDict):
        value: int
        result: int
    
    def process_numbers(state: NumberState) -> dict:
        log("Subgraph A: Processing numbers", indent=1)
        return {"result": state["value"] * 2}
    
    number_builder = StateGraph(NumberState)
    number_builder.add_node("process", process_numbers)
    number_builder.add_edge(START, "process")
    number_builder.add_edge("process", END)
    number_subgraph = number_builder.compile()
    
    # Subgraph B: Text processing
    class TextState(TypedDict):
        text: str
        result: str
    
    def process_text(state: TextState) -> dict:
        log("Subgraph B: Processing text", indent=1)
        return {"result": state["text"].upper()}
    
    text_builder = StateGraph(TextState)
    text_builder.add_node("process", process_text)
    text_builder.add_edge(START, "process")
    text_builder.add_edge("process", END)
    text_subgraph = text_builder.compile()
    
    # Parent graph with parallel subgraphs
    class ParallelState(TypedDict):
        number_input: int
        text_input: str
        number_output: int
        text_output: str
        combined: str
    
    def run_number_subgraph(state: ParallelState) -> dict:
        result = number_subgraph.invoke({
            "value": state["number_input"],
            "result": 0
        })
        return {"number_output": result["result"]}
    
    def run_text_subgraph(state: ParallelState) -> dict:
        result = text_subgraph.invoke({
            "text": state["text_input"],
            "result": ""
        })
        return {"text_output": result["result"]}
    
    def combine_results(state: ParallelState) -> dict:
        log("Parent: Combining results")
        return {
            "combined": f"Number: {state['number_output']}, Text: {state['text_output']}"
        }
    
    parent_builder = StateGraph(ParallelState)
    parent_builder.add_node("number_process", run_number_subgraph)
    parent_builder.add_node("text_process", run_text_subgraph)
    parent_builder.add_node("combine", combine_results)
    
    # Parallel edges from START
    parent_builder.add_edge(START, "number_process")
    parent_builder.add_edge(START, "text_process")
    parent_builder.add_edge("number_process", "combine")
    parent_builder.add_edge("text_process", "combine")
    parent_builder.add_edge("combine", END)
    
    graph = parent_builder.compile()
    
    result = graph.invoke({
        "number_input": 21,
        "text_input": "hello",
        "number_output": 0,
        "text_output": "",
        "combined": ""
    })
    
    log(f"\nCombined result: {result['combined']}")


# =============================================================================
# Demo 7: Conditional Subgraph Selection
# =============================================================================

def demo_conditional_subgraphs():
    """
    Dynamically select which subgraph to run based on state.
    
    Demonstrates:
    - Conditional subgraph routing
    - Different processing paths
    - Dynamic workflow composition
    """
    print_banner("Demo 7: Conditional Subgraph Selection")
    
    # Subgraph for simple tasks
    class SimpleTaskState(TypedDict):
        task: str
        result: str
    
    def simple_process(state: SimpleTaskState) -> dict:
        log("Simple Subgraph: Quick processing", indent=1)
        return {"result": f"SIMPLE: {state['task']}"}
    
    simple_builder = StateGraph(SimpleTaskState)
    simple_builder.add_node("process", simple_process)
    simple_builder.add_edge(START, "process")
    simple_builder.add_edge("process", END)
    simple_subgraph = simple_builder.compile()
    
    # Subgraph for complex tasks
    class ComplexTaskState(TypedDict):
        task: str
        result: str
    
    def complex_analyze(state: ComplexTaskState) -> dict:
        log("Complex Subgraph: Analyzing", indent=1)
        return {}
    
    def complex_process(state: ComplexTaskState) -> dict:
        log("Complex Subgraph: Deep processing", indent=1)
        return {}
    
    def complex_validate(state: ComplexTaskState) -> dict:
        log("Complex Subgraph: Validating", indent=1)
        return {"result": f"COMPLEX-VALIDATED: {state['task']}"}
    
    complex_builder = StateGraph(ComplexTaskState)
    complex_builder.add_node("analyze", complex_analyze)
    complex_builder.add_node("process", complex_process)
    complex_builder.add_node("validate", complex_validate)
    complex_builder.add_edge(START, "analyze")
    complex_builder.add_edge("analyze", "process")
    complex_builder.add_edge("process", "validate")
    complex_builder.add_edge("validate", END)
    complex_subgraph = complex_builder.compile()
    
    # Parent graph with conditional routing
    class RouterState(TypedDict):
        task: str
        complexity: str  # "simple" or "complex"
        result: str
    
    def route_to_subgraph(state: RouterState) -> str:
        if state["complexity"] == "complex":
            return "complex"
        return "simple"
    
    def call_simple(state: RouterState) -> dict:
        log("Router: Delegating to simple subgraph")
        result = simple_subgraph.invoke({
            "task": state["task"],
            "result": ""
        })
        return {"result": result["result"]}
    
    def call_complex(state: RouterState) -> dict:
        log("Router: Delegating to complex subgraph")
        result = complex_subgraph.invoke({
            "task": state["task"],
            "result": ""
        })
        return {"result": result["result"]}
    
    router_builder = StateGraph(RouterState)
    router_builder.add_node("simple", call_simple)
    router_builder.add_node("complex", call_complex)
    router_builder.add_conditional_edges(START, route_to_subgraph)
    router_builder.add_edge("simple", END)
    router_builder.add_edge("complex", END)
    
    graph = router_builder.compile()
    
    # Test both paths
    log("\nTest 1: Simple task")
    result1 = graph.invoke({
        "task": "quick task",
        "complexity": "simple",
        "result": ""
    })
    log(f"Result: {result1['result']}")
    
    log("\nTest 2: Complex task")
    result2 = graph.invoke({
        "task": "detailed analysis",
        "complexity": "complex",
        "result": ""
    })
    log(f"Result: {result2['result']}")


# =============================================================================
# Demo 8: Streaming from Subgraphs
# =============================================================================

def demo_streaming_subgraphs():
    """
    Stream output from nested subgraphs.
    
    Demonstrates:
    - Streaming through subgraph hierarchy
    - Observing nested execution
    - Debugging subgraph workflows
    """
    print_banner("Demo 8: Streaming from Subgraphs")
    
    # Shared state
    class StreamState(TypedDict):
        value: int
        trace: Annotated[list[str], reduce_list]
    
    # Child subgraph
    def child_a(state: StreamState) -> dict:
        return {"value": state["value"] + 10, "trace": ["child_a"]}
    
    def child_b(state: StreamState) -> dict:
        return {"value": state["value"] * 2, "trace": ["child_b"]}
    
    child_builder = StateGraph(StreamState)
    child_builder.add_node("child_a", child_a)
    child_builder.add_node("child_b", child_b)
    child_builder.add_edge(START, "child_a")
    child_builder.add_edge("child_a", "child_b")
    child_builder.add_edge("child_b", END)
    child_graph = child_builder.compile()
    
    # Parent graph
    def parent_start(state: StreamState) -> dict:
        return {"value": state["value"] + 1, "trace": ["parent_start"]}
    
    def parent_end(state: StreamState) -> dict:
        return {"value": state["value"] + 100, "trace": ["parent_end"]}
    
    parent_builder = StateGraph(StreamState)
    parent_builder.add_node("start", parent_start)
    parent_builder.add_node("child", child_graph)  # Direct subgraph as node
    parent_builder.add_node("end", parent_end)
    parent_builder.add_edge(START, "start")
    parent_builder.add_edge("start", "child")
    parent_builder.add_edge("child", "end")
    parent_builder.add_edge("end", END)
    
    graph = parent_builder.compile()
    
    log("Streaming with subgraph_diff=True:")
    for chunk in graph.stream(
        {"value": 5, "trace": []},
        stream_mode="updates",
        subgraphs=True  # Include subgraph updates
    ):
        # chunk is a tuple of (namespace_path, update)
        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, update = chunk
            ns_str = " > ".join(namespace) if namespace else "root"
            log(f"[{ns_str}] {update}", indent=1)
        else:
            log(f"[root] {chunk}", indent=1)


# =============================================================================
# Demo 9: Reusable Subgraph Components
# =============================================================================

def demo_reusable_components():
    """
    Create reusable subgraph components for common patterns.
    
    Demonstrates:
    - Factory pattern for subgraphs
    - Reusable processing pipelines
    - Configurable subgraph behavior
    """
    print_banner("Demo 9: Reusable Subgraph Components")
    
    # Factory function to create validation subgraph
    def create_validator_subgraph(validation_rules: dict):
        """Factory to create customized validator subgraphs."""
        
        class ValidatorState(TypedDict):
            data: dict
            is_valid: bool
            errors: list[str]
        
        def validate(state: ValidatorState) -> dict:
            errors = []
            data = state["data"]
            
            for field, rules in validation_rules.items():
                if "required" in rules and rules["required"]:
                    if field not in data or not data[field]:
                        errors.append(f"{field} is required")
                
                if "min_length" in rules and field in data:
                    if len(str(data[field])) < rules["min_length"]:
                        errors.append(f"{field} must be at least {rules['min_length']} chars")
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors
            }
        
        builder = StateGraph(ValidatorState)
        builder.add_node("validate", validate)
        builder.add_edge(START, "validate")
        builder.add_edge("validate", END)
        return builder.compile()
    
    # Create different validators using the factory
    user_validator = create_validator_subgraph({
        "name": {"required": True, "min_length": 2},
        "email": {"required": True}
    })
    
    product_validator = create_validator_subgraph({
        "title": {"required": True, "min_length": 3},
        "price": {"required": True}
    })
    
    # Test user validator
    log("Testing User Validator:")
    result1 = user_validator.invoke({
        "data": {"name": "A", "email": ""},
        "is_valid": False,
        "errors": []
    })
    log(f"  Valid: {result1['is_valid']}, Errors: {result1['errors']}", indent=1)
    
    result2 = user_validator.invoke({
        "data": {"name": "Alice", "email": "alice@example.com"},
        "is_valid": False,
        "errors": []
    })
    log(f"  Valid: {result2['is_valid']}, Errors: {result2['errors']}", indent=1)
    
    # Test product validator
    log("\nTesting Product Validator:")
    result3 = product_validator.invoke({
        "data": {"title": "AB", "price": ""},
        "is_valid": False,
        "errors": []
    })
    log(f"  Valid: {result3['is_valid']}, Errors: {result3['errors']}", indent=1)


# =============================================================================
# Demo 10: Hierarchical Organization Simulation
# =============================================================================

def demo_hierarchical_org():
    """
    Simulate an organization hierarchy using subgraphs.
    
    Demonstrates:
    - Organizational patterns with subgraphs
    - Delegation and reporting
    - Complex multi-level workflows
    """
    print_banner("Demo 10: Hierarchical Organization Simulation")
    
    # Team Member Subgraph
    class TeamMemberState(TypedDict):
        task: str
        member_name: str
        work_output: str
    
    def create_team_member(name: str):
        def do_work(state: TeamMemberState) -> dict:
            log(f"Team Member {name}: Working on task", indent=2)
            return {"work_output": f"[{name}'s work on: {state['task']}]"}
        
        builder = StateGraph(TeamMemberState)
        builder.add_node("work", do_work)
        builder.add_edge(START, "work")
        builder.add_edge("work", END)
        return builder.compile()
    
    alice = create_team_member("Alice")
    bob = create_team_member("Bob")
    
    # Team Lead Subgraph
    class TeamLeadState(TypedDict):
        project: str
        team_outputs: Annotated[list[str], reduce_list]
        summary: str
    
    def assign_to_alice(state: TeamLeadState) -> dict:
        log("Team Lead: Assigning to Alice", indent=1)
        result = alice.invoke({
            "task": f"Part A of {state['project']}",
            "member_name": "Alice",
            "work_output": ""
        })
        return {"team_outputs": [result["work_output"]]}
    
    def assign_to_bob(state: TeamLeadState) -> dict:
        log("Team Lead: Assigning to Bob", indent=1)
        result = bob.invoke({
            "task": f"Part B of {state['project']}",
            "member_name": "Bob",
            "work_output": ""
        })
        return {"team_outputs": [result["work_output"]]}
    
    def compile_summary(state: TeamLeadState) -> dict:
        log("Team Lead: Compiling team summary", indent=1)
        return {"summary": f"Team completed: {', '.join(state['team_outputs'])}"}
    
    team_lead_builder = StateGraph(TeamLeadState)
    team_lead_builder.add_node("alice", assign_to_alice)
    team_lead_builder.add_node("bob", assign_to_bob)
    team_lead_builder.add_node("compile", compile_summary)
    team_lead_builder.add_edge(START, "alice")
    team_lead_builder.add_edge(START, "bob")
    team_lead_builder.add_edge("alice", "compile")
    team_lead_builder.add_edge("bob", "compile")
    team_lead_builder.add_edge("compile", END)
    team_lead = team_lead_builder.compile()
    
    # Executive Subgraph
    class ExecutiveState(TypedDict):
        initiative: str
        team_summary: str
        executive_decision: str
    
    def delegate_to_team(state: ExecutiveState) -> dict:
        log("Executive: Delegating to team")
        result = team_lead.invoke({
            "project": state["initiative"],
            "team_outputs": [],
            "summary": ""
        })
        return {"team_summary": result["summary"]}
    
    def make_decision(state: ExecutiveState) -> dict:
        log("Executive: Making final decision")
        return {
            "executive_decision": f"Approved: {state['team_summary']}"
        }
    
    exec_builder = StateGraph(ExecutiveState)
    exec_builder.add_node("delegate", delegate_to_team)
    exec_builder.add_node("decide", make_decision)
    exec_builder.add_edge(START, "delegate")
    exec_builder.add_edge("delegate", "decide")
    exec_builder.add_edge("decide", END)
    
    executive = exec_builder.compile()
    
    result = executive.invoke({
        "initiative": "Q4 Product Launch",
        "team_summary": "",
        "executive_decision": ""
    })
    
    log(f"\nTeam Summary: {result['team_summary']}")
    log(f"Executive Decision: {result['executive_decision']}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all subgraph composition demonstrations."""
    print("\n" + "" * 35)
    print(" LangGraph Subgraph Composition ".center(70))
    print("" * 35)
    
    demo_basic_subgraph()
    demo_different_schema()
    demo_nested_subgraphs()
    demo_multi_agent_system()
    demo_private_state()
    demo_parallel_subgraphs()
    demo_conditional_subgraphs()
    demo_streaming_subgraphs()
    demo_reusable_components()
    demo_hierarchical_org()
    
    print("\n" + "=" * 70)
    print(" All Subgraph Demonstrations Complete! ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
