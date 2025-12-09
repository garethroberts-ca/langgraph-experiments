#!/usr/bin/env python3
"""
LangGraph Advanced Example 31: Map-Reduce with Send API
========================================================

This example demonstrates LangGraph's Send API for dynamic map-reduce workflows:
- Using Send to dynamically fan-out to multiple node instances
- Processing varying numbers of items in parallel
- Aggregating results with reducer functions
- Implementing complex data processing pipelines

The Send API enables workflows where the number of parallel branches
is determined at runtime based on the current state.

Author: LangGraph Examples
"""

import asyncio
import operator
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_openai import ChatOpenAI
from langgraph.constants import Send
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


# =============================================================================
# Demo 1: Basic Map-Reduce Pattern
# =============================================================================

def demo_basic_map_reduce():
    """
    Basic map-reduce using Send API.
    
    Demonstrates:
    - Fan-out to multiple instances of the same node
    - Using reducers to aggregate results
    - Dynamic number of parallel tasks
    """
    print_banner("Demo 1: Basic Map-Reduce Pattern")
    
    # Overall state for the graph
    class OverallState(TypedDict):
        items: list[str]
        results: Annotated[list[str], operator.add]  # Reducer for aggregation
    
    # State for individual processing tasks
    class ItemState(TypedDict):
        item: str
    
    def process_item(state: ItemState) -> dict:
        """Process a single item (simulated transformation)."""
        item = state["item"]
        log(f"Processing: {item}")
        result = f"Processed({item.upper()})"
        return {"results": [result]}
    
    def fan_out_to_items(state: OverallState) -> list[Send]:
        """Create Send objects to fan out to each item."""
        log(f"Fanning out to {len(state['items'])} items")
        return [
            Send("process_item", {"item": item})
            for item in state["items"]
        ]
    
    # Build the graph
    builder = StateGraph(OverallState)
    builder.add_node("process_item", process_item)
    builder.add_conditional_edges(
        START,
        fan_out_to_items,
        ["process_item"]
    )
    builder.add_edge("process_item", END)
    
    graph = builder.compile()
    
    # Test with varying number of items
    test_cases = [
        ["apple", "banana", "cherry"],
        ["x", "y"],
        ["one", "two", "three", "four", "five"]
    ]
    
    for items in test_cases:
        log(f"\nProcessing batch: {items}")
        result = graph.invoke({
            "items": items,
            "results": []
        })
        log(f"Results: {result['results']}", indent=1)


# =============================================================================
# Demo 2: Parallel LLM Calls with Send
# =============================================================================

def demo_parallel_llm_calls():
    """
    Use Send API to make parallel LLM calls.
    
    Demonstrates:
    - Parallel LLM invocations for different perspectives
    - Aggregating multiple LLM responses
    - Efficient batch processing
    """
    print_banner("Demo 2: Parallel LLM Calls with Send")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Overall state
    class AnalysisState(TypedDict):
        topic: str
        perspectives: list[str]
        analyses: Annotated[list[dict], operator.add]
    
    # Individual analysis state
    class PerspectiveState(TypedDict):
        topic: str
        perspective: str
    
    def analyze_from_perspective(state: PerspectiveState) -> dict:
        """Analyze topic from a specific perspective."""
        topic = state["topic"]
        perspective = state["perspective"]
        
        log(f"Analyzing '{topic}' from {perspective} perspective")
        
        response = llm.invoke(
            f"In 2-3 sentences, analyze this topic from a {perspective} "
            f"perspective: {topic}"
        )
        
        return {
            "analyses": [{
                "perspective": perspective,
                "analysis": response.content
            }]
        }
    
    def fan_out_to_perspectives(state: AnalysisState) -> list[Send]:
        """Create parallel analysis tasks for each perspective."""
        log(f"Creating {len(state['perspectives'])} parallel analysis tasks")
        return [
            Send("analyze", {
                "topic": state["topic"],
                "perspective": perspective
            })
            for perspective in state["perspectives"]
        ]
    
    # Build graph
    builder = StateGraph(AnalysisState)
    builder.add_node("analyze", analyze_from_perspective)
    builder.add_conditional_edges(
        START,
        fan_out_to_perspectives,
        ["analyze"]
    )
    builder.add_edge("analyze", END)
    
    graph = builder.compile()
    
    # Run analysis
    result = graph.invoke({
        "topic": "Remote work policies",
        "perspectives": ["employee", "manager", "HR"],
        "analyses": []
    })
    
    log("\nAggregated Analyses:")
    for analysis in result["analyses"]:
        log(f"\n[{analysis['perspective'].upper()}]", indent=1)
        log(analysis["analysis"], indent=2)


# =============================================================================
# Demo 3: Multi-Stage Map-Reduce
# =============================================================================

def demo_multi_stage_map_reduce():
    """
    Multi-stage map-reduce with multiple fan-out/fan-in phases.
    
    Demonstrates:
    - Chained map-reduce operations
    - Intermediate aggregation
    - Complex data transformation pipelines
    """
    print_banner("Demo 3: Multi-Stage Map-Reduce")
    
    # State definitions
    class PipelineState(TypedDict):
        documents: list[str]
        chunks: Annotated[list[str], operator.add]
        summaries: Annotated[list[str], operator.add]
        final_summary: str
    
    class ChunkState(TypedDict):
        document: str
    
    class SummaryState(TypedDict):
        chunk: str
    
    # Stage 1: Split documents into chunks
    def chunk_document(state: ChunkState) -> dict:
        """Split a document into smaller chunks."""
        doc = state["document"]
        log(f"Chunking document: {doc[:30]}...")
        # Simulate chunking
        words = doc.split()
        chunk_size = max(1, len(words) // 2)
        chunks = [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        return {"chunks": chunks}
    
    def fan_out_to_documents(state: PipelineState) -> list[Send]:
        """Fan out to chunk each document."""
        return [
            Send("chunk", {"document": doc})
            for doc in state["documents"]
        ]
    
    # Stage 2: Summarize each chunk
    def summarize_chunk(state: SummaryState) -> dict:
        """Create a summary of a chunk."""
        chunk = state["chunk"]
        log(f"Summarizing chunk: {chunk[:20]}...")
        # Simulate summarization
        summary = f"Summary of: {chunk[:15]}..."
        return {"summaries": [summary]}
    
    def fan_out_to_chunks(state: PipelineState) -> list[Send]:
        """Fan out to summarize each chunk."""
        if not state.get("chunks"):
            return []
        return [
            Send("summarize", {"chunk": chunk})
            for chunk in state["chunks"]
        ]
    
    # Stage 3: Combine summaries
    def combine_summaries(state: PipelineState) -> dict:
        """Combine all summaries into final output."""
        summaries = state.get("summaries", [])
        log(f"Combining {len(summaries)} summaries")
        final = f"Combined {len(summaries)} chunks: " + " | ".join(summaries[:3])
        if len(summaries) > 3:
            final += f" ... and {len(summaries) - 3} more"
        return {"final_summary": final}
    
    def route_after_chunking(state: PipelineState) -> list[Send] | str:
        """Route to summarization if chunks exist."""
        if state.get("chunks"):
            return fan_out_to_chunks(state)
        return "combine"
    
    # Build graph
    builder = StateGraph(PipelineState)
    
    # Add nodes
    builder.add_node("chunk", chunk_document)
    builder.add_node("summarize", summarize_chunk)
    builder.add_node("combine", combine_summaries)
    
    # Add edges
    builder.add_conditional_edges(
        START,
        fan_out_to_documents,
        ["chunk"]
    )
    builder.add_conditional_edges(
        "chunk",
        route_after_chunking,
        ["summarize", "combine"]
    )
    builder.add_edge("summarize", "combine")
    builder.add_edge("combine", END)
    
    graph = builder.compile()
    
    # Run pipeline
    result = graph.invoke({
        "documents": [
            "The quick brown fox jumps over the lazy dog repeatedly",
            "LangGraph enables complex workflow orchestration with ease",
            "Map reduce is a powerful pattern for parallel processing"
        ],
        "chunks": [],
        "summaries": [],
        "final_summary": ""
    })
    
    log(f"\nFinal Summary: {result['final_summary']}")


# =============================================================================
# Demo 4: Dynamic Task Distribution
# =============================================================================

def demo_dynamic_task_distribution():
    """
    Dynamic task distribution based on workload characteristics.
    
    Demonstrates:
    - Conditional fan-out based on task properties
    - Load balancing across workers
    - Priority-based processing
    """
    print_banner("Demo 4: Dynamic Task Distribution")
    
    class WorkloadState(TypedDict):
        tasks: list[dict]
        high_priority_results: Annotated[list[str], operator.add]
        normal_results: Annotated[list[str], operator.add]
    
    class TaskState(TypedDict):
        task: dict
    
    def process_high_priority(state: TaskState) -> dict:
        """Process high priority tasks with special handling."""
        task = state["task"]
        log(f"⚡ HIGH PRIORITY: {task['name']}")
        return {
            "high_priority_results": [f"EXPEDITED: {task['name']}"]
        }
    
    def process_normal(state: TaskState) -> dict:
        """Process normal priority tasks."""
        task = state["task"]
        log(f" Normal: {task['name']}")
        return {
            "normal_results": [f"Completed: {task['name']}"]
        }
    
    def distribute_tasks(state: WorkloadState) -> list[Send]:
        """Distribute tasks to appropriate processors based on priority."""
        sends = []
        for task in state["tasks"]:
            if task.get("priority") == "high":
                sends.append(Send("high_priority", {"task": task}))
            else:
                sends.append(Send("normal", {"task": task}))
        
        high_count = sum(1 for t in state["tasks"] if t.get("priority") == "high")
        log(f"Distributing: {high_count} high priority, {len(state['tasks']) - high_count} normal")
        return sends
    
    # Build graph
    builder = StateGraph(WorkloadState)
    builder.add_node("high_priority", process_high_priority)
    builder.add_node("normal", process_normal)
    builder.add_conditional_edges(
        START,
        distribute_tasks,
        ["high_priority", "normal"]
    )
    builder.add_edge("high_priority", END)
    builder.add_edge("normal", END)
    
    graph = builder.compile()
    
    # Run with mixed priority tasks
    result = graph.invoke({
        "tasks": [
            {"name": "Security Update", "priority": "high"},
            {"name": "Code Review", "priority": "normal"},
            {"name": "Production Bug", "priority": "high"},
            {"name": "Documentation", "priority": "normal"},
            {"name": "Customer Escalation", "priority": "high"},
        ],
        "high_priority_results": [],
        "normal_results": []
    })
    
    log("\nResults:")
    log(f"High Priority: {result['high_priority_results']}", indent=1)
    log(f"Normal: {result['normal_results']}", indent=1)


# =============================================================================
# Demo 5: Nested Map-Reduce with Aggregation
# =============================================================================

def demo_nested_map_reduce():
    """
    Nested map-reduce with custom aggregation logic.
    
    Demonstrates:
    - Multiple levels of parallelism
    - Custom reducer functions
    - Hierarchical data processing
    """
    print_banner("Demo 5: Nested Map-Reduce with Aggregation")
    
    # Custom reducer that merges scores
    def merge_scores(left: dict, right: dict) -> dict:
        """Merge score dictionaries."""
        if not left:
            left = {}
        if not right:
            right = {}
        merged = dict(left)
        for key, value in right.items():
            if key in merged:
                merged[key] = merged[key] + value
            else:
                merged[key] = value
        return merged
    
    class EvaluationState(TypedDict):
        candidates: list[str]
        criteria: list[str]
        scores: Annotated[dict, merge_scores]
    
    class CandidateEvalState(TypedDict):
        candidate: str
        criterion: str
    
    def evaluate_candidate(state: CandidateEvalState) -> dict:
        """Evaluate a candidate against a criterion."""
        candidate = state["candidate"]
        criterion = state["criterion"]
        
        # Simulate scoring
        import hashlib
        hash_val = int(hashlib.md5(f"{candidate}{criterion}".encode()).hexdigest()[:4], 16)
        score = (hash_val % 50) + 50  # Score between 50-100
        
        log(f"Evaluating {candidate} on {criterion}: {score}")
        
        return {
            "scores": {f"{candidate}_{criterion}": score}
        }
    
    def create_evaluation_matrix(state: EvaluationState) -> list[Send]:
        """Create evaluation tasks for all candidate-criterion pairs."""
        total_evals = len(state["candidates"]) * len(state["criteria"])
        log(f"Creating {total_evals} evaluation tasks")
        
        sends = []
        for candidate in state["candidates"]:
            for criterion in state["criteria"]:
                sends.append(Send("evaluate", {
                    "candidate": candidate,
                    "criterion": criterion
                }))
        return sends
    
    # Build graph
    builder = StateGraph(EvaluationState)
    builder.add_node("evaluate", evaluate_candidate)
    builder.add_conditional_edges(
        START,
        create_evaluation_matrix,
        ["evaluate"]
    )
    builder.add_edge("evaluate", END)
    
    graph = builder.compile()
    
    # Run evaluation
    result = graph.invoke({
        "candidates": ["Alice", "Bob", "Carol"],
        "criteria": ["Technical", "Communication", "Leadership"],
        "scores": {}
    })
    
    log("\nEvaluation Scores:")
    for key, score in sorted(result["scores"].items()):
        log(f"{key}: {score}", indent=1)
    
    # Calculate totals per candidate
    log("\nCandidate Totals:")
    for candidate in ["Alice", "Bob", "Carol"]:
        total = sum(
            score for key, score in result["scores"].items()
            if key.startswith(candidate)
        )
        log(f"{candidate}: {total}", indent=1)


# =============================================================================
# Demo 6: Map-Reduce with Error Handling
# =============================================================================

def demo_map_reduce_with_errors():
    """
    Map-reduce pattern with error handling for individual tasks.
    
    Demonstrates:
    - Graceful handling of individual task failures
    - Collecting both successes and failures
    - Continuing processing despite errors
    """
    print_banner("Demo 6: Map-Reduce with Error Handling")
    
    class RobustProcessingState(TypedDict):
        items: list[str]
        successes: Annotated[list[dict], operator.add]
        failures: Annotated[list[dict], operator.add]
    
    class ItemProcessState(TypedDict):
        item: str
    
    def process_with_potential_error(state: ItemProcessState) -> dict:
        """Process item, simulating potential failures."""
        item = state["item"]
        
        # Simulate errors for certain items
        if "fail" in item.lower():
            log(f" Failed to process: {item}")
            return {
                "failures": [{"item": item, "error": "Simulated failure"}]
            }
        
        log(f" Successfully processed: {item}")
        return {
            "successes": [{"item": item, "result": f"Processed({item})"}]
        }
    
    def fan_out_items(state: RobustProcessingState) -> list[Send]:
        """Fan out to process each item."""
        return [
            Send("process", {"item": item})
            for item in state["items"]
        ]
    
    # Build graph
    builder = StateGraph(RobustProcessingState)
    builder.add_node("process", process_with_potential_error)
    builder.add_conditional_edges(
        START,
        fan_out_items,
        ["process"]
    )
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    # Run with mixed success/failure items
    result = graph.invoke({
        "items": [
            "task_1",
            "fail_task_2",
            "task_3",
            "task_4_fail",
            "task_5"
        ],
        "successes": [],
        "failures": []
    })
    
    log("\nProcessing Summary:")
    log(f"Successes ({len(result['successes'])}):", indent=1)
    for success in result["successes"]:
        log(f"  {success['item']}: {success['result']}", indent=2)
    
    log(f"Failures ({len(result['failures'])}):", indent=1)
    for failure in result["failures"]:
        log(f"  {failure['item']}: {failure['error']}", indent=2)


# =============================================================================
# Demo 7: Async Map-Reduce
# =============================================================================

async def demo_async_map_reduce():
    """
    Async map-reduce for I/O-bound operations.
    
    Demonstrates:
    - Async processing with Send API
    - Efficient handling of I/O operations
    - Concurrent async tasks
    """
    print_banner("Demo 7: Async Map-Reduce")
    
    class AsyncBatchState(TypedDict):
        urls: list[str]
        responses: Annotated[list[dict], operator.add]
    
    class FetchState(TypedDict):
        url: str
    
    async def simulate_fetch(state: FetchState) -> dict:
        """Simulate async URL fetching."""
        url = state["url"]
        log(f"Fetching: {url}")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        return {
            "responses": [{
                "url": url,
                "status": 200,
                "content_length": len(url) * 100
            }]
        }
    
    def fan_out_urls(state: AsyncBatchState) -> list[Send]:
        """Fan out to fetch each URL."""
        return [
            Send("fetch", {"url": url})
            for url in state["urls"]
        ]
    
    # Build graph
    builder = StateGraph(AsyncBatchState)
    builder.add_node("fetch", simulate_fetch)
    builder.add_conditional_edges(
        START,
        fan_out_urls,
        ["fetch"]
    )
    builder.add_edge("fetch", END)
    
    graph = builder.compile()
    
    # Run async
    import time
    start = time.time()
    
    result = await graph.ainvoke({
        "urls": [
            "https://api.example.com/users",
            "https://api.example.com/products",
            "https://api.example.com/orders",
            "https://api.example.com/inventory",
        ],
        "responses": []
    })
    
    elapsed = time.time() - start
    
    log(f"\nFetched {len(result['responses'])} URLs in {elapsed:.2f}s")
    for resp in result["responses"]:
        log(f"{resp['url']}: {resp['status']} ({resp['content_length']} bytes)", indent=1)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all Send API demonstrations."""
    print("\n" + "🚀" * 35)
    print(" LangGraph Send API: Map-Reduce Patterns ".center(70))
    print("🚀" * 35)
    
    # Run synchronous demos
    demo_basic_map_reduce()
    demo_parallel_llm_calls()
    demo_multi_stage_map_reduce()
    demo_dynamic_task_distribution()
    demo_nested_map_reduce()
    demo_map_reduce_with_errors()
    
    # Run async demo
    asyncio.run(demo_async_map_reduce())
    
    print("\n" + "=" * 70)
    print(" All Send API Demonstrations Complete! ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
