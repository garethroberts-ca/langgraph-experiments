#!/usr/bin/env python3
"""
LangGraph Advanced Example 32: Streaming Modes
===============================================

This example demonstrates LangGraph's comprehensive streaming capabilities:
- Stream modes: values, updates, messages, custom, debug
- Token-by-token streaming from LLMs
- Custom streaming with get_stream_writer()
- Filtering and processing streamed output
- Combining multiple stream modes

Streaming is essential for building responsive UIs and providing
real-time feedback during long-running agent operations.

Author: LangGraph Examples
"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
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
# Demo 1: Stream Mode - Values
# =============================================================================

def demo_stream_values():
    """
    Streaming with mode='values' returns the full state after each step.
    
    Demonstrates:
    - Full state snapshots at each step
    - Observing state evolution
    - Debugging workflow progression
    """
    print_banner("Demo 1: Stream Mode - values")
    
    class CounterState(TypedDict):
        count: int
        history: list[str]
    
    def step_one(state: CounterState) -> dict:
        return {
            "count": state["count"] + 1,
            "history": state["history"] + ["step_one"]
        }
    
    def step_two(state: CounterState) -> dict:
        return {
            "count": state["count"] * 2,
            "history": state["history"] + ["step_two"]
        }
    
    def step_three(state: CounterState) -> dict:
        return {
            "count": state["count"] + 10,
            "history": state["history"] + ["step_three"]
        }
    
    # Build graph
    builder = StateGraph(CounterState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_node("step_three", step_three)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", "step_three")
    builder.add_edge("step_three", END)
    
    graph = builder.compile()
    
    log("Streaming full state after each step:")
    for i, state in enumerate(graph.stream(
        {"count": 5, "history": []},
        stream_mode="values"
    )):
        log(f"Step {i}: count={state['count']}, history={state['history']}", indent=1)


# =============================================================================
# Demo 2: Stream Mode - Updates
# =============================================================================

def demo_stream_updates():
    """
    Streaming with mode='updates' returns only the delta changes.
    
    Demonstrates:
    - Incremental state updates
    - Efficient change tracking
    - Node-specific updates
    """
    print_banner("Demo 2: Stream Mode - updates")
    
    class ProcessState(TypedDict):
        input_data: str
        processed: str
        validated: bool
        output: str
    
    def process(state: ProcessState) -> dict:
        return {"processed": state["input_data"].upper()}
    
    def validate(state: ProcessState) -> dict:
        return {"validated": len(state["processed"]) > 0}
    
    def output(state: ProcessState) -> dict:
        if state["validated"]:
            return {"output": f" {state['processed']}"}
        return {"output": f" Invalid"}
    
    # Build graph
    builder = StateGraph(ProcessState)
    builder.add_node("process", process)
    builder.add_node("validate", validate)
    builder.add_node("output", output)
    builder.add_edge(START, "process")
    builder.add_edge("process", "validate")
    builder.add_edge("validate", "output")
    builder.add_edge("output", END)
    
    graph = builder.compile()
    
    log("Streaming updates (deltas only):")
    for update in graph.stream(
        {"input_data": "hello world", "processed": "", "validated": False, "output": ""},
        stream_mode="updates"
    ):
        for node_name, changes in update.items():
            log(f"Node '{node_name}' updated: {changes}", indent=1)


# =============================================================================
# Demo 3: Stream Mode - Messages (LLM Tokens)
# =============================================================================

def demo_stream_messages():
    """
    Streaming with mode='messages' for token-by-token LLM output.
    
    Demonstrates:
    - Real-time token streaming
    - Message metadata access
    - Building responsive chat interfaces
    """
    print_banner("Demo 3: Stream Mode - messages")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class ChatState(TypedDict):
        messages: list[BaseMessage]
        topic: str
    
    def generate_response(state: ChatState) -> dict:
        """Generate a response using LLM."""
        response = llm.invoke(
            f"Write a very short (2-3 sentence) explanation about: {state['topic']}"
        )
        return {"messages": [response]}
    
    # Build graph
    builder = StateGraph(ChatState)
    builder.add_node("generate", generate_response)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    
    graph = builder.compile()
    
    log("Streaming LLM tokens:")
    print("  Response: ", end="", flush=True)
    
    for chunk, metadata in graph.stream(
        {"messages": [], "topic": "quantum computing"},
        stream_mode="messages"
    ):
        # chunk is the message chunk (token)
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
    
    print()  # New line after streaming


# =============================================================================
# Demo 4: Stream Mode - Custom
# =============================================================================

def demo_stream_custom():
    """
    Streaming custom data from within nodes using get_stream_writer().
    
    Demonstrates:
    - Emitting progress updates
    - Custom status messages
    - Real-time feedback from nodes
    """
    print_banner("Demo 4: Stream Mode - custom")
    
    class TaskState(TypedDict):
        task: str
        result: str
    
    def process_with_progress(state: TaskState) -> dict:
        """Process task while emitting progress updates."""
        writer = get_stream_writer()
        
        # Emit progress updates
        writer({"type": "progress", "message": "Starting task...", "percent": 0})
        
        # Simulate processing stages
        import time
        stages = ["Initializing", "Processing", "Validating", "Finalizing"]
        for i, stage in enumerate(stages):
            time.sleep(0.1)  # Simulate work
            writer({
                "type": "progress",
                "message": f"{stage}...",
                "percent": (i + 1) * 25
            })
        
        writer({"type": "complete", "message": "Task finished!"})
        
        return {"result": f"Processed: {state['task']}"}
    
    # Build graph
    builder = StateGraph(TaskState)
    builder.add_node("process", process_with_progress)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    log("Streaming custom progress updates:")
    for mode, data in graph.stream(
        {"task": "analyze data", "result": ""},
        stream_mode=["custom", "updates"]  # Combine modes
    ):
        if mode == "custom":
            if data.get("type") == "progress":
                log(f"[{data['percent']}%] {data['message']}", indent=1)
            elif data.get("type") == "complete":
                log(f" {data['message']}", indent=1)
        elif mode == "updates":
            log(f"State update: {data}", indent=1)


# =============================================================================
# Demo 5: Stream Mode - Debug
# =============================================================================

def demo_stream_debug():
    """
    Streaming with mode='debug' for detailed execution traces.
    
    Demonstrates:
    - Detailed execution information
    - Task-level visibility
    - Debugging complex workflows
    """
    print_banner("Demo 5: Stream Mode - debug")
    
    class DebugState(TypedDict):
        value: int
    
    def multiply(state: DebugState) -> dict:
        return {"value": state["value"] * 2}
    
    def add(state: DebugState) -> dict:
        return {"value": state["value"] + 100}
    
    # Build graph
    builder = StateGraph(DebugState)
    builder.add_node("multiply", multiply)
    builder.add_node("add", add)
    builder.add_edge(START, "multiply")
    builder.add_edge("multiply", "add")
    builder.add_edge("add", END)
    
    graph = builder.compile()
    
    log("Streaming debug information:")
    for event in graph.stream(
        {"value": 5},
        stream_mode="debug"
    ):
        # Debug events contain detailed execution info
        event_type = event.get("type", "unknown")
        log(f"Event type: {event_type}", indent=1)
        if event_type == "task":
            log(f"  Task: {event.get('payload', {}).get('name', 'N/A')}", indent=1)
        elif event_type == "task_result":
            log(f"  Result: {event.get('payload', {}).get('result', 'N/A')}", indent=1)


# =============================================================================
# Demo 6: Combining Multiple Stream Modes
# =============================================================================

def demo_combined_modes():
    """
    Combine multiple stream modes for comprehensive output.
    
    Demonstrates:
    - Using multiple modes simultaneously
    - Filtering and routing different event types
    - Building comprehensive monitoring
    """
    print_banner("Demo 6: Combining Multiple Stream Modes")
    
    class CombinedState(TypedDict):
        input: str
        output: str
    
    def process_input(state: CombinedState) -> dict:
        writer = get_stream_writer()
        writer({"event": "input_received", "value": state["input"]})
        return {"output": f"Processed: {state['input']}"}
    
    # Build graph
    builder = StateGraph(CombinedState)
    builder.add_node("process", process_input)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    log("Streaming with multiple modes (custom + updates + values):")
    for event in graph.stream(
        {"input": "test data", "output": ""},
        stream_mode=["custom", "updates", "values"]
    ):
        mode, data = event
        log(f"[{mode}] {data}", indent=1)


# =============================================================================
# Demo 7: Async Streaming with Events
# =============================================================================

async def demo_async_stream_events():
    """
    Async streaming using astream_events for fine-grained control.
    
    Demonstrates:
    - Async event streaming
    - Event type filtering
    - Low-level streaming control
    """
    print_banner("Demo 7: Async Streaming with Events")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    
    class AsyncChatState(TypedDict):
        query: str
        response: str
    
    async def generate_async(state: AsyncChatState) -> dict:
        """Generate response asynchronously."""
        response = await llm.ainvoke(
            f"In one sentence, answer: {state['query']}"
        )
        return {"response": response.content}
    
    # Build graph
    builder = StateGraph(AsyncChatState)
    builder.add_node("generate", generate_async)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    
    graph = builder.compile()
    
    log("Async streaming with astream_events:")
    print("  Response: ", end="", flush=True)
    
    async for event in graph.astream_events(
        {"query": "What is machine learning?", "response": ""},
        version="v2"
    ):
        kind = event.get("event", "")
        
        # Filter for chat model stream events
        if kind == "on_chat_model_stream":
            content = event.get("data", {}).get("chunk", {})
            if hasattr(content, 'content') and content.content:
                print(content.content, end="", flush=True)
    
    print()


# =============================================================================
# Demo 8: Streaming with Node Filtering
# =============================================================================

def demo_filtered_streaming():
    """
    Filter streamed messages by node.
    
    Demonstrates:
    - Node-specific streaming
    - Metadata-based filtering
    - Targeted output processing
    """
    print_banner("Demo 8: Streaming with Node Filtering")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    class FilterState(TypedDict):
        topic: str
        joke: str
        poem: str
    
    def generate_joke(state: FilterState) -> dict:
        response = llm.invoke(f"Tell a one-liner joke about {state['topic']}")
        return {"joke": response.content}
    
    def generate_poem(state: FilterState) -> dict:
        response = llm.invoke(f"Write a one-line poem about {state['topic']}")
        return {"poem": response.content}
    
    # Build graph - parallel execution
    builder = StateGraph(FilterState)
    builder.add_node("joke_node", generate_joke)
    builder.add_node("poem_node", generate_poem)
    builder.add_edge(START, "joke_node")
    builder.add_edge(START, "poem_node")
    builder.add_edge("joke_node", END)
    builder.add_edge("poem_node", END)
    
    graph = builder.compile()
    
    log("Streaming with node filtering:")
    
    # Stream and filter by node
    log("\nJoke node output:", indent=1)
    print("    ", end="", flush=True)
    for chunk, metadata in graph.stream(
        {"topic": "programming", "joke": "", "poem": ""},
        stream_mode="messages"
    ):
        if hasattr(chunk, 'content') and chunk.content:
            if metadata.get("langgraph_node") == "joke_node":
                print(chunk.content, end="", flush=True)
    
    print()
    
    # Run again for poem
    log("\nPoem node output:", indent=1)
    print("    ", end="", flush=True)
    for chunk, metadata in graph.stream(
        {"topic": "programming", "joke": "", "poem": ""},
        stream_mode="messages"
    ):
        if hasattr(chunk, 'content') and chunk.content:
            if metadata.get("langgraph_node") == "poem_node":
                print(chunk.content, end="", flush=True)
    
    print()


# =============================================================================
# Demo 9: Custom Progress Streaming
# =============================================================================

def demo_progress_streaming():
    """
    Stream progress for multi-step operations.
    
    Demonstrates:
    - Progress tracking
    - Status updates
    - ETA calculations
    """
    print_banner("Demo 9: Custom Progress Streaming")
    
    class ProgressState(TypedDict):
        items: list[str]
        processed: int
        results: list[str]
    
    def process_batch(state: ProgressState) -> dict:
        """Process items with progress updates."""
        writer = get_stream_writer()
        
        total = len(state["items"])
        results = []
        
        import time
        start_time = time.time()
        
        for i, item in enumerate(state["items"]):
            # Process item
            time.sleep(0.1)  # Simulate work
            results.append(f"processed_{item}")
            
            # Calculate progress and ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = total - (i + 1)
            eta = avg_time * remaining
            
            writer({
                "type": "progress",
                "current": i + 1,
                "total": total,
                "percent": int(((i + 1) / total) * 100),
                "item": item,
                "eta_seconds": round(eta, 1)
            })
        
        writer({"type": "done", "total_time": round(time.time() - start_time, 2)})
        
        return {"processed": total, "results": results}
    
    # Build graph
    builder = StateGraph(ProgressState)
    builder.add_node("process", process_batch)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    log("Streaming batch progress:")
    for mode, data in graph.stream(
        {"items": ["a", "b", "c", "d", "e"], "processed": 0, "results": []},
        stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            if data.get("type") == "progress":
                bar_len = 20
                filled = int(bar_len * data["percent"] / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                log(f"[{bar}] {data['percent']}% - {data['item']} (ETA: {data['eta_seconds']}s)", indent=1)
            elif data.get("type") == "done":
                log(f" Completed in {data['total_time']}s", indent=1)


# =============================================================================
# Demo 10: Real-time Dashboard Streaming
# =============================================================================

def demo_dashboard_streaming():
    """
    Stream data suitable for a real-time dashboard.
    
    Demonstrates:
    - Structured status updates
    - Multiple metrics streaming
    - Dashboard-ready data format
    """
    print_banner("Demo 10: Real-time Dashboard Streaming")
    
    class DashboardState(TypedDict):
        query: str
        status: str
        metrics: dict
    
    def analyze_query(state: DashboardState) -> dict:
        """Analyze query with dashboard updates."""
        writer = get_stream_writer()
        
        import time
        import random
        
        # Phase 1: Initialization
        writer({
            "phase": "init",
            "status": "Initializing analysis...",
            "metrics": {"progress": 0, "steps_completed": 0}
        })
        time.sleep(0.1)
        
        # Phase 2: Processing
        for i in range(5):
            time.sleep(0.1)
            writer({
                "phase": "processing",
                "status": f"Processing step {i+1}/5...",
                "metrics": {
                    "progress": (i + 1) * 20,
                    "steps_completed": i + 1,
                    "tokens_processed": (i + 1) * 100,
                    "confidence": random.uniform(0.7, 0.99)
                }
            })
        
        # Phase 3: Completion
        writer({
            "phase": "complete",
            "status": "Analysis complete!",
            "metrics": {
                "progress": 100,
                "steps_completed": 5,
                "tokens_processed": 500,
                "confidence": 0.95,
                "execution_time_ms": 550
            }
        })
        
        return {"status": "complete", "metrics": {"confidence": 0.95}}
    
    # Build graph
    builder = StateGraph(DashboardState)
    builder.add_node("analyze", analyze_query)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", END)
    
    graph = builder.compile()
    
    log("Streaming dashboard updates:")
    for mode, data in graph.stream(
        {"query": "analyze trends", "status": "", "metrics": {}},
        stream_mode=["custom"]
    ):
        if mode == "custom":
            phase = data.get("phase", "unknown")
            status = data.get("status", "")
            metrics = data.get("metrics", {})
            
            emoji = "" if phase == "processing" else ("" if phase == "complete" else "🚀")
            log(f"{emoji} [{phase.upper()}] {status}", indent=1)
            log(f"   Metrics: {metrics}", indent=1)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all streaming demonstrations."""
    print("\n" + "📡" * 35)
    print(" LangGraph Streaming Modes ".center(70))
    print("📡" * 35)
    
    # Run synchronous demos
    demo_stream_values()
    demo_stream_updates()
    demo_stream_messages()
    demo_stream_custom()
    demo_stream_debug()
    demo_combined_modes()
    demo_filtered_streaming()
    demo_progress_streaming()
    demo_dashboard_streaming()
    
    # Run async demo
    asyncio.run(demo_async_stream_events())
    
    print("\n" + "=" * 70)
    print(" All Streaming Demonstrations Complete! ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
