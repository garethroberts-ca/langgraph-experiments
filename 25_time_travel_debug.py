"""
25. Time Travel & Debugging in LangGraph
=========================================
Demonstrates LangGraph's time travel capabilities:

FEATURES:
├── Checkpoint History: View all saved states
├── State Replay: Re-execute from any checkpoint
├── Branching: Fork execution to explore alternatives
├── State Editing: Modify state and create new branches
└── Debug Inspection: Examine state at any point

USE CASES:
- Debug agent decisions by replaying
- Explore "what if" scenarios
- Recover from errors without restart
- A/B test different agent paths
"""

import asyncio
from typing import Annotated, TypedDict, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# STATE DEFINITION
# ============================================================

class CoachingState(TypedDict):
    """State for time travel demo"""
    messages: list
    current_step: str
    decision_history: list[dict]
    score: int


# ============================================================
# CONFIGURATION
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# GRAPH NODES
# ============================================================

def intake_node(state: CoachingState) -> dict:
    """Initial intake - gather context"""
    log("Intake: Processing initial request", "")
    
    messages = state.get("messages", [])
    user_input = messages[-1].content if messages else "Help me improve"
    
    return {
        "current_step": "intake",
        "decision_history": state.get("decision_history", []) + [{
            "step": "intake",
            "timestamp": datetime.now().isoformat(),
            "input": user_input[:100]
        }],
        "score": 10
    }


def analyze_node(state: CoachingState) -> dict:
    """Analyze the situation"""
    log("Analyze: Evaluating situation", "")
    
    # Simulate analysis with score adjustment
    current_score = state.get("score", 0)
    
    return {
        "current_step": "analyze",
        "decision_history": state.get("decision_history", []) + [{
            "step": "analyze",
            "timestamp": datetime.now().isoformat(),
            "analysis": "Identified growth opportunity"
        }],
        "score": current_score + 20
    }


def recommend_node(state: CoachingState) -> dict:
    """Generate recommendations"""
    log("Recommend: Creating action plan", "")
    
    messages = state.get("messages", [])
    current_score = state.get("score", 0)
    
    # Use LLM for recommendation
    response = llm.invoke([
        SystemMessage(content="You are a brief career coach. Give a 1-sentence recommendation."),
        *messages
    ])
    
    return {
        "messages": messages + [response],
        "current_step": "recommend",
        "decision_history": state.get("decision_history", []) + [{
            "step": "recommend",
            "timestamp": datetime.now().isoformat(),
            "recommendation": response.content[:100]
        }],
        "score": current_score + 30
    }


def finalize_node(state: CoachingState) -> dict:
    """Finalize the coaching session"""
    log("Finalize: Wrapping up session", "")
    
    final_score = state.get("score", 0) + 10
    
    return {
        "current_step": "complete",
        "decision_history": state.get("decision_history", []) + [{
            "step": "finalize",
            "timestamp": datetime.now().isoformat(),
            "final_score": final_score
        }],
        "score": final_score
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_coaching_graph():
    """Build graph with checkpointing enabled"""
    
    workflow = StateGraph(CoachingState)
    
    # Add nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("recommend", recommend_node)
    workflow.add_node("finalize", finalize_node)
    
    # Linear flow for demo
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile with checkpointer for time travel
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer), checkpointer


# ============================================================
# TIME TRAVEL UTILITIES
# ============================================================

def display_checkpoint_history(graph, config: dict):
    """Display all checkpoints for a thread"""
    print("\n" + "="*60)
    print(" CHECKPOINT HISTORY")
    print("="*60)
    
    # Get state history
    states = list(graph.get_state_history(config))
    
    for i, state_snapshot in enumerate(reversed(states)):
        checkpoint_id = state_snapshot.config["configurable"].get("checkpoint_id", "N/A")
        step = state_snapshot.values.get("current_step", "start")
        score = state_snapshot.values.get("score", 0)
        
        print(f"\n  [{i}] Checkpoint: {checkpoint_id[:20]}...")
        print(f"      Step: {step}")
        print(f"      Score: {score}")
        print(f"      Next: {state_snapshot.next}")
    
    return states


def replay_from_checkpoint(graph, config: dict, checkpoint_id: str):
    """Replay execution from a specific checkpoint"""
    print(f"\n REPLAYING from checkpoint: {checkpoint_id[:20]}...")
    
    # Create config pointing to specific checkpoint
    replay_config = {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_id": checkpoint_id
        }
    }
    
    # Get state at that checkpoint
    state = graph.get_state(replay_config)
    print(f"   State at checkpoint: step={state.values.get('current_step')}, score={state.values.get('score')}")
    
    return state


def branch_from_checkpoint(graph, config: dict, checkpoint_id: str, new_state: dict):
    """
    Branch execution from a checkpoint with modified state.
    This creates a NEW execution path (fork).
    """
    print(f"\n🌿 BRANCHING from checkpoint: {checkpoint_id[:20]}...")
    
    # Update state at checkpoint to create a fork
    branch_config = {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_id": checkpoint_id
        }
    }
    
    # update_state creates a new checkpoint with modified state
    new_config = graph.update_state(
        branch_config,
        new_state
    )
    
    print(f"   Created new branch with modified state")
    print(f"   New checkpoint: {new_config['configurable'].get('checkpoint_id', 'N/A')[:20]}...")
    
    return new_config


def inspect_state_diff(state1: dict, state2: dict):
    """Compare two states to see differences"""
    print("\n STATE COMPARISON")
    print("-" * 40)
    
    all_keys = set(state1.keys()) | set(state2.keys())
    
    for key in all_keys:
        v1 = state1.get(key)
        v2 = state2.get(key)
        if v1 != v2:
            print(f"  {key}:")
            print(f"    Before: {str(v1)[:50]}...")
            print(f"    After:  {str(v2)[:50]}...")


# ============================================================
# DEMO: BASIC TIME TRAVEL
# ============================================================

def demo_basic_time_travel():
    """Demo 1: Basic checkpoint inspection and replay"""
    print("\n" + "="*70)
    print("🕐 DEMO 1: BASIC TIME TRAVEL")
    print("="*70)
    
    graph, checkpointer = build_coaching_graph()
    
    config = {"configurable": {"thread_id": "time-travel-demo-1"}}
    
    # Run the graph
    print("\n Running initial execution...")
    initial_state = {
        "messages": [HumanMessage(content="I want to improve my leadership skills")],
        "decision_history": [],
        "score": 0
    }
    
    result = graph.invoke(initial_state, config)
    print(f"\n Execution complete. Final score: {result['score']}")
    
    # Display checkpoint history
    states = display_checkpoint_history(graph, config)
    
    # Replay from an earlier checkpoint
    if len(states) > 2:
        # Get checkpoint from the analyze step
        for state in states:
            if state.values.get("current_step") == "analyze":
                checkpoint_id = state.config["configurable"]["checkpoint_id"]
                replay_from_checkpoint(graph, config, checkpoint_id)
                break


# ============================================================
# DEMO: BRANCHING EXECUTION
# ============================================================

def demo_branching():
    """Demo 2: Fork execution to explore alternatives"""
    print("\n" + "="*70)
    print("🌿 DEMO 2: BRANCHING EXECUTION")
    print("="*70)
    
    graph, checkpointer = build_coaching_graph()
    
    config = {"configurable": {"thread_id": "branching-demo-1"}}
    
    # Run initial execution
    print("\n Running initial execution...")
    initial_state = {
        "messages": [HumanMessage(content="Should I change careers?")],
        "decision_history": [],
        "score": 0
    }
    
    result = graph.invoke(initial_state, config)
    print(f"\n Original path complete. Final score: {result['score']}")
    
    # Get checkpoint history
    states = list(graph.get_state_history(config))
    
    # Find the analyze checkpoint and branch from it
    for state in states:
        if state.values.get("current_step") == "analyze":
            checkpoint_id = state.config["configurable"]["checkpoint_id"]
            
            # Create a branch with boosted score (simulating different analysis)
            print("\n Creating alternate reality with boosted initial score...")
            branch_config = branch_from_checkpoint(
                graph, 
                config, 
                checkpoint_id,
                {"score": 100}  # Boost the score in this branch
            )
            
            # Continue execution from the branch
            print("\n Continuing from branch...")
            branch_result = graph.invoke(None, branch_config)
            print(f" Branch complete. Final score: {branch_result['score']}")
            
            # Compare
            print(f"\n📊 COMPARISON:")
            print(f"   Original path final score: {result['score']}")
            print(f"   Branched path final score: {branch_result['score']}")
            break


# ============================================================
# DEMO: ERROR RECOVERY
# ============================================================

class SimulatedError(Exception):
    pass


def demo_error_recovery():
    """Demo 3: Recover from errors using checkpoints"""
    print("\n" + "="*70)
    print(" DEMO 3: ERROR RECOVERY")
    print("="*70)
    
    # Build a graph that might fail
    error_count = {"count": 0}
    
    def flaky_node(state: CoachingState) -> dict:
        """A node that fails on first attempt"""
        error_count["count"] += 1
        if error_count["count"] == 1:
            log("Flaky node: FAILING (simulated error)", "")
            raise SimulatedError("Network timeout - simulated")
        
        log("Flaky node: SUCCESS on retry", "")
        return {
            "current_step": "flaky_complete",
            "score": state.get("score", 0) + 50
        }
    
    # Build graph with flaky node
    workflow = StateGraph(CoachingState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("flaky", flaky_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "flaky")
    workflow.add_edge("flaky", "finalize")
    workflow.add_edge("finalize", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "error-recovery-demo"}}
    
    initial_state = {
        "messages": [HumanMessage(content="Test error recovery")],
        "decision_history": [],
        "score": 0
    }
    
    # First attempt - will fail
    print("\n First attempt (will fail)...")
    try:
        graph.invoke(initial_state, config)
    except SimulatedError as e:
        print(f"    Error caught: {e}")
    
    # Check current state - should be at intake checkpoint
    current = graph.get_state(config)
    print(f"\n📍 Current state after error:")
    print(f"   Step: {current.values.get('current_step')}")
    print(f"   Score: {current.values.get('score')}")
    print(f"   Next nodes to execute: {current.next}")
    
    # Resume from checkpoint (error is now "fixed" - will succeed)
    print("\n Resuming from last checkpoint...")
    result = graph.invoke(None, config)  # None = continue from current state
    
    print(f"\n Recovery successful!")
    print(f"   Final step: {result['current_step']}")
    print(f"   Final score: {result['score']}")


# ============================================================
# DEMO: STATE INSPECTION FOR DEBUGGING
# ============================================================

def demo_debug_inspection():
    """Demo 4: Deep inspection for debugging"""
    print("\n" + "="*70)
    print(" DEMO 4: DEBUG INSPECTION")
    print("="*70)
    
    graph, checkpointer = build_coaching_graph()
    
    config = {"configurable": {"thread_id": "debug-demo"}}
    
    initial_state = {
        "messages": [HumanMessage(content="Debug my career path")],
        "decision_history": [],
        "score": 0
    }
    
    # Run graph
    result = graph.invoke(initial_state, config)
    
    # Deep inspection
    print("\n FULL EXECUTION TRACE")
    print("-" * 50)
    
    states = list(graph.get_state_history(config))
    
    for i, snapshot in enumerate(reversed(states)):
        print(f"\n[Step {i}]")
        print(f"  Checkpoint ID: {snapshot.config['configurable'].get('checkpoint_id', 'N/A')[:30]}...")
        print(f"  Current Step: {snapshot.values.get('current_step', 'N/A')}")
        print(f"  Score: {snapshot.values.get('score', 'N/A')}")
        print(f"  Messages Count: {len(snapshot.values.get('messages', []))}")
        print(f"  Next: {snapshot.next}")
        
        # Show decision history entries
        history = snapshot.values.get("decision_history", [])
        if history:
            print(f"  Decision History:")
            for entry in history:
                print(f"    - {entry.get('step')}: {entry.get('timestamp', '')[:19]}")
    
    # Show state evolution
    print("\n📈 SCORE EVOLUTION")
    print("-" * 30)
    scores = []
    for snapshot in reversed(states):
        step = snapshot.values.get("current_step", "start")
        score = snapshot.values.get("score", 0)
        scores.append((step, score))
    
    for step, score in scores:
        bar = "█" * (score // 5)
        print(f"  {step:15} | {bar} ({score})")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           🕐 LangGraph Time Travel & Debugging Demo 🕐            ║
╠══════════════════════════════════════════════════════════════════╣
║  Features demonstrated:                                          ║
║  • Checkpoint history inspection                                 ║
║  • State replay from any point                                   ║
║  • Execution branching (forking)                                 ║
║  • Error recovery                                                ║
║  • Debug inspection                                              ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_basic_time_travel()
        demo_branching()
        demo_error_recovery()
        demo_debug_inspection()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:
1. Checkpoints save state at every super-step automatically
2. Use get_state_history() to view all checkpoints
3. Use update_state() to branch/fork from any checkpoint
4. Resume from failures by invoking with None (continues from last checkpoint)
5. Time travel enables powerful debugging and "what-if" exploration
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
