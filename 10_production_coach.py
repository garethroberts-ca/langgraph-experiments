"""
Script 10: Production-Grade HR Coaching System
Demonstrates: Supervisor agent, observability, retries, error handling, 
              rate limiting, structured logging, graceful degradation
"""

import asyncio
import logging
import time
from typing import Annotated, TypedDict, Literal, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import json
import traceback

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import ToolNode


# ============================================================
# LOGGING AND OBSERVABILITY
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("hr_coach")


@dataclass
class MetricsCollector:
    """Collect and report metrics for observability."""
    
    request_count: int = 0
    error_count: int = 0
    tool_calls: int = 0
    total_latency_ms: float = 0
    node_timings: dict = field(default_factory=dict)
    
    def record_request(self, latency_ms: float):
        self.request_count += 1
        self.total_latency_ms += latency_ms
    
    def record_error(self, error_type: str):
        self.error_count += 1
        logger.error(f"Error recorded: {error_type}")
    
    def record_tool_call(self, tool_name: str, latency_ms: float):
        self.tool_calls += 1
        if tool_name not in self.node_timings:
            self.node_timings[tool_name] = []
        self.node_timings[tool_name].append(latency_ms)
    
    def get_summary(self) -> dict:
        avg_latency = (self.total_latency_ms / self.request_count) if self.request_count > 0 else 0
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_latency_ms": avg_latency,
            "tool_calls": self.tool_calls,
            "node_avg_timings": {
                k: sum(v) / len(v) for k, v in self.node_timings.items()
            }
        }


metrics = MetricsCollector()


class ObservabilityCallback(BaseCallbackHandler):
    """Custom callback for LLM observability."""
    
    def __init__(self):
        self.start_time = None
    
    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        logger.debug("LLM call started")
    
    def on_llm_end(self, *args, **kwargs):
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000
            logger.debug(f"LLM call completed in {latency:.2f}ms")
    
    def on_llm_error(self, error, *args, **kwargs):
        logger.error(f"LLM error: {error}")
        metrics.record_error("llm_error")


@contextmanager
def timed_operation(operation_name: str):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        latency_ms = (time.time() - start) * 1000
        metrics.record_tool_call(operation_name, latency_ms)
        logger.debug(f"{operation_name} completed in {latency_ms:.2f}ms")


# ============================================================
# RETRY AND ERROR HANDLING
# ============================================================

def with_retry(max_attempts: int = 3, backoff_factor: float = 1.5):
    """Decorator for retrying failed operations with exponential backoff."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_error
        return wrapper
    return decorator


class GracefulDegradation:
    """Handle failures gracefully with fallback responses."""
    
    FALLBACK_RESPONSES = {
        "tool_failure": "I'm having trouble accessing that information right now. Let me help you with what I can.",
        "llm_failure": "I'm experiencing some technical difficulties. Could you try rephrasing your question?",
        "timeout": "This is taking longer than expected. Let me try a simpler approach.",
        "rate_limit": "We're experiencing high demand. Please try again in a moment."
    }
    
    @classmethod
    def get_fallback(cls, error_type: str, context: str = "") -> str:
        base = cls.FALLBACK_RESPONSES.get(error_type, "Something went wrong. Please try again.")
        if context:
            return f"{base}\n\nContext: {context}"
        return base


# ============================================================
# STATE AND TYPES
# ============================================================

class SupervisorDecision(TypedDict):
    next_agent: str
    reasoning: str
    confidence: float


class ErrorContext(TypedDict):
    error_type: str
    message: str
    recoverable: bool
    suggested_action: str


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    
    # Supervisor state
    supervisor_decision: Optional[SupervisorDecision]
    agent_outputs: dict
    iteration_count: int
    max_iterations: int
    
    # Error handling
    last_error: Optional[ErrorContext]
    degraded_mode: bool
    
    # Quality tracking
    response_quality_score: float
    needs_human_review: bool


# ============================================================
# SPECIALIST AGENTS
# ============================================================

@tool
def get_comprehensive_profile(user_id: str) -> str:
    """Get comprehensive employee profile with all relevant data."""
    with timed_operation("get_comprehensive_profile"):
        return json.dumps({
            "employee": {
                "name": "Casey Martinez",
                "role": "Director of Engineering",
                "level": "L7",
                "tenure_years": 5
            },
            "performance": {
                "current_rating": "Exceeds Expectations",
                "trajectory": "upward"
            },
            "goals": [
                {"goal": "VP readiness", "progress": 45},
                {"goal": "Org-wide initiative leadership", "progress": 70}
            ],
            "development": {
                "strengths": ["Technical vision", "Team building"],
                "areas": ["Executive communication", "Board presentation"]
            }
        }, indent=2)


@tool
def analyze_leadership_style(user_id: str) -> str:
    """Analyze leadership style and effectiveness."""
    with timed_operation("analyze_leadership_style"):
        return json.dumps({
            "dominant_style": "Coaching",
            "secondary_style": "Visionary",
            "effectiveness_score": 82,
            "team_feedback": {
                "psychological_safety": 4.2,
                "clarity_of_direction": 4.5,
                "growth_support": 4.0
            },
            "recommendations": [
                "Increase delegation to stretch reports",
                "More structured 1:1s with skip levels"
            ]
        }, indent=2)


@tool
def get_succession_planning_data(user_id: str) -> str:
    """Get succession planning and org readiness data."""
    with timed_operation("get_succession_planning_data"):
        return json.dumps({
            "ready_now_successors": 0,
            "ready_in_1_year": 1,
            "bench_strength": "moderate",
            "critical_role": True,
            "flight_risk": "low",
            "recommendations": [
                "Identify and develop second successor",
                "Document institutional knowledge"
            ]
        }, indent=2)


@tool
def create_development_plan(
    user_id: str,
    focus_areas: str,
    timeline_months: int,
    milestones: str
) -> str:
    """Create a formal development plan."""
    with timed_operation("create_development_plan"):
        return json.dumps({
            "plan_id": f"dev_plan_{datetime.now().strftime('%Y%m%d')}",
            "status": "created",
            "focus_areas": focus_areas.split(","),
            "timeline_months": timeline_months,
            "milestones": milestones.split(","),
            "next_review_date": "2024-03-01"
        }, indent=2)


TOOLS = [
    get_comprehensive_profile,
    analyze_leadership_style,
    get_succession_planning_data,
    create_development_plan
]


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def create_supervisor_node():
    """Create the supervisor that orchestrates specialist agents."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, callbacks=[ObservabilityCallback()])
    
    SUPERVISOR_PROMPT = """You are a Supervisor Agent coordinating an HR coaching session.

Available specialists:
1. "profile_analyst" - Retrieves and analyzes employee data
2. "leadership_coach" - Provides leadership development guidance
3. "career_strategist" - Strategic career planning
4. "synthesizer" - Combines insights into actionable coaching

Current agent outputs:
{agent_outputs}

Iteration: {iteration}/{max_iterations}

Based on the conversation and work done so far, decide:
1. Which specialist should act next (or "FINISH" if complete)
2. Your reasoning
3. Confidence (0-1)

Respond in JSON:
{{"next_agent": "agent_name or FINISH", "reasoning": "why", "confidence": 0.8}}"""

    @with_retry(max_attempts=2)
    def supervisor(state: CoachingState) -> dict:
        with timed_operation("supervisor"):
            # Check iteration limit
            if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
                logger.warning("Max iterations reached, forcing finish")
                return {
                    "supervisor_decision": {
                        "next_agent": "FINISH",
                        "reasoning": "Max iterations reached",
                        "confidence": 1.0
                    }
                }
            
            prompt = SUPERVISOR_PROMPT.format(
                agent_outputs=json.dumps(state.get("agent_outputs", {}), indent=2),
                iteration=state.get("iteration_count", 0),
                max_iterations=state.get("max_iterations", 5)
            )
            
            messages = [SystemMessage(content=prompt)] + state["messages"]
            
            try:
                response = llm.invoke(messages)
                decision = json.loads(response.content)
            except json.JSONDecodeError:
                decision = {
                    "next_agent": "synthesizer",
                    "reasoning": "Failed to parse, defaulting to synthesis",
                    "confidence": 0.5
                }
            except Exception as e:
                metrics.record_error("supervisor_error")
                decision = {
                    "next_agent": "FINISH",
                    "reasoning": f"Error: {str(e)}",
                    "confidence": 0.0
                }
            
            return {
                "supervisor_decision": decision,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    return supervisor


# ============================================================
# SPECIALIST NODES
# ============================================================

def create_specialist_node(specialist_type: str, specialist_prompt: str, tools: list):
    """Factory for creating specialist agent nodes."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, callbacks=[ObservabilityCallback()])
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    @with_retry(max_attempts=2)
    def specialist(state: CoachingState) -> dict:
        with timed_operation(specialist_type):
            try:
                messages = [SystemMessage(content=specialist_prompt)] + state["messages"]
                response = llm_with_tools.invoke(messages)
                
                # Store output
                outputs = state.get("agent_outputs", {})
                outputs[specialist_type] = {
                    "content": response.content,
                    "timestamp": datetime.now().isoformat(),
                    "has_tool_calls": bool(getattr(response, "tool_calls", None))
                }
                
                return {
                    "messages": [response],
                    "agent_outputs": outputs
                }
                
            except Exception as e:
                logger.error(f"Specialist {specialist_type} failed: {e}")
                metrics.record_error(f"{specialist_type}_error")
                
                fallback_msg = AIMessage(
                    content=GracefulDegradation.get_fallback(
                        "tool_failure",
                        f"Specialist: {specialist_type}"
                    )
                )
                
                return {
                    "messages": [fallback_msg],
                    "degraded_mode": True,
                    "last_error": {
                        "error_type": "specialist_failure",
                        "message": str(e),
                        "recoverable": True,
                        "suggested_action": "Continue with available specialists"
                    }
                }
    
    return specialist


# Specialist configurations
SPECIALISTS = {
    "profile_analyst": {
        "prompt": """You are a Profile Analyst. Retrieve and analyze employee data to understand their current situation, performance history, and development trajectory. Use tools to gather comprehensive data.""",
        "tools": [get_comprehensive_profile]
    },
    "leadership_coach": {
        "prompt": """You are a Leadership Coach specializing in executive development. Analyze leadership style, identify growth opportunities, and provide actionable guidance for leadership effectiveness.""",
        "tools": [analyze_leadership_style]
    },
    "career_strategist": {
        "prompt": """You are a Career Strategist. Focus on long-term career planning, succession readiness, and strategic positioning within the organization and industry.""",
        "tools": [get_succession_planning_data]
    },
    "synthesizer": {
        "prompt": """You are a Synthesis Agent. Combine insights from other specialists into a coherent, actionable coaching response. Create a unified narrative that addresses the user's needs comprehensively.""",
        "tools": [create_development_plan]
    }
}


def route_from_supervisor(state: CoachingState) -> str:
    """Route based on supervisor decision."""
    decision = state.get("supervisor_decision", {})
    next_agent = decision.get("next_agent", "FINISH")
    
    if next_agent == "FINISH" or next_agent not in SPECIALISTS:
        return "final_response"
    
    return next_agent


def check_tools_needed(state: CoachingState) -> Literal["tools", "supervisor"]:
    """Check if tool execution is needed."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "supervisor"


# ============================================================
# QUALITY ASSURANCE
# ============================================================

def quality_check(state: CoachingState) -> dict:
    """Assess response quality and flag for human review if needed."""
    
    with timed_operation("quality_check"):
        # Simple heuristics (would be model-based in production)
        outputs = state.get("agent_outputs", {})
        
        quality_score = 0.5  # Base score
        
        # More specialists consulted = higher quality
        quality_score += min(len(outputs) * 0.1, 0.3)
        
        # Check for errors
        if state.get("degraded_mode"):
            quality_score -= 0.2
        
        # Check iteration efficiency
        iterations = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)
        if iterations < max_iter * 0.5:
            quality_score += 0.1
        
        # Flag for review if quality is low
        needs_review = quality_score < 0.6 or state.get("degraded_mode", False)
        
        if needs_review:
            logger.warning(f"Response flagged for human review. Quality: {quality_score}")
        
        return {
            "response_quality_score": min(quality_score, 1.0),
            "needs_human_review": needs_review
        }


def generate_final_response(state: CoachingState) -> dict:
    """Generate the final coaching response."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, callbacks=[ObservabilityCallback()])
    
    with timed_operation("final_response"):
        outputs = state.get("agent_outputs", {})
        
        prompt = f"""Generate a final coaching response based on all specialist analyses.

Specialist Outputs:
{json.dumps(outputs, indent=2)}

Quality Score: {state.get('response_quality_score', 'N/A')}
Degraded Mode: {state.get('degraded_mode', False)}

Create a warm, professional coaching response that:
1. Acknowledges the user's situation
2. Synthesizes key insights
3. Provides 2-3 prioritized recommendations
4. Offers clear next steps

If in degraded mode, acknowledge limitations gracefully."""
        
        messages = state["messages"] + [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Add quality indicator if needed
        content = response.content
        if state.get("needs_human_review"):
            content += "\n\n_[This response has been flagged for quality review]_"
        
        return {"messages": [AIMessage(content=content)]}


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_production_coach():
    """Build the production-grade coaching system."""
    
    graph = StateGraph(CoachingState)
    
    # Add supervisor
    graph.add_node("supervisor", create_supervisor_node())
    
    # Add specialists
    for name, config in SPECIALISTS.items():
        graph.add_node(
            name,
            create_specialist_node(name, config["prompt"], config["tools"])
        )
    
    # Add tool node
    graph.add_node("tools", ToolNode(TOOLS))
    
    # Add quality and response nodes
    graph.add_node("quality_check", quality_check)
    graph.add_node("final_response", generate_final_response)
    
    # Define flow
    graph.add_edge(START, "supervisor")
    
    # Supervisor routes to specialists
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "profile_analyst": "profile_analyst",
            "leadership_coach": "leadership_coach",
            "career_strategist": "career_strategist",
            "synthesizer": "synthesizer",
            "final_response": "quality_check"
        }
    )
    
    # Each specialist either needs tools or goes back to supervisor
    for name in SPECIALISTS.keys():
        graph.add_conditional_edges(
            name,
            check_tools_needed,
            {
                "tools": "tools",
                "supervisor": "supervisor"
            }
        )
    
    # Tools go back to supervisor
    graph.add_edge("tools", "supervisor")
    
    # Quality check leads to final response
    graph.add_edge("quality_check", "final_response")
    graph.add_edge("final_response", END)
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the production coaching system."""
    
    print("=" * 60)
    print("PRODUCTION HR COACHING SYSTEM")
    print("=" * 60)
    print("\nFeatures:")
    print("  • Supervisor agent orchestration")
    print("  • 4 specialist agents")
    print("  • Retry with exponential backoff")
    print("  • Graceful degradation")
    print("  • Quality scoring")
    print("  • Comprehensive logging")
    print("\nCommands:")
    print("  'metrics' - Show system metrics")
    print("  'quit' - Exit")
    print("=" * 60 + "\n")
    
    coach = build_production_coach()
    
    session_id = f"prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    state = {
        "messages": [],
        "user_id": "user_505",
        "session_id": session_id,
        "supervisor_decision": None,
        "agent_outputs": {},
        "iteration_count": 0,
        "max_iterations": 6,
        "last_error": None,
        "degraded_mode": False,
        "response_quality_score": 0.0,
        "needs_human_review": False
    }
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "metrics":
            print("\n=== System Metrics ===")
            print(json.dumps(metrics.get_summary(), indent=2))
            print()
            continue
        
        state["messages"].append(HumanMessage(content=user_input))
        
        # Reset per-turn state
        state["agent_outputs"] = {}
        state["iteration_count"] = 0
        state["supervisor_decision"] = None
        
        start_time = time.time()
        
        try:
            print("\n[Processing with supervisor orchestration...]")
            result = coach.invoke(state, config)
            state = result
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(latency_ms)
            
            # Display metadata
            print(f"[Iterations: {state.get('iteration_count', 0)} | ", end="")
            print(f"Quality: {state.get('response_quality_score', 0):.2f} | ", end="")
            print(f"Latency: {latency_ms:.0f}ms]")
            
            if state.get("needs_human_review"):
                print("[ Flagged for human review]")
            
            # Display response
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 50:
                    print(f"\nCoach: {msg.content}\n")
                    break
                    
        except Exception as e:
            logger.error(f"Session error: {e}")
            logger.error(traceback.format_exc())
            metrics.record_error("session_error")
            print(f"\nCoach: {GracefulDegradation.get_fallback('llm_failure')}\n")
    
    print("\n=== Session Complete ===")
    print(f"Session ID: {session_id}")
    print(json.dumps(metrics.get_summary(), indent=2))


if __name__ == "__main__":
    main()
