"""
29. Command Routing in LangGraph
=================================
Demonstrates LangGraph's Command object for dynamic control flow:

COMMAND FEATURES:
├── goto: Navigate to specific nodes dynamically
├── update: Modify state from within a node
├── resume: Continue after an interrupt
├── graph: Target subgraphs (for nested graphs)
└── Combined: goto + update in one operation

USE CASES:
- Dynamic routing based on LLM decisions
- Error recovery and retry logic
- Conditional branching without explicit edges
- State updates with navigation
- Complex workflow orchestration
"""

from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# CONFIGURATION
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# STATE DEFINITIONS
# ============================================================

class RouterState(TypedDict):
    """State for routing demo"""
    query: str
    category: str
    response: str
    route_history: list[str]


class RetryState(TypedDict):
    """State for retry demo"""
    task: str
    attempts: int
    max_attempts: int
    result: str
    errors: list[str]


class EscalationState(TypedDict):
    """State for escalation demo"""
    issue: str
    severity: str
    current_tier: int
    resolution: str
    escalation_path: list[str]


# ============================================================
# DEMO 1: BASIC COMMAND.GOTO - DYNAMIC ROUTING
# ============================================================

def build_dynamic_router():
    """Build a router that uses Command for dynamic navigation"""
    
    def classify_query(state: RouterState) -> Command[Literal["technical", "billing", "general", "escalate"]]:
        """Classify the query and route dynamically using Command"""
        log(f"Classifying query: {state['query'][:50]}...", "")
        
        # Use LLM to classify
        response = llm.invoke([
            SystemMessage(content="""Classify this customer query into ONE category:
- technical: Tech support, bugs, how-to questions
- billing: Payments, invoices, subscriptions
- general: General questions, feedback
- escalate: Complaints, urgent issues, refund requests

Respond with ONLY the category name."""),
            HumanMessage(content=state["query"])
        ])
        
        category = response.content.strip().lower()
        log(f"Classified as: {category}", "")
        
        # Validate category
        valid_categories = ["technical", "billing", "general", "escalate"]
        if category not in valid_categories:
            category = "general"
        
        # Use Command to both update state AND navigate
        return Command(
            goto=category,
            update={
                "category": category,
                "route_history": state.get("route_history", []) + ["classify"]
            }
        )
    
    def handle_technical(state: RouterState) -> dict:
        """Handle technical queries"""
        log("Handling technical query...", "")
        
        response = llm.invoke([
            SystemMessage(content="You are a tech support specialist. Give a brief, helpful response."),
            HumanMessage(content=state["query"])
        ])
        
        return {
            "response": response.content,
            "route_history": state.get("route_history", []) + ["technical"]
        }
    
    def handle_billing(state: RouterState) -> dict:
        """Handle billing queries"""
        log("Handling billing query...", "")
        
        response = llm.invoke([
            SystemMessage(content="You are a billing specialist. Give a brief, helpful response."),
            HumanMessage(content=state["query"])
        ])
        
        return {
            "response": response.content,
            "route_history": state.get("route_history", []) + ["billing"]
        }
    
    def handle_general(state: RouterState) -> dict:
        """Handle general queries"""
        log("Handling general query...", "")
        
        response = llm.invoke([
            SystemMessage(content="You are a helpful customer service rep. Give a brief response."),
            HumanMessage(content=state["query"])
        ])
        
        return {
            "response": response.content,
            "route_history": state.get("route_history", []) + ["general"]
        }
    
    def handle_escalation(state: RouterState) -> dict:
        """Handle escalated queries"""
        log("Handling escalated query...", "")
        
        response = llm.invoke([
            SystemMessage(content="""You are a senior customer service manager.
This query was flagged for escalation. Provide a thoughtful, apologetic response
and explain next steps for resolution."""),
            HumanMessage(content=state["query"])
        ])
        
        return {
            "response": response.content,
            "route_history": state.get("route_history", []) + ["escalate"]
        }
    
    # Build graph
    workflow = StateGraph(RouterState)
    
    workflow.add_node("classify", classify_query)
    workflow.add_node("technical", handle_technical)
    workflow.add_node("billing", handle_billing)
    workflow.add_node("general", handle_general)
    workflow.add_node("escalate", handle_escalation)
    
    workflow.add_edge(START, "classify")
    # No conditional edges needed! Command.goto handles routing
    
    # All handler nodes go to END
    workflow.add_edge("technical", END)
    workflow.add_edge("billing", END)
    workflow.add_edge("general", END)
    workflow.add_edge("escalate", END)
    
    return workflow.compile()


def demo_dynamic_routing():
    """Demo: Dynamic routing with Command.goto"""
    print("\n" + "="*70)
    print(" DEMO 1: COMMAND.GOTO - Dynamic Routing")
    print("="*70)
    
    graph = build_dynamic_router()
    
    queries = [
        "How do I reset my password?",
        "I want a refund! Your service is terrible!",
        "What payment methods do you accept?",
        "What are your business hours?"
    ]
    
    for query in queries:
        print(f"\n Query: '{query[:50]}...'")
        
        result = graph.invoke({
            "query": query,
            "route_history": []
        })
        
        print(f"   Category: {result['category']}")
        print(f"   Route: {' → '.join(result['route_history'])}")
        print(f"   Response: {result['response'][:100]}...")


# ============================================================
# DEMO 2: COMMAND WITH RETRY LOGIC
# ============================================================

def build_retry_workflow():
    """Build a workflow with retry capability using Command"""
    
    retry_count = {"count": 0}  # Mutable state for demo
    
    def execute_task(state: RetryState) -> Command[Literal["success", "retry", "failed"]]:
        """Execute task with retry logic"""
        attempts = state.get("attempts", 0) + 1
        max_attempts = state.get("max_attempts", 3)
        
        log(f"Attempt {attempts}/{max_attempts}...", "⚙️")
        
        # Simulate intermittent failure
        retry_count["count"] += 1
        success = retry_count["count"] >= 2  # Succeed on 2nd attempt
        
        if success:
            log("Task succeeded!", "")
            return Command(
                goto="success",
                update={
                    "attempts": attempts,
                    "result": "Task completed successfully"
                }
            )
        
        if attempts >= max_attempts:
            log("Max attempts reached, failing...", "")
            return Command(
                goto="failed",
                update={
                    "attempts": attempts,
                    "errors": state.get("errors", []) + [f"Attempt {attempts} failed"]
                }
            )
        
        log("Task failed, will retry...", "")
        return Command(
            goto="retry",
            update={
                "attempts": attempts,
                "errors": state.get("errors", []) + [f"Attempt {attempts} failed"]
            }
        )
    
    def handle_retry(state: RetryState) -> Command[Literal["execute"]]:
        """Handle retry - just go back to execute"""
        log("Retrying...", "")
        return Command(goto="execute")
    
    def handle_success(state: RetryState) -> dict:
        """Handle successful completion"""
        return {"result": f"SUCCESS after {state['attempts']} attempt(s)"}
    
    def handle_failure(state: RetryState) -> dict:
        """Handle failure after all retries"""
        return {"result": f"FAILED after {state['attempts']} attempts"}
    
    workflow = StateGraph(RetryState)
    
    workflow.add_node("execute", execute_task)
    workflow.add_node("retry", handle_retry)
    workflow.add_node("success", handle_success)
    workflow.add_node("failed", handle_failure)
    
    workflow.add_edge(START, "execute")
    # Command handles routing to retry/success/failed
    workflow.add_edge("success", END)
    workflow.add_edge("failed", END)
    
    return workflow.compile()


def demo_retry_pattern():
    """Demo: Retry pattern with Command"""
    print("\n" + "="*70)
    print(" DEMO 2: COMMAND FOR RETRY LOGIC")
    print("="*70)
    
    graph = build_retry_workflow()
    
    print("\n Executing task with retry logic...")
    
    result = graph.invoke({
        "task": "Connect to external API",
        "attempts": 0,
        "max_attempts": 3,
        "errors": []
    })
    
    print(f"\n📊 Result: {result['result']}")
    print(f"   Attempts: {result['attempts']}")
    print(f"   Errors: {result.get('errors', [])}")


# ============================================================
# DEMO 3: MULTI-LEVEL ESCALATION
# ============================================================

def build_escalation_workflow():
    """Build a tiered escalation workflow"""
    
    def triage(state: EscalationState) -> Command[Literal["tier1", "tier2", "tier3"]]:
        """Initial triage based on severity"""
        log("Triaging issue...", "")
        
        # Determine initial tier based on severity
        severity = state.get("severity", "medium").lower()
        
        tier_map = {
            "low": "tier1",
            "medium": "tier1",
            "high": "tier2",
            "critical": "tier3"
        }
        
        initial_tier = tier_map.get(severity, "tier1")
        tier_num = int(initial_tier[-1])
        
        log(f"Severity '{severity}' → Starting at {initial_tier}", "")
        
        return Command(
            goto=initial_tier,
            update={
                "current_tier": tier_num,
                "escalation_path": ["triage"]
            }
        )
    
    def tier1_support(state: EscalationState) -> Command[Literal["tier2", "resolved"]]:
        """Tier 1 - Basic support"""
        log("Tier 1 support handling...", "")
        
        issue = state.get("issue", "").lower()
        
        # Simple issues resolved at tier 1
        if any(word in issue for word in ["password", "login", "access"]):
            log("Resolved at Tier 1", "")
            return Command(
                goto="resolved",
                update={
                    "resolution": "Password reset link sent. Please check your email.",
                    "escalation_path": state.get("escalation_path", []) + ["tier1"]
                }
            )
        
        # Escalate complex issues
        log("Escalating to Tier 2...", "")
        return Command(
            goto="tier2",
            update={
                "current_tier": 2,
                "escalation_path": state.get("escalation_path", []) + ["tier1"]
            }
        )
    
    def tier2_support(state: EscalationState) -> Command[Literal["tier3", "resolved"]]:
        """Tier 2 - Technical support"""
        log("Tier 2 technical support handling...", "👨‍💻")
        
        issue = state.get("issue", "").lower()
        severity = state.get("severity", "").lower()
        
        # Most issues resolved at tier 2
        if severity != "critical":
            response = llm.invoke([
                SystemMessage(content="You are a technical support specialist. Briefly explain how to resolve this issue."),
                HumanMessage(content=issue)
            ])
            
            log("Resolved at Tier 2", "")
            return Command(
                goto="resolved",
                update={
                    "resolution": response.content,
                    "escalation_path": state.get("escalation_path", []) + ["tier2"]
                }
            )
        
        # Critical issues go to tier 3
        log("Escalating to Tier 3...", "")
        return Command(
            goto="tier3",
            update={
                "current_tier": 3,
                "escalation_path": state.get("escalation_path", []) + ["tier2"]
            }
        )
    
    def tier3_support(state: EscalationState) -> Command[Literal["resolved"]]:
        """Tier 3 - Senior/Executive support"""
        log("Tier 3 senior support handling...", "👔")
        
        response = llm.invoke([
            SystemMessage(content="""You are a senior support executive. This is a critical issue.
Provide a comprehensive resolution and any compensation offered."""),
            HumanMessage(content=f"Issue: {state['issue']}\nSeverity: {state['severity']}")
        ])
        
        log("Resolved at Tier 3", "")
        return Command(
            goto="resolved",
            update={
                "resolution": response.content,
                "escalation_path": state.get("escalation_path", []) + ["tier3"]
            }
        )
    
    def resolved(state: EscalationState) -> dict:
        """Issue resolved"""
        return {
            "escalation_path": state.get("escalation_path", []) + ["resolved"]
        }
    
    workflow = StateGraph(EscalationState)
    
    workflow.add_node("triage", triage)
    workflow.add_node("tier1", tier1_support)
    workflow.add_node("tier2", tier2_support)
    workflow.add_node("tier3", tier3_support)
    workflow.add_node("resolved", resolved)
    
    workflow.add_edge(START, "triage")
    workflow.add_edge("resolved", END)
    
    return workflow.compile()


def demo_escalation():
    """Demo: Multi-level escalation with Command"""
    print("\n" + "="*70)
    print(" DEMO 3: MULTI-LEVEL ESCALATION")
    print("="*70)
    
    graph = build_escalation_workflow()
    
    test_cases = [
        {"issue": "I forgot my password", "severity": "low"},
        {"issue": "The checkout page keeps crashing", "severity": "high"},
        {"issue": "Data breach affecting customer information!", "severity": "critical"}
    ]
    
    for case in test_cases:
        print(f"\n Issue: '{case['issue'][:40]}...'")
        print(f"   Severity: {case['severity']}")
        
        result = graph.invoke(case)
        
        print(f"   Escalation Path: {' → '.join(result['escalation_path'])}")
        print(f"   Resolution: {result['resolution'][:80]}...")


# ============================================================
# DEMO 4: CONDITIONAL LOOPS WITH COMMAND
# ============================================================

class QualityState(TypedDict):
    """State for quality control workflow"""
    content: str
    quality_score: int
    revision_count: int
    max_revisions: int
    final_content: str


def build_quality_loop():
    """Build a quality control loop using Command"""
    
    def generate_content(state: QualityState) -> dict:
        """Generate or revise content"""
        revision = state.get("revision_count", 0)
        
        if revision == 0:
            log("Generating initial content...", "")
            prompt = "Generate a brief product description for a smart water bottle."
        else:
            log(f"Revision {revision}: Improving content...", "✏️")
            prompt = f"Improve this content (current score: {state.get('quality_score', 0)}/100):\n{state['content']}"
        
        response = llm.invoke([
            SystemMessage(content="You are a copywriter. Write concise, engaging content."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "content": response.content,
            "revision_count": revision + 1
        }
    
    def evaluate_quality(state: QualityState) -> Command[Literal["revise", "approve"]]:
        """Evaluate content quality and decide next step"""
        log("Evaluating quality...", "")
        
        # Use LLM to score
        response = llm.invoke([
            SystemMessage(content="""Score this content from 0-100 based on:
- Clarity (0-25)
- Engagement (0-25)
- Accuracy (0-25)
- Conciseness (0-25)

Respond with ONLY a number."""),
            HumanMessage(content=state["content"])
        ])
        
        try:
            score = int(response.content.strip())
        except:
            score = 50
        
        log(f"Quality score: {score}/100", "📊")
        
        max_revisions = state.get("max_revisions", 3)
        revision_count = state.get("revision_count", 1)
        
        if score >= 80:
            log("Content approved!", "")
            return Command(
                goto="approve",
                update={"quality_score": score}
            )
        elif revision_count >= max_revisions:
            log("Max revisions reached, accepting current version", "")
            return Command(
                goto="approve",
                update={"quality_score": score}
            )
        else:
            log("Quality below threshold, revising...", "")
            return Command(
                goto="revise",
                update={"quality_score": score}
            )
    
    def revise(state: QualityState) -> Command[Literal["generate"]]:
        """Trigger revision"""
        return Command(goto="generate")
    
    def approve(state: QualityState) -> dict:
        """Approve final content"""
        return {"final_content": state["content"]}
    
    workflow = StateGraph(QualityState)
    
    workflow.add_node("generate", generate_content)
    workflow.add_node("evaluate", evaluate_quality)
    workflow.add_node("revise", revise)
    workflow.add_node("approve", approve)
    
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("approve", END)
    
    return workflow.compile()


def demo_quality_loop():
    """Demo: Quality control loop with Command"""
    print("\n" + "="*70)
    print("🔁 DEMO 4: QUALITY CONTROL LOOP")
    print("="*70)
    
    graph = build_quality_loop()
    
    print("\n Starting content generation with quality control...")
    
    result = graph.invoke({
        "content": "",
        "quality_score": 0,
        "revision_count": 0,
        "max_revisions": 3
    })
    
    print(f"\n📊 Final Results:")
    print(f"   Revisions: {result['revision_count']}")
    print(f"   Final Score: {result['quality_score']}/100")
    print(f"   Content: {result['final_content'][:200]}...")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            LangGraph Command Routing Demo                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Command Features:                                               ║
║  • goto: Navigate to specific nodes                              ║
║  • update: Modify state during navigation                        ║
║  • Dynamic routing without explicit edges                        ║
║  • Loops and retry patterns                                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_dynamic_routing()
        demo_retry_pattern()
        demo_escalation()
        demo_quality_loop()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:

1. Command.goto for dynamic navigation:
   return Command(goto="target_node")

2. Combine navigation with state updates:
   return Command(
       goto="next_node",
       update={"field": "value"}
   )

3. Type hints for valid destinations:
   def node(state) -> Command[Literal["a", "b", "c"]]:

4. Advantages over conditional edges:
   - Runtime decisions inside the node
   - Combined state update + navigation
   - Cleaner code for complex routing
   - No need to predefine all routes

5. Common patterns:
   - Dynamic routing based on LLM classification
   - Retry loops with backoff
   - Multi-level escalation
   - Quality control loops
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
