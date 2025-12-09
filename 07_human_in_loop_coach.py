"""
Script 7: HR Coach with Human-in-the-Loop
Demonstrates: Interrupts, approval workflows, human review for sensitive actions
"""

from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
import json
from datetime import datetime


class PendingAction(TypedDict):
    id: str
    action_type: str
    description: str
    parameters: dict
    requires_approval: bool
    approved: Optional[bool]
    approver: Optional[str]


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    pending_actions: list[PendingAction]
    completed_actions: list[dict]
    awaiting_human_input: bool
    human_input_type: str  # "approval", "clarification", "review"


# ============================================================
# TOOLS WITH APPROVAL REQUIREMENTS
# ============================================================

@tool
def draft_pip_plan(
    user_id: str,
    performance_issues: str,
    improvement_goals: str,
    timeline_weeks: int
) -> str:
    """Draft a Performance Improvement Plan. REQUIRES MANAGER APPROVAL."""
    return json.dumps({
        "status": "draft_created",
        "requires_approval": True,
        "approver_role": "manager",
        "plan": {
            "employee_id": user_id,
            "issues": performance_issues,
            "goals": improvement_goals,
            "timeline_weeks": timeline_weeks,
            "created": datetime.now().isoformat()
        }
    }, indent=2)


@tool
def schedule_skip_level(
    user_id: str,
    topic: str,
    urgency: str
) -> str:
    """Schedule a skip-level meeting. REQUIRES HR REVIEW for sensitive topics."""
    sensitive_topics = ["harassment", "discrimination", "retaliation", "ethics"]
    needs_review = any(t in topic.lower() for t in sensitive_topics)
    
    return json.dumps({
        "status": "pending" if needs_review else "scheduled",
        "requires_hr_review": needs_review,
        "meeting": {
            "type": "skip_level",
            "topic": topic,
            "urgency": urgency
        }
    }, indent=2)


@tool
def submit_promotion_nomination(
    user_id: str,
    target_role: str,
    justification: str,
    supporting_evidence: str
) -> str:
    """Submit a promotion nomination. REQUIRES CALIBRATION REVIEW."""
    return json.dumps({
        "status": "nomination_submitted",
        "requires_approval": True,
        "approval_process": "calibration_committee",
        "nomination": {
            "employee_id": user_id,
            "target_role": target_role,
            "justification": justification,
            "evidence": supporting_evidence
        }
    }, indent=2)


@tool
def update_compensation_discussion(
    user_id: str,
    discussion_notes: str,
    requested_adjustment: str
) -> str:
    """Log compensation discussion. REQUIRES MANAGER + HR APPROVAL."""
    return json.dumps({
        "status": "logged_pending_approval",
        "requires_approval": True,
        "approvers": ["manager", "hr_partner"],
        "discussion": {
            "notes": discussion_notes,
            "requested_adjustment": requested_adjustment
        }
    }, indent=2)


@tool
def create_learning_plan(user_id: str, skills: str, resources: str) -> str:
    """Create a learning plan. No approval required."""
    return json.dumps({
        "status": "created",
        "requires_approval": False,
        "plan": {"skills": skills, "resources": resources}
    }, indent=2)


TOOLS = [
    draft_pip_plan,
    schedule_skip_level,
    submit_promotion_nomination,
    update_compensation_discussion,
    create_learning_plan
]

# Actions that require human approval
APPROVAL_REQUIRED_TOOLS = {
    "draft_pip_plan": {"approver": "manager", "reason": "Legal and HR implications"},
    "submit_promotion_nomination": {"approver": "calibration_committee", "reason": "Budget and equity"},
    "update_compensation_discussion": {"approver": "manager_and_hr", "reason": "Compensation decisions"}
}


# ============================================================
# GRAPH NODES
# ============================================================

def create_coach_node(llm_with_tools):
    """Main coaching node with tool access."""
    
    def coach(state: CoachingState) -> dict:
        system = """You are an HR Coach with access to organizational systems.

IMPORTANT: Some actions require human approval before execution:
- Performance Improvement Plans → Manager approval
- Promotion nominations → Calibration review  
- Compensation discussions → Manager + HR approval
- Skip-level meetings on sensitive topics → HR review

When proposing such actions, clearly explain what will happen and that approval is needed.
For routine actions like learning plans, proceed directly."""
        
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    return coach


def check_tool_approval(state: CoachingState) -> dict:
    """Check if any tool calls require approval."""
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"pending_actions": [], "awaiting_human_input": False}
    
    pending = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        
        if tool_name in APPROVAL_REQUIRED_TOOLS:
            pending.append({
                "id": tool_call["id"],
                "action_type": tool_name,
                "description": f"Execute {tool_name}",
                "parameters": tool_call["args"],
                "requires_approval": True,
                "approved": None,
                "approver": APPROVAL_REQUIRED_TOOLS[tool_name]["approver"]
            })
    
    return {
        "pending_actions": pending,
        "awaiting_human_input": len(pending) > 0,
        "human_input_type": "approval" if pending else ""
    }


def request_human_approval(state: CoachingState) -> dict:
    """Interrupt execution to request human approval."""
    pending = state.get("pending_actions", [])
    
    if not pending:
        return {}
    
    # Format approval request
    approval_requests = []
    for action in pending:
        approval_requests.append({
            "action_id": action["id"],
            "action_type": action["action_type"],
            "parameters": action["parameters"],
            "required_approver": action["approver"],
            "reason": APPROVAL_REQUIRED_TOOLS.get(action["action_type"], {}).get("reason", "Policy requirement")
        })
    
    # Create the interrupt - this pauses execution
    human_response = interrupt({
        "type": "approval_request",
        "message": "The following actions require approval before proceeding:",
        "actions": approval_requests,
        "instructions": "Respond with: {'approved': [action_ids], 'rejected': [action_ids], 'comments': 'optional'}"
    })
    
    # Process human response
    approved_ids = human_response.get("approved", [])
    rejected_ids = human_response.get("rejected", [])
    
    updated_pending = []
    for action in pending:
        if action["id"] in approved_ids:
            action["approved"] = True
            action["approver"] = human_response.get("approver_name", "human")
        elif action["id"] in rejected_ids:
            action["approved"] = False
        updated_pending.append(action)
    
    return {
        "pending_actions": updated_pending,
        "awaiting_human_input": False
    }


def execute_approved_actions(state: CoachingState) -> dict:
    """Execute only approved actions."""
    pending = state.get("pending_actions", [])
    completed = state.get("completed_actions", [])
    
    # Filter to approved actions only
    approved_actions = [a for a in pending if a.get("approved") is True]
    rejected_actions = [a for a in pending if a.get("approved") is False]
    
    # In real implementation, execute the approved tool calls
    for action in approved_actions:
        completed.append({
            "action_id": action["id"],
            "action_type": action["action_type"],
            "status": "executed",
            "timestamp": datetime.now().isoformat()
        })
    
    # Generate summary message
    summary_parts = []
    if approved_actions:
        summary_parts.append(f" {len(approved_actions)} action(s) approved and executed")
    if rejected_actions:
        summary_parts.append(f" {len(rejected_actions)} action(s) were not approved")
    
    summary_msg = AIMessage(content="\n".join(summary_parts) if summary_parts else "No actions to process.")
    
    return {
        "messages": [summary_msg],
        "pending_actions": [],
        "completed_actions": completed
    }


def route_after_check(state: CoachingState) -> Literal["request_approval", "tools", "end"]:
    """Route based on approval requirements."""
    last_message = state["messages"][-1]
    
    # Check if there are pending actions needing approval
    if state.get("awaiting_human_input"):
        return "request_approval"
    
    # Check if there are tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


def route_after_approval(state: CoachingState) -> Literal["execute", "respond"]:
    """Route after human approval decision."""
    pending = state.get("pending_actions", [])
    has_approved = any(a.get("approved") for a in pending)
    
    if has_approved:
        return "execute"
    return "respond"


def generate_rejection_response(state: CoachingState) -> dict:
    """Generate response when actions are rejected."""
    rejected = [a for a in state.get("pending_actions", []) if a.get("approved") is False]
    
    response = f"""I understand. The following actions were not approved:
{chr(10).join(['- ' + a['action_type'] for a in rejected])}

Would you like to:
1. Discuss alternative approaches?
2. Provide more context for reconsideration?
3. Focus on a different aspect of your development?"""
    
    return {"messages": [AIMessage(content=response)], "pending_actions": []}


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_human_in_loop_graph():
    """Build the graph with human-in-the-loop approval."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(TOOLS)
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("coach", create_coach_node(llm_with_tools))
    graph.add_node("check_approval", check_tool_approval)
    graph.add_node("request_approval", request_human_approval)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("execute", execute_approved_actions)
    graph.add_node("respond_rejected", generate_rejection_response)
    
    # Define flow
    graph.add_edge(START, "coach")
    graph.add_edge("coach", "check_approval")
    
    graph.add_conditional_edges(
        "check_approval",
        route_after_check,
        {
            "request_approval": "request_approval",
            "tools": "tools",
            "end": END
        }
    )
    
    graph.add_conditional_edges(
        "request_approval",
        route_after_approval,
        {
            "execute": "execute",
            "respond": "respond_rejected"
        }
    )
    
    graph.add_edge("tools", "coach")
    graph.add_edge("execute", END)
    graph.add_edge("respond_rejected", END)
    
    # Add checkpointer for interrupt/resume
    checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)


def main():
    """Demo the human-in-the-loop system."""
    
    graph = build_human_in_loop_graph()
    thread_id = "hitl_session_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "messages": [],
        "user_id": "user_202",
        "pending_actions": [],
        "completed_actions": [],
        "awaiting_human_input": False,
        "human_input_type": ""
    }
    
    print("=== HR Coach with Human-in-the-Loop ===")
    print("Sensitive actions require approval before execution")
    print("\nTry: 'I want to submit a promotion nomination'")
    print("Or: 'Help me create a learning plan' (no approval needed)")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            result = graph.invoke(state, config)
            state = result
            
            # Display response
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nCoach: {msg.content}\n")
                    break
                    
        except Exception as e:
            # Handle interrupt for approval
            if "interrupt" in str(type(e).__name__).lower():
                print("\n APPROVAL REQUIRED ")
                print("This action requires human approval.")
                print("In production, this would pause and await manager/HR approval.\n")
                
                # Simulate approval (in production, this would be async)
                approve = input("Approve? (y/n): ").strip().lower()
                
                # Resume with approval decision
                if approve == 'y':
                    # Get pending action IDs and approve them
                    graph_state = graph.get_state(config)
                    pending = graph_state.values.get("pending_actions", [])
                    approved_ids = [a["id"] for a in pending]
                    
                    result = graph.invoke(
                        Command(resume={"approved": approved_ids, "rejected": []}),
                        config
                    )
                else:
                    graph_state = graph.get_state(config)
                    pending = graph_state.values.get("pending_actions", [])
                    rejected_ids = [a["id"] for a in pending]
                    
                    result = graph.invoke(
                        Command(resume={"approved": [], "rejected": rejected_ids}),
                        config
                    )
                
                state = result
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        print(f"\nCoach: {msg.content}\n")
                        break
            else:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
