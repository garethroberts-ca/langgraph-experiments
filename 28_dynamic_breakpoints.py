"""
28. Dynamic Breakpoints & Interrupt Patterns in LangGraph
==========================================================
Demonstrates LangGraph's interrupt capabilities for human-in-the-loop:

INTERRUPT PATTERNS:
├── interrupt_before: Pause before a node executes
├── interrupt_after: Pause after a node executes
├── interrupt(): Dynamic interrupt from within a node
├── NodeInterrupt: Interrupt with custom value
└── Resume: Continue with user-provided input

USE CASES:
- Human approval before sensitive actions
- User input collection mid-workflow
- Review and edit agent outputs
- Debugging and inspection
- Multi-step approval workflows
"""

from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# CONFIGURATION
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# STATE DEFINITIONS
# ============================================================

class ApprovalState(TypedDict):
    """State for approval workflow"""
    request: str
    analysis: str
    recommendation: str
    approved: Optional[bool]
    approver_notes: str
    final_action: str


class FeedbackState(TypedDict):
    """State for feedback collection workflow"""
    messages: list
    draft_response: str
    user_feedback: str
    final_response: str


class MultiStepState(TypedDict):
    """State for multi-step approval"""
    content: str
    reviews: list[dict]
    current_reviewer: str
    status: str


# ============================================================
# DEMO 1: INTERRUPT_BEFORE - APPROVAL WORKFLOW
# ============================================================

def build_approval_workflow():
    """Build a workflow that pauses for human approval"""
    
    def analyze_request(state: ApprovalState) -> dict:
        """Analyze the incoming request"""
        log("Analyzing request...", "")
        
        response = llm.invoke([
            SystemMessage(content="Analyze this request and identify any risks or concerns. Be brief."),
            HumanMessage(content=state["request"])
        ])
        
        return {"analysis": response.content}
    
    def generate_recommendation(state: ApprovalState) -> dict:
        """Generate a recommendation based on analysis"""
        log("Generating recommendation...", "")
        
        response = llm.invoke([
            SystemMessage(content="""Based on the analysis, recommend APPROVE or REJECT.
Format: RECOMMENDATION: [APPROVE/REJECT]
Reason: [brief reason]"""),
            HumanMessage(content=f"Request: {state['request']}\n\nAnalysis: {state['analysis']}")
        ])
        
        return {"recommendation": response.content}
    
    def execute_action(state: ApprovalState) -> dict:
        """Execute the approved action"""
        if state.get("approved"):
            log("Executing approved action...", "")
            return {"final_action": f"Action executed at {datetime.now().isoformat()}"}
        else:
            log("Action rejected", "")
            return {"final_action": "Action was rejected by approver"}
    
    # Build graph
    workflow = StateGraph(ApprovalState)
    
    workflow.add_node("analyze", analyze_request)
    workflow.add_node("recommend", generate_recommendation)
    workflow.add_node("execute", execute_action)
    
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "execute")
    workflow.add_edge("execute", END)
    
    # Compile with interrupt BEFORE execute node
    checkpointer = MemorySaver()
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"]  # Pause before execution
    )
    
    return graph


def demo_interrupt_before():
    """Demo: Pause before a sensitive action for approval"""
    print("\n" + "="*70)
    print(" DEMO 1: INTERRUPT_BEFORE (Approval Workflow)")
    print("="*70)
    
    graph = build_approval_workflow()
    config = {"configurable": {"thread_id": "approval-demo-1"}}
    
    # Start the workflow
    print("\n Starting workflow with request...")
    initial_state = {
        "request": "Delete all records older than 30 days from the database",
        "approved": None,
        "approver_notes": ""
    }
    
    result = graph.invoke(initial_state, config)
    
    # Check where we paused
    state = graph.get_state(config)
    print(f"\n Workflow paused at: {state.next}")
    print(f"\n Analysis: {result.get('analysis', '')[:200]}...")
    print(f"\n Recommendation: {result.get('recommendation', '')}")
    
    # Simulate human approval
    print("\n HUMAN REVIEW:")
    print("   [Simulating human approves the action]")
    
    # Update state with approval and resume
    graph.update_state(
        config,
        {"approved": True, "approver_notes": "Approved by admin - backup confirmed"}
    )
    
    # Continue execution
    print("\n Resuming workflow...")
    final_result = graph.invoke(None, config)
    
    print(f"\n Final action: {final_result.get('final_action', 'N/A')}")


# ============================================================
# DEMO 2: INTERRUPT_AFTER - POST-EXECUTION REVIEW
# ============================================================

def build_review_workflow():
    """Build a workflow that pauses after generating content for review"""
    
    def generate_content(state: FeedbackState) -> dict:
        """Generate initial content"""
        log("Generating draft response...", "")
        
        messages = state.get("messages", [])
        
        response = llm.invoke([
            SystemMessage(content="Generate a helpful, professional response."),
            *messages
        ])
        
        return {"draft_response": response.content}
    
    def finalize_response(state: FeedbackState) -> dict:
        """Finalize response based on feedback"""
        log("Finalizing response...", "")
        
        feedback = state.get("user_feedback", "")
        draft = state.get("draft_response", "")
        
        if feedback and "good" not in feedback.lower():
            # Incorporate feedback
            response = llm.invoke([
                SystemMessage(content="Revise the response based on user feedback."),
                HumanMessage(content=f"Original: {draft}\n\nFeedback: {feedback}")
            ])
            return {"final_response": response.content}
        
        return {"final_response": draft}
    
    workflow = StateGraph(FeedbackState)
    
    workflow.add_node("generate", generate_content)
    workflow.add_node("finalize", finalize_response)
    
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["generate"]  # Pause after generation for review
    )
    
    return graph


def demo_interrupt_after():
    """Demo: Pause after content generation for human review"""
    print("\n" + "="*70)
    print(" DEMO 2: INTERRUPT_AFTER (Content Review)")
    print("="*70)
    
    graph = build_review_workflow()
    config = {"configurable": {"thread_id": "review-demo-1"}}
    
    # Start workflow
    print("\n Starting content generation...")
    result = graph.invoke({
        "messages": [HumanMessage(content="Write a brief apology email for a delayed shipment")]
    }, config)
    
    state = graph.get_state(config)
    print(f"\n Paused after: generate")
    print(f"\n Draft Response:\n{result.get('draft_response', '')[:300]}...")
    
    # Simulate user feedback
    print("\n USER FEEDBACK:")
    feedback = "Make it more empathetic and offer 10% discount"
    print(f"   '{feedback}'")
    
    # Update state with feedback
    graph.update_state(config, {"user_feedback": feedback})
    
    # Resume
    print("\n Resuming with feedback...")
    final_result = graph.invoke(None, config)
    
    print(f"\n Final Response:\n{final_result.get('final_response', '')[:300]}...")


# ============================================================
# DEMO 3: DYNAMIC INTERRUPT - MID-NODE PAUSE
# ============================================================

def build_dynamic_interrupt_workflow():
    """Build a workflow with dynamic interrupts inside nodes"""
    
    def process_with_interrupt(state: ApprovalState) -> dict:
        """A node that can dynamically interrupt"""
        log("Processing request...", "⚙️")
        
        request = state.get("request", "")
        
        # Check if this is a high-risk action
        high_risk_keywords = ["delete", "remove", "drop", "terminate"]
        is_high_risk = any(kw in request.lower() for kw in high_risk_keywords)
        
        if is_high_risk:
            log("High-risk action detected! Requesting approval...", "")
            
            # Dynamic interrupt - pauses execution and returns value to user
            approval = interrupt({
                "type": "approval_required",
                "reason": "High-risk action detected",
                "request": request,
                "question": "Do you approve this action? (yes/no)"
            })
            
            # This code runs AFTER the interrupt is resolved
            if approval and approval.lower() in ["yes", "y", "approve"]:
                return {
                    "analysis": "Action approved by user",
                    "approved": True
                }
            else:
                return {
                    "analysis": "Action rejected by user",
                    "approved": False
                }
        
        # Low-risk actions proceed automatically
        return {
            "analysis": "Low-risk action, proceeding automatically",
            "approved": True
        }
    
    def complete_action(state: ApprovalState) -> dict:
        """Complete the action"""
        if state.get("approved"):
            return {"final_action": "Action completed successfully"}
        return {"final_action": "Action cancelled"}
    
    workflow = StateGraph(ApprovalState)
    
    workflow.add_node("process", process_with_interrupt)
    workflow.add_node("complete", complete_action)
    
    workflow.add_edge(START, "process")
    workflow.add_edge("process", "complete")
    workflow.add_edge("complete", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def demo_dynamic_interrupt():
    """Demo: Dynamic interrupt based on runtime conditions"""
    print("\n" + "="*70)
    print("⚡ DEMO 3: DYNAMIC INTERRUPT (Conditional)")
    print("="*70)
    
    graph = build_dynamic_interrupt_workflow()
    
    # Test 1: Low-risk action (no interrupt)
    print("\n Test 1: Low-risk action")
    print("-" * 40)
    
    config1 = {"configurable": {"thread_id": "dynamic-1"}}
    result1 = graph.invoke({
        "request": "Update user profile picture"
    }, config1)
    
    print(f"   Analysis: {result1.get('analysis')}")
    print(f"   Final: {result1.get('final_action')}")
    
    # Test 2: High-risk action (triggers interrupt)
    print("\n Test 2: High-risk action")
    print("-" * 40)
    
    config2 = {"configurable": {"thread_id": "dynamic-2"}}
    result2 = graph.invoke({
        "request": "Delete user account and all data"
    }, config2)
    
    # Check for interrupt
    state = graph.get_state(config2)
    
    if state.next:
        print("    Workflow interrupted!")
        
        # Get the interrupt value
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    for interrupt_val in task.interrupts:
                        print(f"    Interrupt data: {interrupt_val.value}")
        
        # Simulate user approval
        print("\n    User responds: 'yes'")
        
        # Resume with the response using Command
        final_result = graph.invoke(Command(resume="yes"), config2)
        print(f"\n    Final: {final_result.get('final_action')}")


# ============================================================
# DEMO 4: MULTI-STEP APPROVAL
# ============================================================

def build_multi_approval_workflow():
    """Build a workflow requiring multiple approvals"""
    
    reviewers = ["legal", "finance", "manager"]
    
    def submit_for_review(state: MultiStepState) -> dict:
        """Submit content for review"""
        log("Submitting for multi-step review...", "📤")
        return {
            "reviews": [],
            "status": "pending_review"
        }
    
    def legal_review(state: MultiStepState) -> dict:
        """Legal team review"""
        log("Legal review in progress...", "⚖️")
        
        # Interrupt for legal approval
        approval = interrupt({
            "reviewer": "legal",
            "content": state["content"][:100],
            "question": "Legal team: Approve this content?"
        })
        
        reviews = state.get("reviews", [])
        reviews.append({
            "reviewer": "legal",
            "approved": approval.lower() == "approve",
            "notes": approval
        })
        
        return {"reviews": reviews, "current_reviewer": "legal"}
    
    def finance_review(state: MultiStepState) -> dict:
        """Finance team review"""
        log("Finance review in progress...", "")
        
        approval = interrupt({
            "reviewer": "finance",
            "content": state["content"][:100],
            "question": "Finance team: Approve this content?"
        })
        
        reviews = state.get("reviews", [])
        reviews.append({
            "reviewer": "finance",
            "approved": approval.lower() == "approve",
            "notes": approval
        })
        
        return {"reviews": reviews, "current_reviewer": "finance"}
    
    def manager_review(state: MultiStepState) -> dict:
        """Manager final review"""
        log("Manager review in progress...", "👔")
        
        approval = interrupt({
            "reviewer": "manager",
            "content": state["content"][:100],
            "previous_reviews": state.get("reviews", []),
            "question": "Manager: Final approval?"
        })
        
        reviews = state.get("reviews", [])
        reviews.append({
            "reviewer": "manager",
            "approved": approval.lower() == "approve",
            "notes": approval
        })
        
        return {"reviews": reviews, "current_reviewer": "manager"}
    
    def finalize(state: MultiStepState) -> dict:
        """Finalize based on all reviews"""
        reviews = state.get("reviews", [])
        all_approved = all(r.get("approved", False) for r in reviews)
        
        if all_approved:
            return {"status": "approved"}
        return {"status": "rejected"}
    
    workflow = StateGraph(MultiStepState)
    
    workflow.add_node("submit", submit_for_review)
    workflow.add_node("legal", legal_review)
    workflow.add_node("finance", finance_review)
    workflow.add_node("manager", manager_review)
    workflow.add_node("finalize", finalize)
    
    workflow.add_edge(START, "submit")
    workflow.add_edge("submit", "legal")
    workflow.add_edge("legal", "finance")
    workflow.add_edge("finance", "manager")
    workflow.add_edge("manager", "finalize")
    workflow.add_edge("finalize", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def demo_multi_approval():
    """Demo: Multiple sequential approvals"""
    print("\n" + "="*70)
    print(" DEMO 4: MULTI-STEP APPROVAL")
    print("="*70)
    
    graph = build_multi_approval_workflow()
    config = {"configurable": {"thread_id": "multi-approval-1"}}
    
    print("\n Submitting content for multi-step review...")
    
    # Start workflow
    result = graph.invoke({
        "content": "New company-wide expense policy changes requiring $50k budget increase"
    }, config)
    
    # Process each approval step
    reviewers = [
        ("legal", "approve - no legal concerns"),
        ("finance", "approve - budget available"),
        ("manager", "approve - aligned with goals")
    ]
    
    for reviewer, response in reviewers:
        state = graph.get_state(config)
        
        if state.next:
            print(f"\n Waiting for {reviewer} review...")
            print(f"    {reviewer.title()} responds: '{response}'")
            
            # Resume with approval
            result = graph.invoke(Command(resume=response), config)
    
    # Check final status
    final_state = graph.get_state(config)
    print(f"\n Final Status: {result.get('status', 'unknown')}")
    print(f"   Reviews: {len(result.get('reviews', []))}")
    
    for review in result.get("reviews", []):
        status = "" if review.get("approved") else ""
        print(f"   {status} {review['reviewer']}: {review.get('notes', '')[:30]}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        LangGraph Dynamic Breakpoints & Interrupts             ║
╠══════════════════════════════════════════════════════════════════╣
║  Interrupt Patterns:                                             ║
║  • interrupt_before - Pause before node execution                ║
║  • interrupt_after - Pause after node execution                  ║
║  • interrupt() - Dynamic pause from within a node                ║
║  • Multi-step approval workflows                                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_interrupt_before()
        demo_interrupt_after()
        demo_dynamic_interrupt()
        demo_multi_approval()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:

1. interrupt_before/interrupt_after in compile():
   graph = workflow.compile(interrupt_before=["node_name"])

2. Dynamic interrupt() from within a node:
   value = interrupt({"question": "Approve?"})
   # Execution pauses, value returned when resumed

3. Resume execution:
   graph.invoke(Command(resume="user_response"), config)

4. Update state before resuming:
   graph.update_state(config, {"approved": True})
   graph.invoke(None, config)

5. Check interrupt state:
   state = graph.get_state(config)
   if state.next:  # Still has nodes to execute
       # Handle interrupt
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
