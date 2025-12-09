"""
18. Full Coach Orchestrator
===========================
Production-style orchestrator that routes to specialist agents
based on intent, risk, and context. Combines patterns from
previous examples into a cohesive system.

Key concepts:
- Supervisor-based routing
- Specialist agent delegation
- Shared context and memory
- Risk-aware processing
- Quality assurance checks
"""

from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# SPECIALIST TYPES
# ============================================================

class Specialist(str, Enum):
    GOAL_COACH = "goal_coach"
    FEEDBACK_COACH = "feedback_coach"
    CAREER_COACH = "career_coach"
    WELLBEING_COACH = "wellbeing_coach"
    CONFLICT_COACH = "conflict_coach"
    GENERAL_COACH = "general_coach"
    SAFETY_HANDLER = "safety_handler"


# ============================================================
# ORCHESTRATOR STATE
# ============================================================

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    
    # Routing
    intent: str
    selected_specialist: str
    risk_level: str
    
    # Context
    user_profile: dict
    session_context: dict
    retrieved_memories: list[dict]
    
    # Quality
    response_draft: str
    quality_score: float
    requires_revision: bool
    
    # Audit
    routing_reasoning: str
    specialist_output: str


# ============================================================
# SAMPLE USER PROFILE
# ============================================================

SAMPLE_PROFILE = {
    "name": "Alex",
    "role": "Senior Software Engineer",
    "tenure": "3 years",
    "manager": "Jordan",
    "team": "Platform",
    "current_goals": [
        "Prepare for promotion to Staff Engineer",
        "Improve cross-team collaboration"
    ],
    "recent_topics": ["career growth", "stakeholder management"],
    "preferences": {
        "style": "direct but supportive",
        "motivators": ["impact", "growth", "recognition"]
    }
}


# ============================================================
# ORCHESTRATOR NODES
# ============================================================

def analyze_intent(state: OrchestratorState) -> dict:
    """Analyze user intent and determine routing."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    last_message = state["messages"][-1].content
    profile = state.get("user_profile", SAMPLE_PROFILE)
    
    prompt = f"""Analyze this message to route to the right specialist coach.

User Profile:
- Role: {profile.get('role')}
- Current Goals: {profile.get('current_goals')}
- Recent Topics: {profile.get('recent_topics')}

Message: {last_message}

Determine:
1. Primary intent (what they want help with)
2. Risk level (minimal/moderate/high/critical)
3. Best specialist to handle this

Specialists available:
- goal_coach: Goal setting, progress tracking, OKRs
- feedback_coach: Understanding feedback, 360 reviews, self-assessment
- career_coach: Promotions, career paths, skill development
- wellbeing_coach: Stress, burnout, work-life balance
- conflict_coach: Interpersonal issues, difficult conversations
- general_coach: General questions, casual check-ins
- safety_handler: Mental health crisis, harassment, safety concerns

Return JSON:
{{
    "intent": "brief description of what they want",
    "risk_level": "minimal|moderate|high|critical",
    "selected_specialist": "specialist_name",
    "reasoning": "why this routing makes sense",
    "context_needed": ["any specific context to retrieve"]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "intent": parsed.get("intent", "general coaching"),
            "risk_level": parsed.get("risk_level", "minimal"),
            "selected_specialist": parsed.get("selected_specialist", "general_coach"),
            "routing_reasoning": parsed.get("reasoning", ""),
            "user_profile": profile
        }
    except:
        return {
            "intent": "general coaching",
            "risk_level": "minimal",
            "selected_specialist": "general_coach",
            "user_profile": profile
        }


def route_to_specialist(state: OrchestratorState) -> str:
    """Route to the appropriate specialist."""
    # Safety first
    if state.get("risk_level") == "critical":
        return "safety_handler"
    
    specialist = state.get("selected_specialist", "general_coach")
    
    # Map to node names
    routes = {
        "goal_coach": "goal_specialist",
        "feedback_coach": "feedback_specialist",
        "career_coach": "career_specialist",
        "wellbeing_coach": "wellbeing_specialist",
        "conflict_coach": "conflict_specialist",
        "general_coach": "general_specialist",
        "safety_handler": "safety_handler"
    }
    
    return routes.get(specialist, "general_specialist")


# ============================================================
# SPECIALIST AGENTS
# ============================================================

def goal_specialist(state: OrchestratorState) -> dict:
    """Goal-focused coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    profile = state.get("user_profile", {})
    
    prompt = f"""You are a goal-setting coach specialist.

User: {profile.get('name', 'the user')}
Role: {profile.get('role')}
Current Goals: {profile.get('current_goals')}

Their message: {state["messages"][-1].content}

As a goal specialist, help them:
- Clarify and refine goals
- Break down goals into key results
- Track progress and celebrate wins
- Identify blockers and solutions
- Maintain accountability

Be encouraging but push for specificity. Use frameworks like SMART or OKRs where helpful."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "goal_specialist"
    }


def feedback_specialist(state: OrchestratorState) -> dict:
    """Feedback interpretation coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""You are a feedback interpretation specialist.

User message: {state["messages"][-1].content}

Help them:
- Make sense of feedback they've received
- Identify patterns and themes
- Separate actionable from non-actionable
- Process emotional reactions to feedback
- Create development plans from feedback

Be supportive but help them see feedback as growth opportunity."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "feedback_specialist"
    }


def career_specialist(state: OrchestratorState) -> dict:
    """Career development coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    profile = state.get("user_profile", {})
    
    prompt = f"""You are a career development specialist.

User: {profile.get('name', 'the user')}
Current Role: {profile.get('role')}
Goals: {profile.get('current_goals')}

Their message: {state["messages"][-1].content}

Help them with:
- Career path planning
- Promotion readiness
- Skill gap analysis
- Visibility and sponsorship
- Career conversations with manager

Be strategic and practical. Help them think long-term while taking concrete steps."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "career_specialist"
    }


def wellbeing_specialist(state: OrchestratorState) -> dict:
    """Wellbeing and resilience coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    risk_level = state.get("risk_level", "minimal")
    
    prompt = f"""You are a wellbeing specialist coach.

Risk level: {risk_level}
User message: {state["messages"][-1].content}

Help them with:
- Stress management
- Work-life boundaries
- Burnout prevention
- Resilience building
- Energy management

Be warm and compassionate. Validate their feelings.
If risk is moderate+, gently mention EAP resources are available.
Never provide clinical advice - you're a coach, not a therapist."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "wellbeing_specialist"
    }


def conflict_specialist(state: OrchestratorState) -> dict:
    """Conflict resolution coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""You are a conflict resolution specialist.

User message: {state["messages"][-1].content}

Help them:
- Understand different perspectives
- Prepare for difficult conversations
- De-escalate tensions
- Find win-win solutions
- Set appropriate boundaries

Stay neutral - don't take sides. Help them see the other perspective.
If it sounds like harassment or discrimination, note that HR may need to be involved."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "conflict_specialist"
    }


def general_specialist(state: OrchestratorState) -> dict:
    """General coaching for miscellaneous topics."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    profile = state.get("user_profile", {})
    
    prompt = f"""You are a supportive HR coach.

User: {profile.get('name', 'the user')}
Their style preference: {profile.get('preferences', {}).get('style', 'supportive')}

Their message: {state["messages"][-1].content}

Provide helpful, warm coaching support.
Ask thoughtful questions to help them reflect.
Connect to their goals when relevant: {profile.get('current_goals')}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response_draft": response.content,
        "specialist_output": "general_specialist"
    }


def safety_handler(state: OrchestratorState) -> dict:
    """Handle high-risk situations requiring escalation."""
    
    response = """I want to make sure you get the right support.

What you've shared sounds really important, and I want to connect you with someone who can help properly.

Our Employee Assistance Program (EAP) offers confidential support 24/7. They can help with what you're going through.

Would it be okay if I flagged this for follow-up from our HR team? They're trained to support situations like this, and everything is kept confidential.

In the meantime, please know that you don't have to handle this alone."""

    return {
        "response_draft": response,
        "specialist_output": "safety_handler",
        "requires_revision": False  # Don't revise safety responses
    }


# ============================================================
# QUALITY ASSURANCE
# ============================================================

def quality_check(state: OrchestratorState) -> dict:
    """Check response quality and appropriateness."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Skip QA for safety responses
    if state.get("specialist_output") == "safety_handler":
        return {
            "quality_score": 1.0,
            "requires_revision": False
        }
    
    draft = state.get("response_draft", "")
    user_message = state["messages"][-1].content
    
    prompt = f"""Rate this coaching response for quality.

User message: {user_message}
Coach response: {draft}

Score 0-1 on:
1. Empathy: Does it acknowledge feelings?
2. Helpfulness: Does it address their needs?
3. Appropriateness: Is it suitable for HR coaching?
4. Actionability: Does it help them move forward?

Return JSON:
{{
    "scores": {{
        "empathy": 0.8,
        "helpfulness": 0.8,
        "appropriateness": 0.9,
        "actionability": 0.7
    }},
    "overall": 0.8,
    "issues": ["any specific issues"],
    "requires_revision": false
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "quality_score": parsed.get("overall", 0.8),
            "requires_revision": parsed.get("requires_revision", False) or parsed.get("overall", 1) < 0.6
        }
    except:
        return {"quality_score": 0.8, "requires_revision": False}


def revise_response(state: OrchestratorState) -> dict:
    """Revise response if quality is low."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    draft = state.get("response_draft", "")
    
    prompt = f"""Improve this coaching response. Make it more empathetic and actionable.

Original: {draft}

Revised response (just the response, no explanation):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "response_draft": response.content,
        "requires_revision": False
    }


def finalize_response(state: OrchestratorState) -> dict:
    """Finalize and return the response."""
    return {
        "messages": [AIMessage(content=state.get("response_draft", ""))]
    }


def should_revise(state: OrchestratorState) -> str:
    """Decide if revision is needed."""
    if state.get("requires_revision", False):
        return "revise"
    return "finalize"


# ============================================================
# BUILD ORCHESTRATOR GRAPH
# ============================================================

def build_orchestrator() -> StateGraph:
    """Build the full coach orchestrator."""
    
    graph = StateGraph(OrchestratorState)
    
    # Add nodes
    graph.add_node("analyze", analyze_intent)
    graph.add_node("goal_specialist", goal_specialist)
    graph.add_node("feedback_specialist", feedback_specialist)
    graph.add_node("career_specialist", career_specialist)
    graph.add_node("wellbeing_specialist", wellbeing_specialist)
    graph.add_node("conflict_specialist", conflict_specialist)
    graph.add_node("general_specialist", general_specialist)
    graph.add_node("safety_handler", safety_handler)
    graph.add_node("quality_check", quality_check)
    graph.add_node("revise", revise_response)
    graph.add_node("finalize", finalize_response)
    
    # Entry point
    graph.add_edge(START, "analyze")
    
    # Route to specialists
    graph.add_conditional_edges(
        "analyze",
        route_to_specialist,
        {
            "goal_specialist": "goal_specialist",
            "feedback_specialist": "feedback_specialist",
            "career_specialist": "career_specialist",
            "wellbeing_specialist": "wellbeing_specialist",
            "conflict_specialist": "conflict_specialist",
            "general_specialist": "general_specialist",
            "safety_handler": "safety_handler"
        }
    )
    
    # All specialists go to QA
    for specialist in ["goal_specialist", "feedback_specialist", "career_specialist",
                       "wellbeing_specialist", "conflict_specialist", "general_specialist",
                       "safety_handler"]:
        graph.add_edge(specialist, "quality_check")
    
    # QA routing
    graph.add_conditional_edges(
        "quality_check",
        should_revise,
        {"revise": "revise", "finalize": "finalize"}
    )
    
    graph.add_edge("revise", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    orchestrator = build_orchestrator()
    
    print("=" * 60)
    print("FULL COACH ORCHESTRATOR")
    print("=" * 60)
    print("\nThis orchestrator routes to specialist coaches based on your needs.")
    print(f"\nUser Profile: {SAMPLE_PROFILE['name']}, {SAMPLE_PROFILE['role']}")
    print(f"Current Goals: {', '.join(SAMPLE_PROFILE['current_goals'])}")
    print("\nSpecialists: Goals, Feedback, Career, Wellbeing, Conflict, General")
    print("\nTry different topics to see routing in action.\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nGood luck on your journey! 🚀")
            break
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "intent": "",
            "selected_specialist": "",
            "risk_level": "minimal",
            "user_profile": SAMPLE_PROFILE,
            "session_context": {},
            "retrieved_memories": [],
            "response_draft": "",
            "quality_score": 0.0,
            "requires_revision": False,
            "routing_reasoning": "",
            "specialist_output": ""
        }
        
        result = orchestrator.invoke(state)
        
        # Show routing info
        specialist = result.get("specialist_output", "unknown")
        intent = result.get("intent", "")
        quality = result.get("quality_score", 0)
        
        print(f"\n[Routed to: {specialist} | Intent: {intent[:40]}... | Quality: {quality:.0%}]")
        
        response = result["messages"][-1].content
        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    main()
