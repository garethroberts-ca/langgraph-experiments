"""
Script 4: HR Coach with Safety Guardrails
Demonstrates: Risk classification, safety agents, human escalation, interrupt handling
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    risk_level: str
    risk_flags: list[str]
    escalation_required: bool
    escalation_reason: str


# Risk categories that require special handling
RISK_CATEGORIES = {
    "mental_health": {
        "keywords": ["suicide", "self-harm", "depressed", "anxiety attack", "panic", "can't go on"],
        "level": RiskLevel.CRITICAL,
        "action": "immediate_escalation"
    },
    "harassment": {
        "keywords": ["harassment", "bullying", "discriminat", "hostile", "inappropriate touch"],
        "level": RiskLevel.HIGH,
        "action": "hr_escalation"
    },
    "legal": {
        "keywords": ["lawsuit", "legal action", "lawyer", "whistleblow", "fraud"],
        "level": RiskLevel.HIGH,
        "action": "legal_escalation"
    },
    "performance_crisis": {
        "keywords": ["fired", "pip", "termination", "getting let go"],
        "level": RiskLevel.MEDIUM,
        "action": "coach_with_caution"
    }
}


SAFETY_RESPONSES = {
    RiskLevel.CRITICAL: """I can hear that you're going through something really difficult right now. 
Your wellbeing is the most important thing, and I want to make sure you get the right support.

I'm connecting you with our Employee Assistance Program (EAP) who have trained counselors available 24/7.
**EAP Hotline: 1-800-XXX-XXXX**

Would you like me to have someone reach out to you directly?""",

    RiskLevel.HIGH: """Thank you for trusting me with this. What you're describing sounds serious and 
deserves proper attention from the right people.

I'm flagging this for our HR team who can provide appropriate support and guidance.
A member of our HR team will reach out to you within 24 hours.

In the meantime, is there anything immediate I can help you think through?"""
}


COACH_PROMPT = """You are an HR coaching assistant with strict safety boundaries.

## Safety Guidelines
- NEVER provide medical, legal, or clinical psychological advice
- If you detect distress, validate feelings but don't attempt to counsel
- Stay within your expertise: career development, performance, professional growth
- Be warm but maintain appropriate boundaries

## Current Risk Context
Risk Level: {risk_level}
Flags: {risk_flags}

If risk level is medium or above, be extra thoughtful and supportive while staying within boundaries."""


def create_risk_classifier(llm: ChatOpenAI):
    """Classify risk level of user message."""
    
    def classify_risk(state: CoachingState) -> dict:
        last_message = state["messages"][-1].content.lower()
        
        # Rule-based initial check
        risk_flags = []
        max_risk = RiskLevel.LOW
        
        for category, config in RISK_CATEGORIES.items():
            for keyword in config["keywords"]:
                if keyword in last_message:
                    risk_flags.append(category)
                    if config["level"].value > max_risk.value:
                        max_risk = config["level"]
                    break
        
        # LLM-based nuanced classification for edge cases
        if max_risk == RiskLevel.LOW:
            classification_prompt = f"""Analyze this message for coaching risk level:

Message: "{last_message}"

Consider:
- Signs of emotional distress (not just venting)
- References to serious workplace issues
- Requests for advice outside coaching scope

Rate as: low, medium, high, or critical
Respond with only the rating."""

            response = llm.invoke([HumanMessage(content=classification_prompt)])
            llm_risk = response.content.strip().lower()
            
            if llm_risk in [r.value for r in RiskLevel]:
                if llm_risk != RiskLevel.LOW.value:
                    max_risk = RiskLevel(llm_risk)
                    risk_flags.append("llm_detected")
        
        escalation_required = max_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        escalation_reason = f"Risk flags: {', '.join(risk_flags)}" if escalation_required else ""
        
        return {
            "risk_level": max_risk.value,
            "risk_flags": risk_flags,
            "escalation_required": escalation_required,
            "escalation_reason": escalation_reason
        }
    
    return classify_risk


def create_safety_responder():
    """Handle high-risk situations with appropriate responses."""
    
    def safety_response(state: CoachingState) -> dict:
        risk_level = RiskLevel(state["risk_level"])
        response_text = SAFETY_RESPONSES.get(
            risk_level,
            "I want to make sure you get the right support. Let me connect you with someone who can help."
        )
        
        return {"messages": [AIMessage(content=response_text)]}
    
    return safety_response


def create_safe_coach(llm: ChatOpenAI):
    """Coach that operates within safety boundaries."""
    
    def coach(state: CoachingState) -> dict:
        system_content = COACH_PROMPT.format(
            risk_level=state.get("risk_level", "low"),
            risk_flags=state.get("risk_flags", [])
        )
        
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm.invoke(messages)
        
        return {"messages": [response]}
    
    return coach


def route_by_risk(state: CoachingState) -> Literal["safety_response", "coach"]:
    """Route based on risk classification."""
    if state.get("escalation_required", False):
        return "safety_response"
    return "coach"


def build_safety_coach_graph():
    """Build coach with safety guardrails."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("classify_risk", create_risk_classifier(llm))
    graph.add_node("safety_response", create_safety_responder())
    graph.add_node("coach", create_safe_coach(llm))
    
    # Define flow
    graph.add_edge(START, "classify_risk")
    
    # Conditional routing based on risk
    graph.add_conditional_edges(
        "classify_risk",
        route_by_risk,
        {
            "safety_response": "safety_response",
            "coach": "coach"
        }
    )
    
    graph.add_edge("safety_response", END)
    graph.add_edge("coach", END)
    
    return graph.compile()


def main():
    """Demo the safety-aware coach."""
    
    coach = build_safety_coach_graph()
    
    state = {
        "messages": [],
        "user_id": "user_456",
        "risk_level": "low",
        "risk_flags": [],
        "escalation_required": False,
        "escalation_reason": ""
    }
    
    print("=== Safety-Aware HR Coach ===")
    print("This coach has guardrails for sensitive topics")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        result = coach.invoke(state)
        state = result
        
        # Show risk assessment
        if state["risk_level"] != "low":
            print(f"[Risk: {state['risk_level']} | Flags: {state['risk_flags']}]")
        
        print(f"\nCoach: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
