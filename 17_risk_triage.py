"""
17. Risk-Aware Triage System
============================
Multi-level risk classification with appropriate escalation pathways.
Demonstrates safety-first architecture for HR coaching.

Key concepts:
- Multi-dimensional risk classification
- Confidence-calibrated decisions
- Human escalation pathways
- Safe default behaviors
- Audit logging for high-risk interactions
"""

from typing import Annotated, TypedDict, Literal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# RISK CATEGORIES AND LEVELS
# ============================================================

class RiskCategory(str, Enum):
    MENTAL_HEALTH = "mental_health"
    HARASSMENT = "harassment"
    DISCRIMINATION = "discrimination"
    LEGAL = "legal"
    SAFETY = "safety"
    POLICY_VIOLATION = "policy_violation"
    CONFLICT = "conflict"
    PERFORMANCE = "performance"
    GENERAL = "general"


class RiskLevel(str, Enum):
    CRITICAL = "critical"    # Immediate human intervention
    HIGH = "high"            # Route to specialist, may need escalation
    MODERATE = "moderate"    # Handle with care, document
    LOW = "low"              # Standard coaching with awareness
    MINIMAL = "minimal"      # Normal coaching interaction


@dataclass
class RiskAssessment:
    category: RiskCategory
    level: RiskLevel
    confidence: float
    signals: list[str]
    recommended_action: str
    
    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "level": self.level.value,
            "confidence": self.confidence,
            "signals": self.signals,
            "recommended_action": self.recommended_action
        }


# ============================================================
# ESCALATION CONTACTS (would be configured per org)
# ============================================================

ESCALATION_CONTACTS = {
    RiskCategory.MENTAL_HEALTH: {
        "team": "Employee Assistance Program (EAP)",
        "contact": "eap@company.com",
        "hotline": "1-800-EAP-HELP"
    },
    RiskCategory.HARASSMENT: {
        "team": "HR Business Partner",
        "contact": "hr-concerns@company.com"
    },
    RiskCategory.DISCRIMINATION: {
        "team": "HR Business Partner / Legal",
        "contact": "hr-concerns@company.com"
    },
    RiskCategory.LEGAL: {
        "team": "Legal / Employment Counsel",
        "contact": "legal@company.com"
    },
    RiskCategory.SAFETY: {
        "team": "Security / HR",
        "contact": "security@company.com",
        "emergency": "911"
    },
}


# ============================================================
# STATE DEFINITION
# ============================================================

class TriageState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_assessments: list[dict]
    primary_risk: dict
    escalation_triggered: bool
    escalation_info: dict
    safe_response: str
    audit_entry: dict


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(state: TriageState) -> dict:
    """Multi-dimensional risk classification."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    last_message = state["messages"][-1].content
    
    prompt = f"""You are a risk classifier for an HR coaching system.
Analyze this message for potential risks that require special handling.

Message: {last_message}

Classify across these risk categories:
- mental_health: Signs of distress, burnout, anxiety, depression, self-harm
- harassment: Reports of bullying, sexual harassment, hostile work environment
- discrimination: Reports of unfair treatment based on protected characteristics
- legal: Potential legal exposure, whistleblowing, contract issues
- safety: Threats, violence, workplace safety concerns
- policy_violation: Potential violations of company policy
- conflict: Interpersonal conflicts that may need HR involvement
- performance: Standard performance/career coaching
- general: General coaching conversation

For each relevant category, provide:
- level: critical/high/moderate/low/minimal
- confidence: 0-1 how confident you are
- signals: specific phrases or indicators

Return JSON:
{{
    "assessments": [
        {{
            "category": "category_name",
            "level": "risk_level",
            "confidence": 0.8,
            "signals": ["signal1", "signal2"],
            "reasoning": "why this assessment"
        }}
    ],
    "primary_concern": "category_name or null",
    "overall_risk_level": "highest applicable level",
    "requires_immediate_escalation": true/false
}}

Be conservative - if unsure, err on the side of higher risk.
Mental health keywords to watch: "overwhelmed", "can't cope", "hopeless", "thinking of quitting", "at my limit", "panic", "anxiety", "depressed"."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        
        assessments = []
        for a in parsed.get("assessments", []):
            assessments.append({
                "category": a.get("category", "general"),
                "level": a.get("level", "minimal"),
                "confidence": a.get("confidence", 0.5),
                "signals": a.get("signals", []),
                "reasoning": a.get("reasoning", "")
            })
        
        # Find primary risk
        primary = None
        if parsed.get("primary_concern"):
            primary = next(
                (a for a in assessments if a["category"] == parsed["primary_concern"]),
                None
            )
        
        return {
            "risk_assessments": assessments,
            "primary_risk": primary or {"category": "general", "level": "minimal"},
            "escalation_triggered": parsed.get("requires_immediate_escalation", False)
        }
    except:
        return {
            "risk_assessments": [{"category": "general", "level": "minimal"}],
            "primary_risk": {"category": "general", "level": "minimal"},
            "escalation_triggered": False
        }


def route_by_risk(state: TriageState) -> str:
    """Route based on risk assessment."""
    primary = state.get("primary_risk", {})
    level = primary.get("level", "minimal")
    
    if level == "critical":
        return "handle_critical"
    elif level == "high":
        return "handle_high_risk"
    elif level == "moderate":
        return "handle_moderate_risk"
    else:
        return "handle_standard"


# ============================================================
# RISK HANDLERS
# ============================================================

def handle_critical(state: TriageState) -> dict:
    """Handle critical risk - immediate escalation."""
    primary = state.get("primary_risk", {})
    category = RiskCategory(primary.get("category", "general"))
    
    # Get escalation contact
    contact = ESCALATION_CONTACTS.get(category, {
        "team": "HR Business Partner",
        "contact": "hr@company.com"
    })
    
    # Prepare compassionate but clear response
    response = f"""I want to make sure you get the right support for what you're going through.

What you've shared sounds really important, and I want to make sure you're connected with someone who can help properly.

I'm going to flag this conversation for our {contact['team']} team to follow up with you. They're the right people to support you with this.

In the meantime:"""
    
    if category == RiskCategory.MENTAL_HEALTH:
        response += f"""
- If you're in crisis, please reach out to the Employee Assistance Program: {contact.get('hotline', 'Contact HR')}
- They offer confidential support 24/7
- You don't have to go through this alone"""
    
    elif category == RiskCategory.SAFETY:
        response += """
- If you're in immediate danger, please call emergency services (911)
- Our security team is being notified
- Your safety is the top priority"""
    
    else:
        response += f"""
- Someone from {contact['team']} will reach out to you soon
- Everything you've shared will be handled with appropriate confidentiality
- Is there anything else you'd like me to document for them?"""
    
    # Create audit entry
    audit = {
        "timestamp": datetime.now().isoformat(),
        "risk_category": category.value,
        "risk_level": "critical",
        "escalation_team": contact.get("team"),
        "user_message_summary": state["messages"][-1].content[:200],
        "action_taken": "Escalated to human support"
    }
    
    return {
        "messages": [AIMessage(content=response)],
        "escalation_triggered": True,
        "escalation_info": contact,
        "audit_entry": audit
    }


def handle_high_risk(state: TriageState) -> dict:
    """Handle high risk - careful response with potential escalation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    primary = state.get("primary_risk", {})
    category = primary.get("category", "general")
    signals = primary.get("signals", [])
    
    prompt = f"""You are an HR coach handling a sensitive situation.

Risk Assessment:
- Category: {category}
- Risk Level: HIGH
- Signals detected: {signals}

User message: {state["messages"][-1].content}

Respond with:
1. Empathy and validation first
2. Acknowledge the seriousness appropriately
3. Clarify what support is available
4. Gently assess if they want to escalate to HR
5. Offer to continue supporting them

DO NOT:
- Minimize their concerns
- Promise outcomes you can't guarantee
- Provide legal or medical advice
- Try to handle harassment/discrimination reports yourself

Keep response compassionate but appropriately bounded."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    audit = {
        "timestamp": datetime.now().isoformat(),
        "risk_category": category,
        "risk_level": "high",
        "escalation_offered": True,
        "action_taken": "Provided careful support, offered escalation"
    }
    
    return {
        "messages": [AIMessage(content=response.content)],
        "audit_entry": audit
    }


def handle_moderate_risk(state: TriageState) -> dict:
    """Handle moderate risk - enhanced care and documentation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    primary = state.get("primary_risk", {})
    category = primary.get("category", "general")
    
    # Category-specific guidance
    guidance = {
        "mental_health": "Be especially warm and supportive. Check in on their wellbeing. Mention EAP if relevant.",
        "conflict": "Stay neutral. Focus on understanding all perspectives. Don't take sides.",
        "performance": "Be encouraging but realistic. Focus on growth and development.",
        "policy_violation": "Don't provide legal advice. Suggest they review policy or talk to HR if needed.",
    }
    
    specific_guidance = guidance.get(category, "Provide supportive coaching.")
    
    prompt = f"""You are an HR coach. This conversation has moderate sensitivity.

Category: {category}
Guidance: {specific_guidance}

User message: {state["messages"][-1].content}

Respond supportively while being mindful of the sensitivity.
Document any concerning patterns but don't alarm the user unnecessarily."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "audit_entry": {
            "timestamp": datetime.now().isoformat(),
            "risk_category": category,
            "risk_level": "moderate",
            "action_taken": "Handled with enhanced care"
        }
    }


def handle_standard(state: TriageState) -> dict:
    """Handle standard coaching - normal flow."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""You are a supportive HR coach.

User message: {state["messages"][-1].content}

Provide helpful, warm coaching support. Ask thoughtful questions.
Focus on their growth and development."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_triage_coach() -> StateGraph:
    """Build the risk-aware triage coaching graph."""
    
    graph = StateGraph(TriageState)
    
    # Add nodes
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("handle_critical", handle_critical)
    graph.add_node("handle_high_risk", handle_high_risk)
    graph.add_node("handle_moderate_risk", handle_moderate_risk)
    graph.add_node("handle_standard", handle_standard)
    
    # Flow
    graph.add_edge(START, "classify_risk")
    graph.add_conditional_edges(
        "classify_risk",
        route_by_risk,
        {
            "handle_critical": "handle_critical",
            "handle_high_risk": "handle_high_risk",
            "handle_moderate_risk": "handle_moderate_risk",
            "handle_standard": "handle_standard"
        }
    )
    
    # All handlers end
    graph.add_edge("handle_critical", END)
    graph.add_edge("handle_high_risk", END)
    graph.add_edge("handle_moderate_risk", END)
    graph.add_edge("handle_standard", END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    coach = build_triage_coach()
    
    print("=" * 60)
    print("RISK-AWARE TRIAGE COACH")
    print("=" * 60)
    print("\nThis coach demonstrates safety-first architecture.")
    print("It classifies risk and routes to appropriate handlers.")
    print("\nTry different scenarios:")
    print("  - Standard: 'How can I improve my presentation skills?'")
    print("  - Moderate: 'I'm having conflict with a coworker'")
    print("  - High: 'My manager is treating me unfairly because of...'")
    print("  - Critical: 'I feel completely overwhelmed and hopeless'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nTake care! 💙")
            break
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "risk_assessments": [],
            "primary_risk": {},
            "escalation_triggered": False,
            "escalation_info": {},
            "safe_response": "",
            "audit_entry": {}
        }
        
        result = coach.invoke(state)
        
        # Show risk assessment (for demo purposes)
        primary = result.get("primary_risk", {})
        if primary:
            level = primary.get("level", "minimal")
            category = primary.get("category", "general")
            print(f"\n[Risk: {level.upper()} - {category}]")
        
        response = result["messages"][-1].content
        print(f"\nCoach: {response}\n")
        
        # Show if escalation was triggered
        if result.get("escalation_triggered"):
            info = result.get("escalation_info", {})
            print(f"  ESCALATION TRIGGERED to {info.get('team', 'HR')}")
            print()


if __name__ == "__main__":
    main()
