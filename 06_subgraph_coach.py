"""
Script 6: HR Coach with Subgraphs
Demonstrates: Nested graphs, modular agent composition, domain-specific workflows
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import json


# Shared state schema
class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    domain: str  # career, performance, wellbeing
    domain_context: dict
    recommendations: list[dict]


# ============================================================
# CAREER DEVELOPMENT SUBGRAPH
# ============================================================

@tool
def get_career_paths(current_role: str) -> str:
    """Get possible career paths from current role."""
    paths = {
        "Software Engineer": [
            {"path": "Technical", "next_roles": ["Senior Engineer", "Staff Engineer", "Principal"]},
            {"path": "Management", "next_roles": ["Tech Lead", "Engineering Manager", "Director"]},
            {"path": "Specialist", "next_roles": ["Security Engineer", "ML Engineer", "Platform Engineer"]}
        ],
        "default": [{"path": "General", "next_roles": ["Senior", "Lead", "Manager"]}]
    }
    return json.dumps(paths.get(current_role, paths["default"]), indent=2)


@tool
def assess_promotion_readiness(user_id: str, target_role: str) -> str:
    """Assess readiness for promotion to target role."""
    return json.dumps({
        "target_role": target_role,
        "readiness_score": 72,
        "strengths": ["Technical skills", "Collaboration"],
        "gaps": ["Stakeholder management", "Strategic thinking"],
        "recommended_timeline": "6-12 months"
    }, indent=2)


@tool
def get_skill_requirements(role: str) -> str:
    """Get skill requirements for a specific role."""
    return json.dumps({
        "role": role,
        "required_skills": ["Leadership", "Technical depth", "Communication"],
        "nice_to_have": ["Cross-functional experience", "Mentoring track record"]
    }, indent=2)


CAREER_TOOLS = [get_career_paths, assess_promotion_readiness, get_skill_requirements]


def build_career_subgraph():
    """Build the career development coaching subgraph."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(CAREER_TOOLS)
    
    def career_coach(state: CoachingState) -> dict:
        system = """You are a Career Development Coach specializing in:
- Career path exploration and planning
- Promotion readiness assessment
- Skill gap analysis and development planning

Use tools to provide data-driven career guidance.
Build a concrete development plan with timelines."""
        
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_use_tools(state: CoachingState) -> Literal["tools", "synthesize"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "synthesize"
    
    def synthesize_career_plan(state: CoachingState) -> dict:
        """Synthesize findings into a career development plan."""
        llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        
        synthesis_prompt = """Based on the conversation, create a structured career development plan.
Include: target role, timeline, key milestones, and immediate next steps.
Format as a clear, actionable plan."""
        
        messages = state["messages"] + [HumanMessage(content=synthesis_prompt)]
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "recommendations": state.get("recommendations", []) + [{
                "type": "career_plan",
                "generated": True
            }]
        }
    
    graph = StateGraph(CoachingState)
    graph.add_node("career_coach", career_coach)
    graph.add_node("tools", ToolNode(CAREER_TOOLS))
    graph.add_node("synthesize", synthesize_career_plan)
    
    graph.add_edge(START, "career_coach")
    graph.add_conditional_edges("career_coach", should_use_tools)
    graph.add_edge("tools", "career_coach")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


# ============================================================
# PERFORMANCE COACHING SUBGRAPH
# ============================================================

@tool
def get_performance_data(user_id: str) -> str:
    """Retrieve performance review data."""
    return json.dumps({
        "overall_rating": "Meets Expectations",
        "strengths": ["Technical delivery", "Team collaboration"],
        "development_areas": ["Proactive communication", "Scope management"],
        "goals_completion": "85%",
        "peer_feedback_sentiment": "positive"
    }, indent=2)


@tool
def get_360_feedback_summary(user_id: str) -> str:
    """Get summarized 360 feedback."""
    return json.dumps({
        "manager": {"sentiment": "positive", "themes": ["reliable", "could be more visible"]},
        "peers": {"sentiment": "very positive", "themes": ["helpful", "great collaborator"]},
        "direct_reports": {"sentiment": "positive", "themes": ["supportive", "could delegate more"]}
    }, indent=2)


@tool
def create_development_goal(user_id: str, goal: str, category: str, timeline: str) -> str:
    """Create a new development goal."""
    return f"Created goal: '{goal}' in category '{category}' with timeline '{timeline}'"


PERFORMANCE_TOOLS = [get_performance_data, get_360_feedback_summary, create_development_goal]


def build_performance_subgraph():
    """Build the performance coaching subgraph."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(PERFORMANCE_TOOLS)
    
    def analyze_performance(state: CoachingState) -> dict:
        """Analyze performance data and feedback."""
        system = """You are a Performance Coach. Your role:
1. Review performance data and 360 feedback
2. Identify patterns and blind spots
3. Help create actionable development goals
4. Provide balanced perspective on strengths and growth areas

Always ground your coaching in the actual data."""
        
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_use_tools(state: CoachingState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(CoachingState)
    graph.add_node("analyze", analyze_performance)
    graph.add_node("tools", ToolNode(PERFORMANCE_TOOLS))
    
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", should_use_tools)
    graph.add_edge("tools", "analyze")
    
    return graph.compile()


# ============================================================
# WELLBEING COACHING SUBGRAPH
# ============================================================

@tool
def get_workload_metrics(user_id: str) -> str:
    """Get workload and work pattern metrics."""
    return json.dumps({
        "avg_meeting_hours_per_week": 22,
        "after_hours_work_frequency": "occasional",
        "focus_time_blocks_per_week": 3,
        "vacation_days_taken_ytd": 8,
        "vacation_days_available": 12
    }, indent=2)


@tool
def get_wellbeing_resources(topic: str) -> str:
    """Get relevant wellbeing resources."""
    resources = {
        "stress": ["Mindfulness workshop", "EAP counseling", "Stress management toolkit"],
        "burnout": ["Manager conversation guide", "Workload assessment", "Recovery planning"],
        "balance": ["Flexible work guidelines", "Boundary setting workshop", "Time audit tool"]
    }
    return json.dumps(resources.get(topic, resources["balance"]), indent=2)


WELLBEING_TOOLS = [get_workload_metrics, get_wellbeing_resources]


def build_wellbeing_subgraph():
    """Build the wellbeing coaching subgraph with safety checks."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(WELLBEING_TOOLS)
    
    def check_severity(state: CoachingState) -> dict:
        """Check if wellbeing concern requires escalation."""
        last_message = state["messages"][-1].content.lower()
        
        crisis_indicators = ["crisis", "can't cope", "breaking point", "hopeless"]
        needs_escalation = any(ind in last_message for ind in crisis_indicators)
        
        return {
            "domain_context": {
                **state.get("domain_context", {}),
                "needs_escalation": needs_escalation
            }
        }
    
    def route_by_severity(state: CoachingState) -> Literal["escalate", "coach"]:
        if state.get("domain_context", {}).get("needs_escalation"):
            return "escalate"
        return "coach"
    
    def escalate_to_eap(state: CoachingState) -> dict:
        """Provide crisis response and EAP referral."""
        response = """I can hear that you're going through a really difficult time.
Your wellbeing matters, and I want to make sure you get the right support.

**Immediate Resources:**
- Employee Assistance Program: 1-800-XXX-XXXX (24/7)
- Mental Health First Aiders: [Internal Directory Link]
- Manager notification (optional): Would you like me to help facilitate a conversation?

I'm here to support you, but a trained counselor can provide the specialized help you deserve."""
        
        return {"messages": [AIMessage(content=response)]}
    
    def wellbeing_coach(state: CoachingState) -> dict:
        """Provide wellbeing coaching."""
        system = """You are a Wellbeing Coach focusing on sustainable work practices.

Your approach:
- Normalize the experience while taking it seriously
- Explore root causes with curiosity
- Suggest concrete, small changes
- Connect to appropriate resources
- NEVER provide clinical mental health advice

If someone seems in crisis, validate and refer to professional support."""
        
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_use_tools(state: CoachingState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(CoachingState)
    graph.add_node("check_severity", check_severity)
    graph.add_node("escalate", escalate_to_eap)
    graph.add_node("coach", wellbeing_coach)
    graph.add_node("tools", ToolNode(WELLBEING_TOOLS))
    
    graph.add_edge(START, "check_severity")
    graph.add_conditional_edges("check_severity", route_by_severity)
    graph.add_edge("escalate", END)
    graph.add_conditional_edges("coach", should_use_tools)
    graph.add_edge("tools", "coach")
    
    return graph.compile()


# ============================================================
# MAIN ORCHESTRATOR WITH SUBGRAPHS
# ============================================================

def build_orchestrator_graph():
    """Build the main orchestrator that routes to domain subgraphs."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Compile subgraphs
    career_graph = build_career_subgraph()
    performance_graph = build_performance_subgraph()
    wellbeing_graph = build_wellbeing_subgraph()
    
    def classify_domain(state: CoachingState) -> dict:
        """Classify the coaching domain needed."""
        last_message = state["messages"][-1].content
        
        prompt = f"""Classify this coaching request into one domain:
- career: Career planning, promotions, skill development, job transitions
- performance: Performance reviews, feedback, goals, work output
- wellbeing: Stress, burnout, work-life balance, workload

Message: "{last_message}"

Respond with only the domain name."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        domain = response.content.strip().lower()
        
        if domain not in ["career", "performance", "wellbeing"]:
            domain = "career"  # Default
        
        return {"domain": domain}
    
    def route_to_domain(state: CoachingState) -> str:
        return state["domain"]
    
    def run_career_subgraph(state: CoachingState) -> dict:
        result = career_graph.invoke(state)
        return {
            "messages": result["messages"][len(state["messages"]):],
            "recommendations": result.get("recommendations", [])
        }
    
    def run_performance_subgraph(state: CoachingState) -> dict:
        result = performance_graph.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}
    
    def run_wellbeing_subgraph(state: CoachingState) -> dict:
        result = wellbeing_graph.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}
    
    graph = StateGraph(CoachingState)
    
    graph.add_node("classify", classify_domain)
    graph.add_node("career", run_career_subgraph)
    graph.add_node("performance", run_performance_subgraph)
    graph.add_node("wellbeing", run_wellbeing_subgraph)
    
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_domain,
        {"career": "career", "performance": "performance", "wellbeing": "wellbeing"}
    )
    
    for domain in ["career", "performance", "wellbeing"]:
        graph.add_edge(domain, END)
    
    return graph.compile()


def main():
    """Demo the subgraph-based coaching system."""
    
    orchestrator = build_orchestrator_graph()
    
    state = {
        "messages": [],
        "user_id": "user_101",
        "domain": "",
        "domain_context": {},
        "recommendations": []
    }
    
    print("=== HR Coach with Domain Subgraphs ===")
    print("Domains: career, performance, wellbeing")
    print("Each domain has specialized tools and workflows")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        result = orchestrator.invoke(state)
        state = result
        
        print(f"[Domain: {state['domain']}]")
        
        # Get last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nCoach: {msg.content}\n")
                break


if __name__ == "__main__":
    main()
