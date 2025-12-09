"""
Script 8: HR Coach with Parallel Agent Execution
Demonstrates: Fan-out/fan-in, parallel specialist analysis, result aggregation
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import operator
from concurrent.futures import ThreadPoolExecutor


class SpecialistAnalysis(TypedDict):
    specialist: str
    findings: str
    confidence: float
    recommendations: list[str]


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_context: dict
    # Parallel analysis results - use operator.add to merge lists
    specialist_analyses: Annotated[list[SpecialistAnalysis], operator.add]
    synthesis: str
    action_plan: list[dict]


# ============================================================
# SPECIALIST ANALYZERS (Run in Parallel)
# ============================================================

SPECIALIST_PROMPTS = {
    "skills_analyst": """You are a Skills Gap Analyst. Analyze the user's situation for:
- Technical skill gaps relative to their goals
- Soft skill development needs
- Learning priorities and recommendations

Be specific and actionable. Rate your confidence (0-1) in your analysis.""",

    "career_strategist": """You are a Career Strategy Specialist. Analyze for:
- Career trajectory alignment with goals
- Market positioning and timing
- Strategic moves and pivots needed

Consider industry trends and organizational context. Rate confidence (0-1).""",

    "relationship_advisor": """You are a Workplace Relationship Advisor. Analyze for:
- Key stakeholder relationships to develop
- Sponsorship and mentorship opportunities
- Political dynamics affecting career growth

Focus on the human element of career success. Rate confidence (0-1).""",

    "risk_assessor": """You are a Career Risk Assessor. Analyze for:
- Potential obstacles and blockers
- Blind spots the user may have
- Contingency planning needs

Be direct about risks without being discouraging. Rate confidence (0-1)."""
}


def create_specialist_analyzer(specialist_type: str):
    """Create a specialist analysis function."""
    
    def analyze(state: CoachingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.6)
        
        # Build context
        context = f"""
User Context:
{json.dumps(state.get('user_context', {}), indent=2)}

Recent conversation:
{state['messages'][-1].content if state['messages'] else 'No message'}
"""
        
        analysis_prompt = f"""{SPECIALIST_PROMPTS[specialist_type]}

{context}

Respond in JSON format:
{{
    "findings": "Your key findings",
    "confidence": 0.0-1.0,
    "recommendations": ["rec1", "rec2", "rec3"]
}}"""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "findings": response.content,
                "confidence": 0.5,
                "recommendations": []
            }
        
        analysis: SpecialistAnalysis = {
            "specialist": specialist_type,
            "findings": result.get("findings", ""),
            "confidence": result.get("confidence", 0.5),
            "recommendations": result.get("recommendations", [])
        }
        
        return {"specialist_analyses": [analysis]}
    
    return analyze


def create_parallel_analysis_node():
    """Create a node that runs all specialists in parallel."""
    
    specialists = list(SPECIALIST_PROMPTS.keys())
    analyzers = {s: create_specialist_analyzer(s) for s in specialists}
    
    def parallel_analyze(state: CoachingState) -> dict:
        """Run all specialist analyses in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(specialists)) as executor:
            futures = {
                executor.submit(analyzer, state): name
                for name, analyzer in analyzers.items()
            }
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.extend(result.get("specialist_analyses", []))
                except Exception as e:
                    # Handle timeout or error gracefully
                    results.append({
                        "specialist": futures[future],
                        "findings": f"Analysis failed: {str(e)}",
                        "confidence": 0.0,
                        "recommendations": []
                    })
        
        return {"specialist_analyses": results}
    
    return parallel_analyze


# ============================================================
# SYNTHESIS AND ACTION PLANNING
# ============================================================

def synthesize_analyses(state: CoachingState) -> dict:
    """Synthesize findings from all specialists into coherent insights."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    analyses = state.get("specialist_analyses", [])
    
    # Sort by confidence
    analyses_sorted = sorted(analyses, key=lambda x: x.get("confidence", 0), reverse=True)
    
    synthesis_prompt = f"""You are synthesizing insights from multiple specialist analyses.

Specialist Findings:
{json.dumps(analyses_sorted, indent=2)}

Create a unified synthesis that:
1. Identifies convergent themes across specialists
2. Highlights high-confidence insights
3. Notes areas of disagreement or uncertainty
4. Prioritizes the most actionable findings

Be concise but comprehensive."""
    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    
    return {"synthesis": response.content}


def create_action_plan(state: CoachingState) -> dict:
    """Create a prioritized action plan from synthesized insights."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    
    analyses = state.get("specialist_analyses", [])
    synthesis = state.get("synthesis", "")
    
    # Gather all recommendations
    all_recommendations = []
    for analysis in analyses:
        for rec in analysis.get("recommendations", []):
            all_recommendations.append({
                "recommendation": rec,
                "source": analysis["specialist"],
                "confidence": analysis["confidence"]
            })
    
    plan_prompt = f"""Based on the synthesis and recommendations, create a prioritized action plan.

Synthesis:
{synthesis}

All Recommendations:
{json.dumps(all_recommendations, indent=2)}

Create an action plan with 5-7 items in JSON format:
{{
    "actions": [
        {{
            "priority": 1,
            "action": "Description",
            "category": "skills|career|relationships|risk_mitigation",
            "timeline": "immediate|short_term|medium_term",
            "effort": "low|medium|high",
            "impact": "low|medium|high"
        }}
    ]
}}

Prioritize by impact and confidence. Balance quick wins with strategic moves."""
    
    response = llm.invoke([HumanMessage(content=plan_prompt)])
    
    try:
        plan = json.loads(response.content)
        actions = plan.get("actions", [])
    except json.JSONDecodeError:
        actions = [{"priority": 1, "action": "Review specialist analyses", "category": "general"}]
    
    return {"action_plan": actions}


def generate_coaching_response(state: CoachingState) -> dict:
    """Generate the final coaching response."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    response_prompt = f"""You are delivering coaching insights based on multi-specialist analysis.

Synthesis of Findings:
{state.get('synthesis', '')}

Prioritized Action Plan:
{json.dumps(state.get('action_plan', []), indent=2)}

Deliver this as a warm, encouraging coaching conversation:
1. Acknowledge the user's situation
2. Share the key insights (without being overwhelming)
3. Present 2-3 highest priority actions
4. Offer to dive deeper into any area

Be conversational, not clinical. This is coaching, not a report."""
    
    response = llm.invoke([HumanMessage(content=response_prompt)])
    
    return {"messages": [AIMessage(content=response.content)]}


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_parallel_coach_graph():
    """Build the parallel analysis coaching graph."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    def initial_engagement(state: CoachingState) -> dict:
        """Initial response while analysis runs."""
        return {
            "messages": [AIMessage(content="Let me analyze your situation from multiple perspectives...")]
        }
    
    def should_analyze(state: CoachingState) -> Literal["analyze", "respond"]:
        """Decide if we need full analysis or can respond directly."""
        last_message = state["messages"][-1].content.lower()
        
        # Trigger analysis for substantive career questions
        analysis_triggers = [
            "career", "promotion", "growth", "development", "stuck",
            "next step", "advice", "help me", "what should", "feedback"
        ]
        
        if any(trigger in last_message for trigger in analysis_triggers):
            return "analyze"
        return "respond"
    
    def quick_response(state: CoachingState) -> dict:
        """Quick response for simple queries."""
        system = "You are a helpful HR coach. Respond naturally to this message."
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("router", lambda s: s)  # Pass-through for routing
    graph.add_node("engage", initial_engagement)
    graph.add_node("parallel_analysis", create_parallel_analysis_node())
    graph.add_node("synthesize", synthesize_analyses)
    graph.add_node("plan", create_action_plan)
    graph.add_node("respond", generate_coaching_response)
    graph.add_node("quick_respond", quick_response)
    
    # Define flow
    graph.add_edge(START, "router")
    
    graph.add_conditional_edges(
        "router",
        should_analyze,
        {
            "analyze": "engage",
            "respond": "quick_respond"
        }
    )
    
    # Analysis pipeline
    graph.add_edge("engage", "parallel_analysis")
    graph.add_edge("parallel_analysis", "synthesize")
    graph.add_edge("synthesize", "plan")
    graph.add_edge("plan", "respond")
    
    graph.add_edge("respond", END)
    graph.add_edge("quick_respond", END)
    
    return graph.compile()


def main():
    """Demo the parallel analysis coaching system."""
    
    coach = build_parallel_coach_graph()
    
    state = {
        "messages": [],
        "user_id": "user_303",
        "user_context": {
            "name": "Morgan Lee",
            "role": "Product Manager",
            "level": "Senior",
            "tenure_years": 3,
            "goals": ["Director role", "Build PM team"],
            "recent_feedback": "Strong execution, needs more strategic visibility"
        },
        "specialist_analyses": [],
        "synthesis": "",
        "action_plan": []
    }
    
    print("=== Parallel Analysis HR Coach ===")
    print(f"Coaching: {state['user_context']['name']} ({state['user_context']['role']})")
    print("\n4 specialists analyze in parallel:")
    print("  • Skills Analyst • Career Strategist")
    print("  • Relationship Advisor • Risk Assessor")
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        print("\n[Running parallel analysis...]")
        result = coach.invoke(state)
        state = result
        
        # Show analysis summary if performed
        if state.get("specialist_analyses"):
            print(f"[Analyzed by {len(state['specialist_analyses'])} specialists]")
            if state.get("action_plan"):
                print(f"[Generated {len(state['action_plan'])} action items]")
        
        # Display response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and "analyzing" not in msg.content.lower():
                print(f"\nCoach: {msg.content}\n")
                break
        
        # Reset for next turn (keep context, clear analyses)
        state["specialist_analyses"] = []
        state["synthesis"] = ""
        state["action_plan"] = []


if __name__ == "__main__":
    main()
