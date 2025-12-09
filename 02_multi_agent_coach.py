"""
Script 2: Multi-Agent HR Coaching System
Demonstrates: Router pattern, specialist agents, conditional edges
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    current_intent: str  # Detected intent from router
    goals: list[dict]    # User's goals


# Specialist agent prompts
AGENT_PROMPTS = {
    "goal_setting": """You are a Goal-Setting Coach specializing in helping employees define clear, achievable objectives.

Use the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound).
Help break down large goals into actionable steps.
Connect goals to career aspirations and organizational objectives.""",

    "feedback": """You are a Feedback Coach helping employees process and act on feedback.

Help users identify patterns across multiple feedback sources.
Reframe critical feedback constructively.
Suggest concrete actions to address development areas.""",

    "reflection": """You are a Reflection Coach facilitating self-discovery and insight.

Ask powerful open-ended questions.
Help users explore different perspectives.
Guide users to identify their own solutions and insights.""",

    "general": """You are a General HR Coach providing supportive guidance.

Listen actively and validate emotions.
Help clarify thinking and options.
Know when to refer to specialist coaching areas."""
}


def create_router_node(llm: ChatOpenAI):
    """Route user messages to appropriate specialist agent."""
    
    def router(state: CoachingState) -> dict:
        last_message = state["messages"][-1].content
        
        classification_prompt = f"""Classify this coaching request into one category:
- goal_setting: User wants to set, refine, or track goals
- feedback: User wants to discuss or process feedback they received
- reflection: User wants to reflect on experiences or decisions
- general: Other coaching needs

User message: "{last_message}"

Respond with only the category name."""

        response = llm.invoke([HumanMessage(content=classification_prompt)])
        intent = response.content.strip().lower()
        
        # Validate intent
        if intent not in AGENT_PROMPTS:
            intent = "general"
        
        return {"current_intent": intent}
    
    return router


def create_specialist_node(llm: ChatOpenAI, agent_type: str):
    """Create a specialist coaching agent node."""
    
    def specialist(state: CoachingState) -> dict:
        system_msg = SystemMessage(content=AGENT_PROMPTS[agent_type])
        
        # Include goal context if available
        context = ""
        if state.get("goals"):
            context = f"\n\nUser's current goals: {json.dumps(state['goals'], indent=2)}"
        
        messages = [
            SystemMessage(content=AGENT_PROMPTS[agent_type] + context)
        ] + state["messages"]
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    return specialist


def route_to_specialist(state: CoachingState) -> str:
    """Conditional edge function to route to correct specialist."""
    return state["current_intent"]


def build_multi_agent_graph():
    """Build the multi-agent coaching graph."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    graph = StateGraph(CoachingState)
    
    # Add router node
    graph.add_node("router", create_router_node(llm))
    
    # Add specialist nodes
    for agent_type in AGENT_PROMPTS.keys():
        graph.add_node(agent_type, create_specialist_node(llm, agent_type))
    
    # Define edges
    graph.add_edge(START, "router")
    
    # Conditional routing from router to specialists
    graph.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "goal_setting": "goal_setting",
            "feedback": "feedback",
            "reflection": "reflection",
            "general": "general"
        }
    )
    
    # All specialists end the turn
    for agent_type in AGENT_PROMPTS.keys():
        graph.add_edge(agent_type, END)
    
    return graph.compile()


def main():
    """Demo the multi-agent coaching system."""
    
    coach = build_multi_agent_graph()
    
    state = {
        "messages": [],
        "user_name": "Jordan",
        "current_intent": "",
        "goals": [
            {"title": "Improve presentation skills", "status": "in_progress"},
            {"title": "Get promoted to senior role", "status": "planning"}
        ]
    }
    
    print("=== Multi-Agent HR Coach ===")
    print("Specialists: goal_setting, feedback, reflection, general")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        result = coach.invoke(state)
        state = result
        
        print(f"[Routed to: {state['current_intent']}]")
        print(f"\nCoach: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
