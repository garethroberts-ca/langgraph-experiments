"""
Script 5: Full HR Coach with Tools and Integrations
Demonstrates: Tool calling, RAG, action execution, complete agentic loop
"""

from typing import Annotated, TypedDict, Literal
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict
    pending_actions: list[dict]


# Simulated data stores
USER_PROFILES = {
    "user_789": {
        "name": "Sam Chen",
        "role": "Senior Software Engineer",
        "manager": "Lisa Park",
        "tenure_months": 24,
        "goals": [
            {"id": "g1", "title": "Transition to Tech Lead", "progress": 60},
            {"id": "g2", "title": "Improve stakeholder communication", "progress": 40}
        ],
        "recent_feedback": [
            {"source": "peer", "summary": "Great technical skills, could delegate more"},
            {"source": "manager", "summary": "Ready for more leadership responsibilities"}
        ]
    }
}

POLICY_DOCS = {
    "promotion_criteria": """
    Tech Lead Promotion Criteria:
    1. Demonstrated technical excellence (3+ years experience)
    2. Successfully led at least 2 cross-functional projects
    3. Mentored junior team members
    4. Strong stakeholder communication skills
    5. Manager and skip-level endorsement
    """,
    "development_resources": """
    Leadership Development Resources:
    - "Effective Delegation" workshop (2 hours)
    - "Stakeholder Management" e-learning (1 hour)
    - Tech Lead shadowing program (apply via HR portal)
    - Monthly leadership book club
    """
}


# Define tools for the coach
@tool
def get_user_goals(user_id: str) -> str:
    """Retrieve the user's current goals and progress."""
    profile = USER_PROFILES.get(user_id, {})
    goals = profile.get("goals", [])
    return json.dumps(goals, indent=2)


@tool
def get_recent_feedback(user_id: str) -> str:
    """Retrieve recent feedback the user has received."""
    profile = USER_PROFILES.get(user_id, {})
    feedback = profile.get("recent_feedback", [])
    return json.dumps(feedback, indent=2)


@tool
def search_policies(query: str) -> str:
    """Search company policies and frameworks for relevant information."""
    # Simple keyword matching (would be vector search in production)
    results = []
    for doc_name, content in POLICY_DOCS.items():
        if any(word.lower() in content.lower() for word in query.split()):
            results.append({"document": doc_name, "excerpt": content[:500]})
    
    if not results:
        return "No relevant policies found."
    return json.dumps(results, indent=2)


@tool
def create_action_item(
    user_id: str,
    title: str,
    due_date: str,
    linked_goal_id: str = None
) -> str:
    """Create a new action item for the user."""
    action = {
        "id": f"action_{datetime.now().timestamp()}",
        "title": title,
        "due_date": due_date,
        "linked_goal": linked_goal_id,
        "status": "pending",
        "created": datetime.now().isoformat()
    }
    return f"Created action item: {json.dumps(action)}"


@tool
def suggest_learning_resource(topic: str) -> str:
    """Suggest relevant learning resources for a development area."""
    resources = {
        "leadership": ["Tech Lead Fundamentals course", "Leadership coaching sessions"],
        "communication": ["Stakeholder Management workshop", "Presentation skills training"],
        "delegation": ["Effective Delegation e-learning", "Manager toolkit resources"],
        "default": ["Career development portal", "Internal learning catalog"]
    }
    
    for key, items in resources.items():
        if key in topic.lower():
            return f"Recommended resources for {topic}: {', '.join(items)}"
    
    return f"Recommended resources: {', '.join(resources['default'])}"


@tool
def schedule_followup(
    user_id: str,
    days_from_now: int,
    topic: str
) -> str:
    """Schedule a follow-up coaching session."""
    followup_date = datetime.now() + timedelta(days=days_from_now)
    return f"Follow-up scheduled for {followup_date.strftime('%Y-%m-%d')} to discuss: {topic}"


# All available tools
TOOLS = [
    get_user_goals,
    get_recent_feedback,
    search_policies,
    create_action_item,
    suggest_learning_resource,
    schedule_followup
]


COACH_PROMPT = """You are an advanced HR coaching assistant with access to organizational tools and data.

## Your Capabilities
- Access user's goals and track progress
- Review feedback from peers and managers
- Search company policies and frameworks
- Create action items and learning plans
- Schedule follow-up sessions

## User Profile
{user_profile}

## Coaching Approach
1. Ground advice in actual data (goals, feedback, policies)
2. Create concrete action items when appropriate
3. Connect recommendations to available resources
4. Schedule follow-ups for accountability
5. Always explain your reasoning

Be proactive in using tools to provide evidence-based coaching."""


def create_tool_coach(llm_with_tools):
    """Create the main coaching agent with tools."""
    
    def coach(state: CoachingState) -> dict:
        system_content = COACH_PROMPT.format(
            user_profile=json.dumps(state.get("user_profile", {}), indent=2)
        )
        
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    return coach


def should_continue(state: CoachingState) -> Literal["tools", "end"]:
    """Determine if we should call tools or end."""
    last_message = state["messages"][-1]
    
    # If the LLM made tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


def build_tool_enabled_coach():
    """Build the full tool-enabled coaching graph."""
    
    # Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(TOOLS)
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("coach", create_tool_coach(llm_with_tools))
    graph.add_node("tools", ToolNode(TOOLS))
    
    # Define flow
    graph.add_edge(START, "coach")
    
    # Conditional: continue with tools or end
    graph.add_conditional_edges(
        "coach",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, go back to coach for response
    graph.add_edge("tools", "coach")
    
    return graph.compile()


def main():
    """Demo the full tool-enabled coach."""
    
    coach = build_tool_enabled_coach()
    
    # Initialize with user profile
    state = {
        "messages": [],
        "user_id": "user_789",
        "user_profile": USER_PROFILES["user_789"],
        "pending_actions": []
    }
    
    print("=== Full HR Coaching System ===")
    print(f"Coaching: {state['user_profile']['name']} ({state['user_profile']['role']})")
    print("\nAvailable tools: goals, feedback, policies, actions, learning, scheduling")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        result = coach.invoke(state)
        state = result
        
        # Find the last AI message (not tool messages)
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nCoach: {msg.content}\n")
                break


if __name__ == "__main__":
    main()
