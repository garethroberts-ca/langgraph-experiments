"""
Script 3: HR Coach with Memory and Session Management
Demonstrates: Checkpointing, memory persistence, longitudinal coaching
"""

from typing import Annotated, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json


class CoachingMemory(TypedDict):
    """Structured memory for longitudinal coaching."""
    goals: list[dict]
    action_items: list[dict]
    key_insights: list[str]
    preferences: dict  # How user likes to be coached
    session_summaries: list[dict]


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    memory: CoachingMemory
    current_session_start: str


COACH_PROMPT = """You are a longitudinal HR coach maintaining an ongoing relationship with this employee.

## User's Coaching Memory
Goals: {goals}
Recent Action Items: {action_items}
Key Insights from Past Sessions: {insights}
Coaching Preferences: {preferences}

## Your Approach
- Reference past conversations and progress naturally
- Track commitments and follow up on action items
- Build on established insights and patterns
- Adapt your style to their preferences
- Celebrate progress and gently explore setbacks

Current session started: {session_start}"""


def create_memory_aware_coach(llm: ChatOpenAI):
    """Coach that uses and updates memory."""
    
    def coach(state: CoachingState) -> dict:
        memory = state.get("memory", {})
        
        system_content = COACH_PROMPT.format(
            goals=json.dumps(memory.get("goals", []), indent=2),
            action_items=json.dumps(memory.get("action_items", [])[-5:], indent=2),
            insights=memory.get("key_insights", [])[-5:],
            preferences=memory.get("preferences", {}),
            session_start=state.get("current_session_start", "unknown")
        )
        
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm.invoke(messages)
        
        return {"messages": [response]}
    
    return coach


def create_memory_updater(llm: ChatOpenAI):
    """Extract and update memory from conversation."""
    
    def update_memory(state: CoachingState) -> dict:
        # Only update periodically (every 3 messages from user)
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if len(user_messages) % 3 != 0:
            return {}
        
        memory = state.get("memory", {
            "goals": [],
            "action_items": [],
            "key_insights": [],
            "preferences": {},
            "session_summaries": []
        })
        
        # Get recent conversation for extraction
        recent_messages = state["messages"][-6:]
        conversation = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Coach'}: {m.content}"
            for m in recent_messages
        ])
        
        extraction_prompt = f"""Analyze this coaching conversation and extract updates:

{conversation}

Return a JSON object with any NEW items to add:
{{
    "new_goals": ["goal text if any new goals mentioned"],
    "new_action_items": ["action item if any commitments made"],
    "new_insights": ["key insight if any breakthroughs or realizations"],
    "preference_updates": {{"key": "value"}} // e.g., {{"prefers_direct_feedback": true}}
}}

Only include fields with actual new content. Return empty arrays/objects if nothing new."""

        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        
        try:
            updates = json.loads(response.content)
            
            # Merge updates into memory
            if updates.get("new_goals"):
                for goal in updates["new_goals"]:
                    memory["goals"].append({
                        "title": goal,
                        "created": datetime.now().isoformat(),
                        "status": "active"
                    })
            
            if updates.get("new_action_items"):
                for item in updates["new_action_items"]:
                    memory["action_items"].append({
                        "task": item,
                        "created": datetime.now().isoformat(),
                        "status": "pending"
                    })
            
            if updates.get("new_insights"):
                memory["key_insights"].extend(updates["new_insights"])
            
            if updates.get("preference_updates"):
                memory["preferences"].update(updates["preference_updates"])
            
            return {"memory": memory}
            
        except json.JSONDecodeError:
            return {}
    
    return update_memory


def build_memory_coach_graph():
    """Build coach with memory and checkpointing."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("coach", create_memory_aware_coach(llm))
    graph.add_node("update_memory", create_memory_updater(llm))
    
    # Define flow: coach responds, then memory updates
    graph.add_edge(START, "coach")
    graph.add_edge("coach", "update_memory")
    graph.add_edge("update_memory", END)
    
    # Add checkpointer for persistence across sessions
    checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)


def main():
    """Demo the memory-enabled coach."""
    
    coach = build_memory_coach_graph()
    
    # Thread ID enables persistence across invocations
    thread_id = "user_123_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state for new session
    initial_state = {
        "messages": [],
        "user_id": "user_123",
        "memory": {
            "goals": [{"title": "Develop leadership skills", "status": "active"}],
            "action_items": [],
            "key_insights": ["Works best with structured feedback"],
            "preferences": {"communication_style": "direct"},
            "session_summaries": []
        },
        "current_session_start": datetime.now().isoformat()
    }
    
    print("=== Memory-Enabled HR Coach ===")
    print("Your progress is tracked across sessions")
    print("Type 'memory' to see current memory state")
    print("Type 'quit' to exit\n")
    
    state = initial_state
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "memory":
            print(f"\n=== Current Memory ===\n{json.dumps(state.get('memory', {}), indent=2)}\n")
            continue
        
        state["messages"].append(HumanMessage(content=user_input))
        result = coach.invoke(state, config)
        state = result
        
        print(f"\nCoach: {state['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
