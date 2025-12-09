"""
Script 1: Basic HR Coaching Bot
Demonstrates: Simple graph with state, single coach node, OpenAI integration
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Define the state schema
class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    coaching_focus: str  # e.g., "career", "performance", "wellbeing"


# System prompt for the HR coach
COACH_SYSTEM_PROMPT = """You are an empathetic and professional HR coaching assistant.

Your role is to:
- Help employees reflect on their career development and goals
- Provide supportive, non-judgmental guidance
- Ask open-ended questions to promote self-discovery
- Never give medical, legal, or clinical psychological advice
- Escalate concerns about harassment, discrimination, or mental health crises

Coaching focus for this session: {coaching_focus}

Remember: You are a coach, not a therapist. Your role is to facilitate thinking, not prescribe solutions."""


def create_coach_node(llm: ChatOpenAI):
    """Create the main coaching node."""
    
    def coach(state: CoachingState) -> dict:
        # Build the system message with context
        system_msg = SystemMessage(
            content=COACH_SYSTEM_PROMPT.format(
                coaching_focus=state.get("coaching_focus", "general development")
            )
        )
        
        # Combine system message with conversation history
        messages = [system_msg] + state["messages"]
        
        # Get response from LLM
        response = llm.invoke(messages)
        
        return {"messages": [response]}
    
    return coach


def build_basic_coach_graph():
    """Build and compile the basic coaching graph."""
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Create the graph
    graph = StateGraph(CoachingState)
    
    # Add the coach node
    graph.add_node("coach", create_coach_node(llm))
    
    # Define edges
    graph.add_edge(START, "coach")
    graph.add_edge("coach", END)
    
    # Compile and return
    return graph.compile()


def main():
    """Demo the basic coaching bot."""
    
    # Build the graph
    coach = build_basic_coach_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "user_name": "Alex",
        "coaching_focus": "career development"
    }
    
    print("=== HR Coaching Bot (Basic) ===")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        # Add a user message and invoke
        state["messages"].append(HumanMessage(content=user_input))
        result = coach.invoke(state)
        
        # Update state and display response
        state = result
        ai_message = state["messages"][-1]
        print(f"\nCoach: {ai_message.content}\n")


if __name__ == "__main__":
    main()
