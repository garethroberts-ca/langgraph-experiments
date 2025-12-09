"""
Script 13 (Fast): Lightweight Self-Reflective Agent
Same pattern, 3x fewer LLM calls
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    critique: str
    score: float
    iteration: int


def generate(state: State) -> dict:
    """Generate initial response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    query = state["messages"][-1].content
    
    response = llm.invoke([HumanMessage(content=f"You are an HR coach. Respond to: {query}")])
    return {"draft": response.content, "iteration": 1}


def critique_and_score(state: State) -> dict:
    """Single-pass critique with score."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = f"""Rate this HR coaching response 0-1 and give brief critique.

Query: {state["messages"][-1].content}
Response: {state["draft"]}

JSON only: {{"score": 0.8, "critique": "brief issues"}}"""
    
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(result.content)
        return {"score": parsed.get("score", 0.7), "critique": parsed.get("critique", "")}
    except:
        return {"score": 0.7, "critique": result.content}


def refine(state: State) -> dict:
    """Refine based on critique."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    
    prompt = f"""Improve this response based on critique.

Original: {state["draft"]}
Critique: {state["critique"]}

Provide only the improved response."""
    
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": result.content, "iteration": state["iteration"] + 1}


def should_refine(state: State) -> Literal["refine", "done"]:
    if state["score"] >= 0.8 or state["iteration"] >= 2:
        return "done"
    return "refine"


def finalize(state: State) -> dict:
    return {"messages": [AIMessage(content=state["draft"])]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique_and_score)
    graph.add_node("refine", refine)
    graph.add_node("finalize", finalize)
    
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_refine, {"refine": "refine", "done": "finalize"})
    graph.add_edge("refine", "critique")
    graph.add_edge("finalize", END)
    
    return graph.compile()


if __name__ == "__main__":
    print("=== Fast Reflective Coach ===\n")
    graph = build_graph()
    
    while True:
        q = input("You: ").strip()
        if q.lower() == "quit":
            break
        
        state = {"messages": [HumanMessage(content=q)], "draft": "", "critique": "", "score": 0, "iteration": 0}
        
        for event in graph.stream(state, stream_mode="updates"):
            for node, output in event.items():
                if node == "critique":
                    print(f"[Score: {output.get('score', 0):.2f}]")
        
        final = graph.invoke(state)
        print(f"\nCoach: {final['messages'][-1].content}\n")
