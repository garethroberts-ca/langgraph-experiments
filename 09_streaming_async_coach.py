"""
Script 9: HR Coach with Streaming and Async Execution
Demonstrates: Async execution, streaming responses, SQLite persistence, event handling
"""

import asyncio
from typing import Annotated, TypedDict, Literal, AsyncIterator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import json
from datetime import datetime
import aiosqlite


class SessionMetadata(TypedDict):
    session_id: str
    started_at: str
    turns: int
    topics_discussed: list[str]


class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_metadata: SessionMetadata
    current_topic: str
    insights_generated: list[str]
    streaming_enabled: bool


# ============================================================
# ASYNC TOOLS
# ============================================================

@tool
async def async_get_employee_profile(user_id: str) -> str:
    """Asynchronously retrieve employee profile data."""
    # Simulate async database call
    await asyncio.sleep(0.1)
    
    profiles = {
        "user_404": {
            "name": "Riley Johnson",
            "role": "Engineering Manager",
            "direct_reports": 6,
            "tenure_months": 18,
            "current_goals": ["Scale team to 10", "Improve delivery velocity"],
            "development_areas": ["Executive presence", "Strategic planning"]
        }
    }
    return json.dumps(profiles.get(user_id, {"error": "Profile not found"}), indent=2)


@tool
async def async_get_team_health(user_id: str) -> str:
    """Asynchronously retrieve team health metrics."""
    await asyncio.sleep(0.1)
    
    return json.dumps({
        "engagement_score": 78,
        "attrition_risk": "low",
        "recent_departures": 0,
        "eNPS": 42,
        "concerns": ["Workload balance", "Career growth visibility"]
    }, indent=2)


@tool
async def async_get_performance_trends(user_id: str, period: str = "6m") -> str:
    """Asynchronously retrieve performance trends."""
    await asyncio.sleep(0.1)
    
    return json.dumps({
        "period": period,
        "delivery_trend": "improving",
        "quality_trend": "stable",
        "velocity_change": "+15%",
        "blockers_resolved": 12,
        "escalations": 2
    }, indent=2)


@tool
async def async_search_learning_catalog(query: str) -> str:
    """Asynchronously search learning resources."""
    await asyncio.sleep(0.1)
    
    resources = {
        "leadership": [
            {"title": "Executive Presence Masterclass", "duration": "4 hours", "rating": 4.8},
            {"title": "Strategic Leadership Workshop", "duration": "2 days", "rating": 4.6}
        ],
        "management": [
            {"title": "Scaling Engineering Teams", "duration": "3 hours", "rating": 4.7},
            {"title": "Performance Conversations", "duration": "90 min", "rating": 4.5}
        ]
    }
    
    results = []
    for category, items in resources.items():
        if query.lower() in category or any(query.lower() in item["title"].lower() for item in items):
            results.extend(items)
    
    return json.dumps(results[:5] if results else [{"message": "No matching resources"}], indent=2)


@tool
async def async_schedule_coaching_session(
    user_id: str,
    topic: str,
    preferred_time: str
) -> str:
    """Asynchronously schedule a follow-up coaching session."""
    await asyncio.sleep(0.1)
    
    return json.dumps({
        "status": "scheduled",
        "session_id": f"coach_{datetime.now().timestamp()}",
        "topic": topic,
        "scheduled_time": preferred_time,
        "reminder_set": True
    }, indent=2)


ASYNC_TOOLS = [
    async_get_employee_profile,
    async_get_team_health,
    async_get_performance_trends,
    async_search_learning_catalog,
    async_schedule_coaching_session
]


# ============================================================
# ASYNC GRAPH NODES
# ============================================================

def create_async_coach_node(llm_with_tools):
    """Create async coaching node with streaming support."""
    
    async def coach(state: CoachingState) -> dict:
        system = """You are an experienced HR Coach for engineering leaders.

Your expertise:
- Engineering team scaling and organizational design
- Leadership development for technical managers
- Performance management and feedback
- Career development and succession planning

Use available tools to ground your coaching in data.
Be warm but direct. Engineering leaders appreciate efficiency."""
        
        messages = [SystemMessage(content=system)] + state["messages"]
        
        # Use ainvoke for async
        response = await llm_with_tools.ainvoke(messages)
        
        # Update session metadata
        metadata = state.get("session_metadata", {})
        metadata["turns"] = metadata.get("turns", 0) + 1
        
        return {
            "messages": [response],
            "session_metadata": metadata
        }
    
    return coach


async def async_extract_topic(state: CoachingState) -> dict:
    """Extract the main topic being discussed."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if not state["messages"]:
        return {"current_topic": "general"}
    
    last_message = state["messages"][-1].content
    
    prompt = f"""Extract the main coaching topic from this message in 2-3 words.
Message: "{last_message}"

Topics: team_scaling, leadership_development, performance_management, 
        career_planning, conflict_resolution, general

Respond with only the topic."""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    topic = response.content.strip().lower().replace(" ", "_")
    
    # Update topics discussed
    metadata = state.get("session_metadata", {})
    topics = metadata.get("topics_discussed", [])
    if topic not in topics:
        topics.append(topic)
    metadata["topics_discussed"] = topics
    
    return {
        "current_topic": topic,
        "session_metadata": metadata
    }


async def async_generate_insight(state: CoachingState) -> dict:
    """Generate and store coaching insights from the conversation."""
    
    # Only generate insights every few turns
    turns = state.get("session_metadata", {}).get("turns", 0)
    if turns % 3 != 0:
        return {}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    recent_messages = state["messages"][-6:]
    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Coach'}: {m.content}"
        for m in recent_messages if hasattr(m, 'content') and m.content
    ])
    
    prompt = f"""Extract one key coaching insight from this conversation snippet:

{conversation}

Format: A single actionable insight in one sentence."""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    insights = state.get("insights_generated", [])
    insights.append({
        "insight": response.content,
        "turn": turns,
        "topic": state.get("current_topic", "general")
    })
    
    return {"insights_generated": insights}


def should_use_tools(state: CoachingState) -> Literal["tools", "insight"]:
    """Check if tools were called."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "insight"


# ============================================================
# BUILD ASYNC GRAPH
# ============================================================

async def build_streaming_coach():
    """Build the async streaming coach with SQLite persistence."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
    llm_with_tools = llm.bind_tools(ASYNC_TOOLS)
    
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("extract_topic", async_extract_topic)
    graph.add_node("coach", create_async_coach_node(llm_with_tools))
    graph.add_node("tools", ToolNode(ASYNC_TOOLS))
    graph.add_node("insight", async_generate_insight)
    
    # Define flow
    graph.add_edge(START, "extract_topic")
    graph.add_edge("extract_topic", "coach")
    
    graph.add_conditional_edges(
        "coach",
        should_use_tools,
        {
            "tools": "tools",
            "insight": "insight"
        }
    )
    
    graph.add_edge("tools", "coach")
    graph.add_edge("insight", END)
    
    # Use async SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string("coaching_sessions.db") as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        yield compiled


async def stream_response(
    graph,
    state: CoachingState,
    config: dict
) -> AsyncIterator[str]:
    """Stream the response token by token."""
    
    async for event in graph.astream_events(state, config, version="v2"):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
        
        elif kind == "on_tool_start":
            yield f"\n[Using tool: {event['name']}...]\n"
        
        elif kind == "on_tool_end":
            yield f"[Tool complete]\n"


async def run_with_streaming(graph, state: CoachingState, config: dict):
    """Run the graph with streaming output."""
    
    print("\nCoach: ", end="", flush=True)
    
    full_response = ""
    async for chunk in stream_response(graph, state, config):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n")
    return full_response


# ============================================================
# MAIN ASYNC LOOP
# ============================================================

async def main():
    """Main async entry point."""
    
    print("=== Streaming Async HR Coach ===")
    print("Responses stream in real-time")
    print("Session persisted to SQLite")
    print("\nType 'insights' to see generated insights")
    print("Type 'session' to see session metadata")
    print("Type 'quit' to exit\n")
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    state = {
        "messages": [],
        "user_id": "user_404",
        "session_metadata": {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "turns": 0,
            "topics_discussed": []
        },
        "current_topic": "",
        "insights_generated": [],
        "streaming_enabled": True
    }
    
    # Build graph with async context manager
    async with AsyncSqliteSaver.from_conn_string("coaching_sessions.db") as checkpointer:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
        llm_with_tools = llm.bind_tools(ASYNC_TOOLS)
        
        graph_builder = StateGraph(CoachingState)
        graph_builder.add_node("extract_topic", async_extract_topic)
        graph_builder.add_node("coach", create_async_coach_node(llm_with_tools))
        graph_builder.add_node("tools", ToolNode(ASYNC_TOOLS))
        graph_builder.add_node("insight", async_generate_insight)
        
        graph_builder.add_edge(START, "extract_topic")
        graph_builder.add_edge("extract_topic", "coach")
        graph_builder.add_conditional_edges("coach", should_use_tools)
        graph_builder.add_edge("tools", "coach")
        graph_builder.add_edge("insight", END)
        
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
            except EOFError:
                break
            
            if user_input.lower() == "quit":
                break
            
            if user_input.lower() == "insights":
                print("\n=== Generated Insights ===")
                for idx, insight in enumerate(state.get("insights_generated", []), 1):
                    print(f"{idx}. [{insight.get('topic', 'general')}] {insight.get('insight', '')}")
                print()
                continue
            
            if user_input.lower() == "session":
                print("\n=== Session Metadata ===")
                print(json.dumps(state.get("session_metadata", {}), indent=2))
                print()
                continue
            
            state["messages"].append(HumanMessage(content=user_input))
            
            # Stream the response
            await run_with_streaming(graph, state, config)
            
            # Get updated state
            graph_state = await graph.aget_state(config)
            state = dict(graph_state.values)
    
    print(f"\nSession saved: {session_id}")
    print(f"Total turns: {state.get('session_metadata', {}).get('turns', 0)}")
    print(f"Topics: {', '.join(state.get('session_metadata', {}).get('topics_discussed', []))}")


if __name__ == "__main__":
    asyncio.run(main())
