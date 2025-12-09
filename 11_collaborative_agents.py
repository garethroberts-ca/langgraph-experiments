"""
Script 11: Multi-Agent Collaboration with Shared Workspace
Demonstrates: Agent-to-agent messaging, shared blackboard, negotiation,
              collaborative problem-solving, agent teams with roles
"""

from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import operator
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool


# ============================================================
# AGENT MESSAGING SYSTEM
# ============================================================

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NEGOTIATE = "negotiate"
    DELEGATE = "delegate"
    ESCALATE = "escalate"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str
    sender: str
    recipient: str  # Can be "all" for broadcast
    message_type: MessageType
    content: str
    context: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    requires_response: bool = False
    in_reply_to: Optional[str] = None


class SharedWorkspace(TypedDict):
    """Shared blackboard for agent collaboration."""
    problem_statement: str
    constraints: list[str]
    proposed_solutions: list[dict]
    decisions_made: list[dict]
    open_questions: list[str]
    artifacts: dict  # Shared documents, plans, etc.


class AgentState(TypedDict):
    """State for each agent in the collaboration."""
    agent_id: str
    role: str
    current_task: str
    completed_tasks: list[str]
    pending_messages: list[dict]


class CollaborativeState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    
    # Collaboration infrastructure
    workspace: SharedWorkspace
    agent_messages: Annotated[list[dict], operator.add]
    agent_states: dict[str, AgentState]
    
    # Orchestration
    current_speaker: str
    conversation_round: int
    max_rounds: int
    consensus_reached: bool
    final_recommendation: str


# ============================================================
# AGENT DEFINITIONS
# ============================================================

AGENT_CONFIGS = {
    "hr_strategist": {
        "role": "HR Strategy Lead",
        "expertise": ["organizational design", "talent strategy", "culture"],
        "prompt": """You are the HR Strategy Lead in a collaborative coaching team.

Your expertise: Organizational design, talent strategy, company culture
Your role: Provide strategic HR perspective, ensure alignment with business goals

When collaborating:
- Consider long-term organizational implications
- Challenge proposals that don't align with HR strategy
- Delegate tactical details to specialists
- Synthesize inputs into coherent strategy

Respond with your analysis AND any messages to other agents."""
    },
    
    "career_advisor": {
        "role": "Career Development Specialist",
        "expertise": ["career pathing", "skill development", "transitions"],
        "prompt": """You are the Career Development Specialist in a collaborative team.

Your expertise: Career pathing, skill development, career transitions
Your role: Focus on individual career growth and development

When collaborating:
- Advocate for the employee's career aspirations
- Identify skill gaps and development opportunities
- Propose concrete career plans
- Request data from other specialists when needed

Respond with your analysis AND any messages to other agents."""
    },
    
    "org_psychologist": {
        "role": "Organizational Psychologist",
        "expertise": ["behavior", "motivation", "team dynamics"],
        "prompt": """You are the Organizational Psychologist in a collaborative team.

Your expertise: Human behavior, motivation, team dynamics, wellbeing
Your role: Provide psychological insights and ensure human-centered solutions

When collaborating:
- Assess psychological implications of proposals
- Identify potential resistance or challenges
- Suggest approaches for behavior change
- Flag wellbeing concerns

Respond with your analysis AND any messages to other agents."""
    },
    
    "data_analyst": {
        "role": "People Analytics Specialist",
        "expertise": ["metrics", "benchmarks", "data analysis"],
        "prompt": """You are the People Analytics Specialist in a collaborative team.

Your expertise: HR metrics, industry benchmarks, data-driven insights
Your role: Ground discussions in data and provide evidence-based recommendations

When collaborating:
- Request clarification on data needs
- Provide relevant metrics and benchmarks
- Challenge assumptions with data
- Quantify impact of proposed solutions

Respond with your analysis AND any messages to other agents."""
    }
}


# ============================================================
# COLLABORATION TOOLS
# ============================================================

@tool
def post_to_workspace(
    section: str,
    content: str,
    agent_id: str
) -> str:
    """Post content to the shared workspace."""
    return json.dumps({
        "action": "workspace_update",
        "section": section,
        "content": content,
        "posted_by": agent_id,
        "timestamp": datetime.now().isoformat()
    })


@tool
def send_agent_message(
    recipient: str,
    message_type: str,
    content: str,
    requires_response: bool = False
) -> str:
    """Send a message to another agent."""
    return json.dumps({
        "action": "message_sent",
        "recipient": recipient,
        "type": message_type,
        "content": content,
        "requires_response": requires_response
    })


@tool
def propose_solution(
    title: str,
    description: str,
    pros: str,
    cons: str,
    confidence: float
) -> str:
    """Propose a solution to the shared workspace."""
    return json.dumps({
        "action": "solution_proposed",
        "solution": {
            "title": title,
            "description": description,
            "pros": pros.split(","),
            "cons": cons.split(","),
            "confidence": confidence
        }
    })


@tool
def vote_on_solution(
    solution_id: str,
    vote: str,
    rationale: str
) -> str:
    """Vote on a proposed solution (support/oppose/abstain)."""
    return json.dumps({
        "action": "vote_cast",
        "solution_id": solution_id,
        "vote": vote,
        "rationale": rationale
    })


@tool
def request_data(
    data_type: str,
    context: str
) -> str:
    """Request specific data from the data analyst."""
    return json.dumps({
        "action": "data_requested",
        "data_type": data_type,
        "context": context
    })


COLLAB_TOOLS = [
    post_to_workspace,
    send_agent_message,
    propose_solution,
    vote_on_solution,
    request_data
]


# ============================================================
# AGENT NODES
# ============================================================

def create_agent_node(agent_id: str, config: dict):
    """Create a collaborative agent node."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(COLLAB_TOOLS)
    
    def agent(state: CollaborativeState) -> dict:
        # Build context for this agent
        workspace = state.get("workspace", {})
        pending_messages = [
            m for m in state.get("agent_messages", [])
            if m.get("recipient") in [agent_id, "all"]
            and m.get("processed_by", []) is None or agent_id not in m.get("processed_by", [])
        ]
        
        context = f"""
## Current Workspace
Problem: {workspace.get('problem_statement', 'Not defined')}
Constraints: {workspace.get('constraints', [])}
Proposed Solutions: {json.dumps(workspace.get('proposed_solutions', []), indent=2)}
Open Questions: {workspace.get('open_questions', [])}

## Messages for You
{json.dumps(pending_messages, indent=2) if pending_messages else 'No pending messages'}

## Conversation Round: {state.get('conversation_round', 1)}/{state.get('max_rounds', 5)}

Your task: Contribute your expertise to solving this problem collaboratively.
- Analyze the current state
- Respond to any messages addressed to you
- Propose solutions or refinements
- Coordinate with other agents as needed
"""
        
        system_prompt = config["prompt"] + context
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        response = llm_with_tools.invoke(messages)
        
        # Process tool calls to update workspace and messages
        new_messages = []
        workspace_updates = {}
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                try:
                    result = json.loads(tool_call.get("args", {}).get("__result__", "{}"))
                    action = result.get("action")
                    
                    if action == "message_sent":
                        new_messages.append({
                            "id": str(uuid.uuid4()),
                            "sender": agent_id,
                            "recipient": result.get("recipient"),
                            "type": result.get("type"),
                            "content": result.get("content"),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif action == "solution_proposed":
                        workspace_updates["proposed_solutions"] = workspace.get("proposed_solutions", []) + [
                            {**result.get("solution", {}), "proposed_by": agent_id}
                        ]
                except:
                    pass
        
        # Update agent state
        agent_states = state.get("agent_states", {})
        if agent_id in agent_states:
            agent_states[agent_id]["completed_tasks"].append(
                f"Round {state.get('conversation_round', 1)} contribution"
            )
        
        return {
            "messages": [AIMessage(content=f"[{config['role']}]: {response.content}")],
            "agent_messages": new_messages,
            "agent_states": agent_states,
            "workspace": {**workspace, **workspace_updates} if workspace_updates else workspace
        }
    
    return agent


# ============================================================
# ORCHESTRATION
# ============================================================

def create_facilitator_node():
    """Create the facilitator that orchestrates collaboration."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def facilitator(state: CollaborativeState) -> dict:
        workspace = state.get("workspace", {})
        round_num = state.get("conversation_round", 1)
        max_rounds = state.get("max_rounds", 5)
        
        # Analyze collaboration state
        solutions = workspace.get("proposed_solutions", [])
        
        analysis_prompt = f"""You are facilitating a multi-agent HR coaching collaboration.

Current Round: {round_num}/{max_rounds}
Problem: {workspace.get('problem_statement', '')}
Proposed Solutions: {len(solutions)}
Open Questions: {len(workspace.get('open_questions', []))}

Recent discussion:
{state['messages'][-4:] if len(state['messages']) > 4 else state['messages']}

Decide:
1. Has consensus been reached? (true/false)
2. Which agent should speak next? (hr_strategist/career_advisor/org_psychologist/data_analyst/synthesize)
3. What should be the focus of the next contribution?

Respond in JSON:
{{"consensus": false, "next_speaker": "agent_id", "focus": "what to discuss"}}
"""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            decision = {"consensus": False, "next_speaker": "hr_strategist", "focus": "continue discussion"}
        
        # Check for forced consensus at max rounds
        if round_num >= max_rounds:
            decision["consensus"] = True
            decision["next_speaker"] = "synthesize"
        
        return {
            "current_speaker": decision.get("next_speaker", "hr_strategist"),
            "consensus_reached": decision.get("consensus", False),
            "conversation_round": round_num + 1
        }
    
    return facilitator


def synthesize_recommendation(state: CollaborativeState) -> dict:
    """Synthesize all agent contributions into final recommendation."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    workspace = state.get("workspace", {})
    
    synthesis_prompt = f"""Synthesize the collaborative discussion into a final HR coaching recommendation.

## Problem Statement
{workspace.get('problem_statement', '')}

## Constraints
{workspace.get('constraints', [])}

## Proposed Solutions
{json.dumps(workspace.get('proposed_solutions', []), indent=2)}

## Key Decisions Made
{json.dumps(workspace.get('decisions_made', []), indent=2)}

## Full Discussion
{[m.content for m in state['messages'] if isinstance(m, AIMessage)]}

Create a comprehensive recommendation that:
1. Synthesizes the best elements from all proposals
2. Addresses all constraints
3. Incorporates insights from each specialist
4. Provides clear, actionable next steps
5. Notes any areas of disagreement or uncertainty

Format as a structured coaching recommendation."""
    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    
    return {
        "final_recommendation": response.content,
        "messages": [AIMessage(content=f"[FINAL RECOMMENDATION]\n\n{response.content}")]
    }


def route_collaboration(state: CollaborativeState) -> str:
    """Route to next speaker or synthesis."""
    
    if state.get("consensus_reached"):
        return "synthesize"
    
    speaker = state.get("current_speaker", "hr_strategist")
    
    if speaker in AGENT_CONFIGS:
        return speaker
    
    return "synthesize"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_collaborative_graph():
    """Build the multi-agent collaboration graph."""
    
    graph = StateGraph(CollaborativeState)
    
    # Add facilitator
    graph.add_node("facilitator", create_facilitator_node())
    
    # Add all agents
    for agent_id, config in AGENT_CONFIGS.items():
        graph.add_node(agent_id, create_agent_node(agent_id, config))
    
    # Add synthesis node
    graph.add_node("synthesize", synthesize_recommendation)
    
    # Flow: facilitator decides who speaks
    graph.add_edge(START, "facilitator")
    
    graph.add_conditional_edges(
        "facilitator",
        route_collaboration,
        {
            "hr_strategist": "hr_strategist",
            "career_advisor": "career_advisor",
            "org_psychologist": "org_psychologist",
            "data_analyst": "data_analyst",
            "synthesize": "synthesize"
        }
    )
    
    # Each agent goes back to facilitator
    for agent_id in AGENT_CONFIGS.keys():
        graph.add_edge(agent_id, "facilitator")
    
    graph.add_edge("synthesize", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# MAIN
# ============================================================

def main():
    """Demo the collaborative coaching system."""
    
    print("=" * 70)
    print("MULTI-AGENT COLLABORATIVE HR COACHING")
    print("=" * 70)
    print("\nAgents:")
    for agent_id, config in AGENT_CONFIGS.items():
        print(f"  • {config['role']}: {', '.join(config['expertise'])}")
    print("\nThe agents will collaborate to solve your HR challenge.")
    print("Type 'quit' to exit\n")
    print("=" * 70)
    
    graph = build_collaborative_graph()
    
    session_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    # Get the problem from user
    problem = input("\nDescribe your HR coaching challenge:\n> ").strip()
    
    if problem.lower() == "quit":
        return
    
    state = {
        "messages": [HumanMessage(content=problem)],
        "user_id": "user_collab",
        "workspace": {
            "problem_statement": problem,
            "constraints": [],
            "proposed_solutions": [],
            "decisions_made": [],
            "open_questions": [],
            "artifacts": {}
        },
        "agent_messages": [],
        "agent_states": {
            agent_id: {
                "agent_id": agent_id,
                "role": cfg["role"],
                "current_task": "",
                "completed_tasks": [],
                "pending_messages": []
            }
            for agent_id, cfg in AGENT_CONFIGS.items()
        },
        "current_speaker": "",
        "conversation_round": 1,
        "max_rounds": 4,
        "consensus_reached": False,
        "final_recommendation": ""
    }
    
    print("\n[Starting collaborative session...]\n")
    
    result = graph.invoke(state, config)
    
    # Show the collaboration
    print("\n" + "=" * 70)
    print("COLLABORATION SUMMARY")
    print("=" * 70)
    
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content}\n")
            print("-" * 50)
    
    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Rounds: {result.get('conversation_round', 0)}")
    print(f"Solutions Proposed: {len(result.get('workspace', {}).get('proposed_solutions', []))}")


if __name__ == "__main__":
    main()
