"""
16. Memory-Augmented Coach
==========================
Implements a sophisticated memory system with episodic, semantic,
and procedural memory for longitudinal coaching relationships.

Key concepts:
- Episodic memory: Session summaries and key moments
- Semantic memory: Stable inferences about the person
- Procedural memory: How to work with this person
- Memory retrieval and relevance scoring
- Memory updates and lifecycle management
"""

from typing import Annotated, TypedDict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# MEMORY DATA STRUCTURES
# ============================================================

@dataclass
class EpisodicMemory:
    """A memory of a specific session or interaction."""
    id: str
    timestamp: str
    summary: str
    key_moments: list[str]
    topics: list[str]
    emotional_tone: str
    commitments: list[str]
    insights: list[str]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "key_moments": self.key_moments,
            "topics": self.topics,
            "emotional_tone": self.emotional_tone,
            "commitments": self.commitments,
            "insights": self.insights
        }


@dataclass
class SemanticMemory:
    """A stable inference about the person."""
    id: str
    statement: str
    confidence: float  # 0-1
    evidence: list[str]
    category: str  # values, strengths, challenges, preferences
    last_updated: str
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "category": self.category,
            "last_updated": self.last_updated
        }


@dataclass
class ProceduralMemory:
    """How to work effectively with this person."""
    communication_style: str  # direct, exploratory, supportive
    preferred_question_types: list[str]
    topics_to_avoid: list[str]
    motivators: list[str]
    best_session_length: str
    preferred_check_in_frequency: str
    custom_instructions: list[str]
    
    def to_dict(self) -> dict:
        return {
            "communication_style": self.communication_style,
            "preferred_question_types": self.preferred_question_types,
            "topics_to_avoid": self.topics_to_avoid,
            "motivators": self.motivators,
            "best_session_length": self.best_session_length,
            "preferred_check_in_frequency": self.preferred_check_in_frequency,
            "custom_instructions": self.custom_instructions
        }


# ============================================================
# MEMORY STORE
# ============================================================

class MemoryStore:
    """Manages all memory types for a user."""
    
    def __init__(self):
        self.episodic: list[EpisodicMemory] = []
        self.semantic: list[SemanticMemory] = []
        self.procedural: ProceduralMemory = self._default_procedural()
        self._seed_sample_memories()
    
    def _default_procedural(self) -> ProceduralMemory:
        return ProceduralMemory(
            communication_style="supportive",
            preferred_question_types=["reflective", "exploratory"],
            topics_to_avoid=[],
            motivators=["growth", "impact"],
            best_session_length="15-20 minutes",
            preferred_check_in_frequency="weekly",
            custom_instructions=[]
        )
    
    def _seed_sample_memories(self):
        """Add sample memories for demonstration."""
        
        # Episodic memories (past sessions)
        self.episodic = [
            EpisodicMemory(
                id="ep-1",
                timestamp="2024-11-15T10:00:00",
                summary="Discussed frustration with lack of promotion. Explored feelings of being overlooked. "
                        "Identified need for more visibility with leadership.",
                key_moments=[
                    "Expressed frustration about peers being promoted",
                    "Realized they hadn't clearly communicated career goals to manager",
                    "Felt relieved after venting"
                ],
                topics=["career", "promotion", "visibility", "frustration"],
                emotional_tone="frustrated then relieved",
                commitments=["Schedule career conversation with manager within 2 weeks"],
                insights=["Recognition that self-advocacy is uncomfortable but necessary"]
            ),
            EpisodicMemory(
                id="ep-2",
                timestamp="2024-11-22T14:30:00",
                summary="Followed up on career conversation. They had the talk with their manager and it went well. "
                        "Manager was supportive and outlined path to promotion. Discussed specific skills to develop.",
                key_moments=[
                    "Manager was receptive and supportive",
                    "Got clarity on promotion criteria",
                    "Identified stakeholder management as key gap"
                ],
                topics=["career", "promotion", "stakeholder management", "success"],
                emotional_tone="optimistic and motivated",
                commitments=["Start monthly skip-levels with director", "Take on cross-team project"],
                insights=["Conversations they fear often go better than expected"]
            ),
            EpisodicMemory(
                id="ep-3",
                timestamp="2024-12-01T11:00:00",
                summary="Discussed work-life balance challenges. Heavy workload causing stress. "
                        "Explored delegation opportunities and boundary setting.",
                key_moments=[
                    "Working late most nights",
                    "Difficulty saying no to requests",
                    "Identified perfectionism as contributing factor"
                ],
                topics=["workload", "stress", "delegation", "boundaries"],
                emotional_tone="stressed but self-aware",
                commitments=["Delegate one task this week", "Leave on time twice"],
                insights=["Perfectionism and difficulty delegating are connected"]
            ),
        ]
        
        # Semantic memories (stable inferences)
        self.semantic = [
            SemanticMemory(
                id="sem-1",
                statement="Values autonomy and ownership in their work",
                confidence=0.9,
                evidence=["Frustrated when micromanaged", "Thrives with clear goals and freedom"],
                category="values",
                last_updated="2024-11-22"
            ),
            SemanticMemory(
                id="sem-2",
                statement="Struggles with self-advocacy and asking for what they need",
                confidence=0.85,
                evidence=["Delayed career conversation", "Difficulty saying no", "Discomfort being visible"],
                category="challenges",
                last_updated="2024-12-01"
            ),
            SemanticMemory(
                id="sem-3",
                statement="Highly conscientious with perfectionist tendencies",
                confidence=0.8,
                evidence=["Works late to ensure quality", "Difficulty delegating", "High standards"],
                category="strengths",  # Can be both strength and challenge
                last_updated="2024-12-01"
            ),
            SemanticMemory(
                id="sem-4",
                statement="Responds well to frameworks and structured approaches",
                confidence=0.75,
                evidence=["Engaged with career criteria framework", "Likes clear action items"],
                category="preferences",
                last_updated="2024-11-22"
            ),
            SemanticMemory(
                id="sem-5",
                statement="Career growth and making impact are primary motivators",
                confidence=0.9,
                evidence=["Promotion importance", "Cross-team project interest", "Visibility goals"],
                category="values",
                last_updated="2024-11-22"
            ),
        ]
        
        # Procedural memory
        self.procedural = ProceduralMemory(
            communication_style="supportive with gentle challenge",
            preferred_question_types=["reflective", "reframing", "action-oriented"],
            topics_to_avoid=[],
            motivators=["career growth", "impact", "recognition"],
            best_session_length="20 minutes",
            preferred_check_in_frequency="weekly",
            custom_instructions=[
                "They appreciate direct feedback but need it framed constructively",
                "Connect discussions back to career goals when possible",
                "Celebrate small wins - they tend to minimize their successes"
            ]
        )
    
    def get_relevant_episodic(self, topics: list[str], limit: int = 3) -> list[EpisodicMemory]:
        """Retrieve episodic memories relevant to given topics."""
        scored = []
        for mem in self.episodic:
            score = len(set(topics) & set(mem.topics))
            scored.append((score, mem))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit] if _ > 0]
    
    def get_semantic_by_category(self, category: str) -> list[SemanticMemory]:
        """Get semantic memories by category."""
        return [m for m in self.semantic if m.category == category]
    
    def add_episodic(self, memory: EpisodicMemory):
        """Add a new episodic memory."""
        self.episodic.append(memory)
    
    def update_semantic(self, memory: SemanticMemory):
        """Update or add a semantic memory."""
        for i, existing in enumerate(self.semantic):
            if existing.id == memory.id:
                self.semantic[i] = memory
                return
        self.semantic.append(memory)


MEMORY_STORE = MemoryStore()


# ============================================================
# STATE DEFINITION
# ============================================================

class MemoryCoachState(TypedDict):
    messages: Annotated[list, add_messages]
    current_topics: list[str]
    retrieved_episodic: list[dict]
    retrieved_semantic: list[dict]
    procedural_context: dict
    session_insights: list[str]
    proposed_memory_updates: dict


# ============================================================
# NODE FUNCTIONS
# ============================================================

def extract_topics(state: MemoryCoachState) -> dict:
    """Extract topics from the current message."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    last_message = state["messages"][-1].content
    
    prompt = f"""Extract key topics from this message for memory retrieval.

Message: {last_message}

Return JSON:
{{
    "topics": ["topic1", "topic2"],
    "emotional_indicators": ["any emotions detected"],
    "is_follow_up": true/false
}}

Topics should be single words or short phrases like: career, stress, delegation, relationships, feedback, goals, promotion, workload, boundaries"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {"current_topics": parsed.get("topics", [])}
    except:
        return {"current_topics": []}


def retrieve_memories(state: MemoryCoachState) -> dict:
    """Retrieve relevant memories based on current context."""
    topics = state.get("current_topics", [])
    
    # Get relevant episodic memories
    relevant_episodic = MEMORY_STORE.get_relevant_episodic(topics, limit=3)
    
    # Get all semantic memories (they're usually compact enough)
    semantic = MEMORY_STORE.semantic
    
    # Get procedural
    procedural = MEMORY_STORE.procedural
    
    return {
        "retrieved_episodic": [m.to_dict() for m in relevant_episodic],
        "retrieved_semantic": [m.to_dict() for m in semantic],
        "procedural_context": procedural.to_dict()
    }


def generate_response(state: MemoryCoachState) -> dict:
    """Generate a memory-informed coaching response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    last_message = state["messages"][-1].content
    episodic = state.get("retrieved_episodic", [])
    semantic = state.get("retrieved_semantic", [])
    procedural = state.get("procedural_context", {})
    
    # Build context from memories
    memory_context = ""
    
    if episodic:
        memory_context += "RELEVANT PAST SESSIONS:\n"
        for mem in episodic:
            memory_context += f"- {mem['timestamp'][:10]}: {mem['summary']}\n"
            if mem['commitments']:
                memory_context += f"  Commitments made: {', '.join(mem['commitments'])}\n"
        memory_context += "\n"
    
    if semantic:
        memory_context += "WHAT I KNOW ABOUT THIS PERSON:\n"
        for mem in semantic:
            memory_context += f"- {mem['statement']} (confidence: {mem['confidence']:.0%})\n"
        memory_context += "\n"
    
    if procedural:
        memory_context += "HOW TO WORK WITH THEM:\n"
        memory_context += f"- Style: {procedural.get('communication_style', 'supportive')}\n"
        memory_context += f"- Motivators: {', '.join(procedural.get('motivators', []))}\n"
        for instruction in procedural.get('custom_instructions', []):
            memory_context += f"- {instruction}\n"
    
    prompt = f"""You are an HR coach with history with this person.

{memory_context}

CURRENT MESSAGE: {last_message}

Respond as their coach:
1. Reference relevant history naturally (don't list it, weave it in)
2. Build on past conversations and commitments
3. Follow their preferred communication style
4. Ask questions that deepen their reflection
5. Connect to their stated goals and values

Be warm, personal, and demonstrate that you remember them."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [AIMessage(content=response.content)]}


def extract_session_insights(state: MemoryCoachState) -> dict:
    """Extract insights from this session to potentially store."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {"session_insights": [], "proposed_memory_updates": {}}
    
    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Coach'}: {m.content}"
        for m in messages[-4:]  # Last few exchanges
    ])
    
    semantic = state.get("retrieved_semantic", [])
    
    prompt = f"""Analyze this coaching exchange for memory updates.

Conversation:
{conversation}

Existing beliefs about this person:
{json.dumps(semantic, indent=2)}

Return JSON:
{{
    "session_insights": ["key insight or moment from this exchange"],
    "new_semantic_inferences": [
        {{
            "statement": "new stable inference",
            "confidence": 0.7,
            "category": "values|strengths|challenges|preferences",
            "evidence": "what supports this"
        }}
    ],
    "semantic_updates": [
        {{
            "id": "existing-id",
            "confidence_delta": 0.1,
            "new_evidence": "what reinforces or contradicts"
        }}
    ],
    "topics_discussed": ["topic1", "topic2"]
}}

Only include meaningful insights, not trivial observations."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "session_insights": parsed.get("session_insights", []),
            "proposed_memory_updates": {
                "new_semantic": parsed.get("new_semantic_inferences", []),
                "updates": parsed.get("semantic_updates", []),
                "topics": parsed.get("topics_discussed", [])
            }
        }
    except:
        return {"session_insights": [], "proposed_memory_updates": {}}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_memory_coach() -> StateGraph:
    """Build the memory-augmented coaching graph."""
    
    graph = StateGraph(MemoryCoachState)
    
    # Add nodes
    graph.add_node("extract_topics", extract_topics)
    graph.add_node("retrieve_memories", retrieve_memories)
    graph.add_node("generate_response", generate_response)
    graph.add_node("extract_insights", extract_session_insights)
    
    # Flow
    graph.add_edge(START, "extract_topics")
    graph.add_edge("extract_topics", "retrieve_memories")
    graph.add_edge("retrieve_memories", "generate_response")
    graph.add_edge("generate_response", "extract_insights")
    graph.add_edge("extract_insights", END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    coach = build_memory_coach()
    
    print("=" * 60)
    print("MEMORY-AUGMENTED COACH")
    print("=" * 60)
    print("\nThis coach remembers your history and builds on past conversations.")
    print("It has memories from 3 previous sessions about:")
    print("  - Career/promotion conversations")
    print("  - Work-life balance challenges")
    print("  - Delegation and boundaries")
    print("\nTry asking about these topics to see memory in action.\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nUntil next time! 🌟")
            break
        if user_input.lower() == 'memories':
            # Debug: show current memories
            print("\n--- EPISODIC MEMORIES ---")
            for m in MEMORY_STORE.episodic:
                print(f"{m.timestamp[:10]}: {m.summary[:80]}...")
            print("\n--- SEMANTIC MEMORIES ---")
            for m in MEMORY_STORE.semantic:
                print(f"[{m.category}] {m.statement} ({m.confidence:.0%})")
            print()
            continue
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "current_topics": [],
            "retrieved_episodic": [],
            "retrieved_semantic": [],
            "procedural_context": {},
            "session_insights": [],
            "proposed_memory_updates": {}
        }
        
        result = coach.invoke(state)
        
        # Show response
        response = result["messages"][-1].content
        print(f"\nCoach: {response}\n")
        
        # Show what memories were used (debug info)
        if result.get("retrieved_episodic"):
            print(f"  [Used {len(result['retrieved_episodic'])} past sessions for context]")
        if result.get("session_insights"):
            print(f"  [Noted: {result['session_insights'][0][:60]}...]")
        print()


if __name__ == "__main__":
    main()
