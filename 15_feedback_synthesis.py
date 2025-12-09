"""
15. Feedback Synthesis Agent
============================
Aggregates feedback from multiple sources (reviews, peers, surveys, self-assessment)
and synthesizes into actionable coaching insights.

Key concepts:
- Multi-source feedback aggregation
- Theme extraction and pattern detection
- Contradiction identification
- Strength/growth area mapping
- Evidence-anchored recommendations
"""

from typing import Annotated, TypedDict, Literal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# FEEDBACK DATA MODEL
# ============================================================

class FeedbackSource(str, Enum):
    MANAGER_REVIEW = "manager_review"
    PEER_FEEDBACK = "peer_feedback"
    DIRECT_REPORT = "direct_report"
    SELF_ASSESSMENT = "self_assessment"
    PULSE_SURVEY = "pulse_survey"
    SKIP_LEVEL = "skip_level"
    EXTERNAL = "external_stakeholder"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONSTRUCTIVE = "constructive"  # Areas for growth, framed positively


@dataclass
class FeedbackItem:
    id: str
    source: FeedbackSource
    author_role: str  # e.g., "Manager", "Peer - Engineering", "Direct Report"
    date: str
    content: str
    competency_tags: list[str]
    sentiment: Sentiment
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "author_role": self.author_role,
            "date": self.date,
            "content": self.content,
            "competency_tags": self.competency_tags,
            "sentiment": self.sentiment.value
        }


# ============================================================
# SAMPLE FEEDBACK DATA
# ============================================================

SAMPLE_FEEDBACK = [
    FeedbackItem(
        id="fb-1",
        source=FeedbackSource.MANAGER_REVIEW,
        author_role="Manager",
        date="2024-12-01",
        content="Sarah consistently delivers high-quality work and is a reliable team member. "
                "She's grown significantly in technical depth this year. One area to develop: "
                "Sarah tends to take on too much herself rather than delegating or asking for help. "
                "I'd like to see her build more influence across teams.",
        competency_tags=["delivery", "technical_skills", "delegation", "influence"],
        sentiment=Sentiment.POSITIVE
    ),
    FeedbackItem(
        id="fb-2",
        source=FeedbackSource.PEER_FEEDBACK,
        author_role="Peer - Engineering",
        date="2024-11-15",
        content="Working with Sarah is always productive. She explains complex concepts clearly "
                "and is patient when onboarding new team members. Sometimes she could speak up more "
                "in large meetings - her ideas are valuable but she often holds back.",
        competency_tags=["communication", "mentoring", "assertiveness"],
        sentiment=Sentiment.POSITIVE
    ),
    FeedbackItem(
        id="fb-3",
        source=FeedbackSource.PEER_FEEDBACK,
        author_role="Peer - Product",
        date="2024-11-20",
        content="Sarah is excellent at understanding requirements and translating them into solutions. "
                "She's my go-to person for complex technical questions. The one thing I'd suggest is "
                "more proactive communication on timelines - sometimes I'm not sure where things stand.",
        competency_tags=["collaboration", "technical_skills", "communication"],
        sentiment=Sentiment.POSITIVE
    ),
    FeedbackItem(
        id="fb-4",
        source=FeedbackSource.DIRECT_REPORT,
        author_role="Direct Report",
        date="2024-11-10",
        content="Sarah is a supportive manager who genuinely cares about my growth. She gives good "
                "technical guidance but could provide more regular feedback on my performance. "
                "Sometimes I'm unsure if I'm meeting expectations.",
        competency_tags=["people_management", "feedback", "support"],
        sentiment=Sentiment.CONSTRUCTIVE
    ),
    FeedbackItem(
        id="fb-5",
        source=FeedbackSource.SELF_ASSESSMENT,
        author_role="Self",
        date="2024-12-05",
        content="I feel I've made good progress on technical skills this year. My main challenge is "
                "time management - I often feel overwhelmed and struggle to prioritize. I also want "
                "to improve my visibility with senior leadership.",
        competency_tags=["time_management", "prioritization", "visibility"],
        sentiment=Sentiment.NEUTRAL
    ),
    FeedbackItem(
        id="fb-6",
        source=FeedbackSource.SKIP_LEVEL,
        author_role="Skip-level Manager",
        date="2024-10-30",
        content="Sarah's technical contributions are well-regarded. For her next level, she'll need "
                "to demonstrate more strategic thinking and cross-team influence. I'd recommend she "
                "take on a project with broader organizational impact.",
        competency_tags=["strategic_thinking", "influence", "career_growth"],
        sentiment=Sentiment.CONSTRUCTIVE
    ),
]


# ============================================================
# STATE DEFINITION
# ============================================================

class FeedbackSynthesisState(TypedDict):
    messages: Annotated[list, add_messages]
    feedback_items: list[dict]
    themes: list[dict]
    strengths: list[str]
    growth_areas: list[str]
    contradictions: list[str]
    synthesis: str
    coaching_questions: list[str]


# ============================================================
# SYNTHESIS NODES
# ============================================================

def load_feedback(state: FeedbackSynthesisState) -> dict:
    """Load and structure feedback data."""
    return {
        "feedback_items": [f.to_dict() for f in SAMPLE_FEEDBACK]
    }


def extract_themes(state: FeedbackSynthesisState) -> dict:
    """Extract recurring themes from feedback."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    feedback_text = json.dumps(state["feedback_items"], indent=2)
    
    prompt = f"""Analyze this feedback and extract key themes.

Feedback:
{feedback_text}

For each theme:
1. Name the theme (e.g., "Communication", "Technical Leadership")
2. List evidence from multiple sources
3. Note if sources agree or disagree
4. Rate theme strength (strong/moderate/emerging)

Return JSON:
{{
    "themes": [
        {{
            "name": "theme name",
            "evidence": ["quote/paraphrase 1", "quote/paraphrase 2"],
            "sources": ["source1", "source2"],
            "consensus": "agreement|mixed|contradiction",
            "strength": "strong|moderate|emerging",
            "valence": "strength|growth_area|both"
        }}
    ]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {"themes": parsed.get("themes", [])}
    except:
        return {"themes": []}


def identify_patterns(state: FeedbackSynthesisState) -> dict:
    """Identify strengths, growth areas, and contradictions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    feedback_text = json.dumps(state["feedback_items"], indent=2)
    themes_text = json.dumps(state.get("themes", []), indent=2)
    
    prompt = f"""Based on this feedback and themes, identify:

Feedback:
{feedback_text}

Themes:
{themes_text}

Analyze and return JSON:
{{
    "strengths": [
        "Clear strength with evidence - mention which sources agree"
    ],
    "growth_areas": [
        "Development area with evidence - be specific and constructive"
    ],
    "contradictions": [
        "Any contradictions between sources (e.g., 'Manager says X but peer says Y')"
    ],
    "blind_spots": [
        "Things self-assessment missed that others mentioned"
    ],
    "self_awareness_gaps": [
        "Gaps between self-perception and others' perceptions"
    ]
}}

Be specific and cite sources. Focus on actionable insights."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "strengths": parsed.get("strengths", []),
            "growth_areas": parsed.get("growth_areas", []),
            "contradictions": parsed.get("contradictions", []) + 
                             parsed.get("blind_spots", []) +
                             parsed.get("self_awareness_gaps", [])
        }
    except:
        return {"strengths": [], "growth_areas": [], "contradictions": []}


def generate_synthesis(state: FeedbackSynthesisState) -> dict:
    """Generate a coaching-oriented synthesis."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""Create a coaching-oriented feedback synthesis.

THEMES:
{json.dumps(state.get('themes', []), indent=2)}

STRENGTHS:
{json.dumps(state.get('strengths', []), indent=2)}

GROWTH AREAS:
{json.dumps(state.get('growth_areas', []), indent=2)}

CONTRADICTIONS/BLIND SPOTS:
{json.dumps(state.get('contradictions', []), indent=2)}

Write a synthesis that:
1. Leads with strengths (what's working well)
2. Frames growth areas constructively
3. Highlights patterns across sources
4. Notes any contradictions worth exploring
5. Suggests 2-3 development priorities

Tone: Warm, encouraging, but honest. Write as a coach would speak.
Length: 3-4 paragraphs."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"synthesis": response.content}


def generate_coaching_questions(state: FeedbackSynthesisState) -> dict:
    """Generate powerful coaching questions based on the synthesis."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""Based on this feedback synthesis, generate powerful coaching questions.

Synthesis:
{state.get('synthesis', '')}

Growth Areas:
{json.dumps(state.get('growth_areas', []), indent=2)}

Contradictions:
{json.dumps(state.get('contradictions', []), indent=2)}

Generate 5-7 coaching questions that:
1. Invite reflection (not yes/no answers)
2. Explore contradictions or blind spots
3. Connect strengths to growth opportunities
4. Consider underlying beliefs or assumptions
5. Point toward action

Return JSON:
{{
    "questions": [
        {{
            "question": "The actual question",
            "purpose": "Why this question matters",
            "theme": "Related theme/area"
        }}
    ]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {"coaching_questions": [q["question"] for q in parsed.get("questions", [])]}
    except:
        return {"coaching_questions": []}


def create_response(state: FeedbackSynthesisState) -> dict:
    """Create the final coaching response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    user_query = state["messages"][-1].content if state["messages"] else "Review my feedback"
    
    prompt = f"""You are an HR coach helping someone understand their feedback.

User's question: {user_query}

FEEDBACK SYNTHESIS:
{state.get('synthesis', '')}

KEY STRENGTHS:
{json.dumps(state.get('strengths', []), indent=2)}

GROWTH AREAS:
{json.dumps(state.get('growth_areas', []), indent=2)}

COACHING QUESTIONS TO EXPLORE:
{json.dumps(state.get('coaching_questions', []), indent=2)}

Respond to the user conversationally. If they asked a specific question, answer it.
If they want an overview, give them the synthesis and one powerful coaching question.
Be warm, supportive, and grounded in the actual feedback data."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [AIMessage(content=response.content)]}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_feedback_synthesis() -> StateGraph:
    """Build the feedback synthesis graph."""
    
    graph = StateGraph(FeedbackSynthesisState)
    
    # Add nodes in sequence
    graph.add_node("load_feedback", load_feedback)
    graph.add_node("extract_themes", extract_themes)
    graph.add_node("identify_patterns", identify_patterns)
    graph.add_node("generate_synthesis", generate_synthesis)
    graph.add_node("generate_questions", generate_coaching_questions)
    graph.add_node("create_response", create_response)
    
    # Linear flow
    graph.add_edge(START, "load_feedback")
    graph.add_edge("load_feedback", "extract_themes")
    graph.add_edge("extract_themes", "identify_patterns")
    graph.add_edge("identify_patterns", "generate_synthesis")
    graph.add_edge("generate_synthesis", "generate_questions")
    graph.add_edge("generate_questions", "create_response")
    graph.add_edge("create_response", END)
    
    return graph.compile()


# ============================================================
# INTERACTIVE COACHING SESSION
# ============================================================

def build_interactive_coach() -> StateGraph:
    """Build an interactive coach that can answer questions about feedback."""
    
    class InteractiveState(TypedDict):
        messages: Annotated[list, add_messages]
        synthesis_complete: bool
        synthesis: str
        strengths: list[str]
        growth_areas: list[str]
        themes: list[dict]
        coaching_questions: list[str]
    
    def check_synthesis(state: InteractiveState) -> dict:
        """Check if we have a synthesis or need to create one."""
        return {"synthesis_complete": bool(state.get("synthesis"))}
    
    def run_synthesis(state: InteractiveState) -> dict:
        """Run the full synthesis pipeline."""
        synth_graph = build_feedback_synthesis()
        result = synth_graph.invoke({
            "messages": state["messages"],
            "feedback_items": [],
            "themes": [],
            "strengths": [],
            "growth_areas": [],
            "contradictions": [],
            "synthesis": "",
            "coaching_questions": []
        })
        
        return {
            "synthesis": result.get("synthesis", ""),
            "strengths": result.get("strengths", []),
            "growth_areas": result.get("growth_areas", []),
            "themes": result.get("themes", []),
            "coaching_questions": result.get("coaching_questions", []),
            "synthesis_complete": True,
            "messages": result["messages"]
        }
    
    def answer_question(state: InteractiveState) -> dict:
        """Answer follow-up questions about the feedback."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        user_question = state["messages"][-1].content
        
        prompt = f"""You are an HR coach. The user has questions about their feedback.

THEIR FEEDBACK SYNTHESIS:
{state.get('synthesis', '')}

STRENGTHS: {json.dumps(state.get('strengths', []))}
GROWTH AREAS: {json.dumps(state.get('growth_areas', []))}

Their question: {user_question}

Answer helpfully, referencing their specific feedback.
If they seem stuck, offer a coaching question to help them reflect.
If they want to dig into something, help them explore it."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    
    def route_by_synthesis(state: InteractiveState) -> str:
        if state.get("synthesis_complete"):
            return "answer"
        return "synthesize"
    
    graph = StateGraph(InteractiveState)
    
    graph.add_node("check", check_synthesis)
    graph.add_node("synthesize", run_synthesis)
    graph.add_node("answer", answer_question)
    
    graph.add_edge(START, "check")
    graph.add_conditional_edges("check", route_by_synthesis, {
        "synthesize": "synthesize",
        "answer": "answer"
    })
    graph.add_edge("synthesize", END)
    graph.add_edge("answer", END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    print("=" * 60)
    print("FEEDBACK SYNTHESIS COACH")
    print("=" * 60)
    print("\nThis coach synthesizes feedback from multiple sources")
    print("and helps you understand patterns and growth opportunities.\n")
    
    # Run initial synthesis
    print("Analyzing your feedback from 6 sources...")
    print("(Manager, 2 Peers, Direct Report, Self, Skip-level)\n")
    
    coach = build_interactive_coach()
    
    # Session state persists across turns
    session_state = {
        "messages": [HumanMessage(content="Please review my feedback and help me understand the key themes.")],
        "synthesis_complete": False,
        "synthesis": "",
        "strengths": [],
        "growth_areas": [],
        "themes": [],
        "coaching_questions": []
    }
    
    result = coach.invoke(session_state)
    print(f"Coach: {result['messages'][-1].content}\n")
    
    # Update session state
    session_state.update(result)
    
    # Interactive loop
    print("-" * 40)
    print("Ask follow-up questions or type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nGood luck with your development! 🌱")
            break
        
        session_state["messages"] = [HumanMessage(content=user_input)]
        result = coach.invoke(session_state)
        print(f"\nCoach: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
