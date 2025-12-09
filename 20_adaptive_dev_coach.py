"""
20. Adaptive Development Coach
==============================
Advanced plan-and-execute coaching system with:
- Dynamic plan generation and iterative execution
- Adaptive user model that evolves with each interaction
- Multi-phase development journeys (90-day plans)
- Commitment tracking with follow-up workflows
- Self-reflection and strategy adaptation
- Personalized learning path generation
- Progress analytics and insights
- Proactive check-ins and nudges

This demonstrates sophisticated agentic patterns for
longitudinal coaching relationships.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# DEVELOPMENT JOURNEY MODEL
# ============================================================

class JourneyPhase(str, Enum):
    DISCOVERY = "discovery"           # Understanding current state
    GOAL_SETTING = "goal_setting"     # Defining objectives
    PLANNING = "planning"             # Creating action plan
    EXECUTION = "execution"           # Working the plan
    REFLECTION = "reflection"         # Reviewing progress
    ADAPTATION = "adaptation"         # Adjusting approach


class DevelopmentArea(str, Enum):
    TECHNICAL = "technical"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    STRATEGIC = "strategic"
    WELLBEING = "wellbeing"
    RELATIONSHIPS = "relationships"


@dataclass
class DevelopmentGoal:
    id: str
    title: str
    area: DevelopmentArea
    description: str
    success_criteria: list[str]
    milestones: list[dict]
    target_date: str
    current_progress: float = 0.0
    blockers: list[str] = field(default_factory=list)
    learnings: list[str] = field(default_factory=list)


@dataclass
class ActionPlan:
    id: str
    goal_id: str
    steps: list[dict]  # {description, status, due_date, dependencies}
    created_at: str
    last_updated: str
    completion_rate: float = 0.0


@dataclass
class Commitment:
    id: str
    description: str
    due_date: str
    goal_id: Optional[str]
    status: str  # pending, completed, missed, rescheduled
    check_in_dates: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass
class UserModel:
    """Dynamic model that evolves with each interaction."""
    
    # Core traits (stable)
    communication_style: str = "balanced"  # direct, supportive, analytical, balanced
    learning_style: str = "mixed"  # visual, reading, doing, discussing
    decision_style: str = "collaborative"  # autonomous, collaborative, guided
    energy_pattern: str = "steady"  # morning, afternoon, evening, steady
    
    # Dynamic factors (evolve)
    current_motivation: float = 0.7  # 0-1
    current_confidence: float = 0.6  # 0-1
    current_stress: float = 0.4  # 0-1
    engagement_trend: str = "stable"  # increasing, stable, decreasing
    
    # Observed patterns
    strengths_demonstrated: list[str] = field(default_factory=list)
    challenges_observed: list[str] = field(default_factory=list)
    successful_strategies: list[str] = field(default_factory=list)
    ineffective_strategies: list[str] = field(default_factory=list)
    
    # Preferences learned
    preferred_meeting_length: int = 20  # minutes
    preferred_check_in_frequency: str = "weekly"
    topics_of_interest: list[str] = field(default_factory=list)
    topics_to_avoid: list[str] = field(default_factory=list)
    
    # Coaching relationship
    sessions_completed: int = 0
    total_commitments: int = 0
    commitments_completed: int = 0
    trust_level: float = 0.5  # 0-1, builds over time
    
    def commitment_completion_rate(self) -> float:
        if self.total_commitments == 0:
            return 0.0
        return self.commitments_completed / self.total_commitments
    
    def to_dict(self) -> dict:
        return {
            "communication_style": self.communication_style,
            "learning_style": self.learning_style,
            "decision_style": self.decision_style,
            "current_motivation": self.current_motivation,
            "current_confidence": self.current_confidence,
            "current_stress": self.current_stress,
            "engagement_trend": self.engagement_trend,
            "strengths": self.strengths_demonstrated[-5:],
            "challenges": self.challenges_observed[-5:],
            "successful_strategies": self.successful_strategies[-5:],
            "sessions_completed": self.sessions_completed,
            "commitment_rate": f"{self.commitment_completion_rate():.0%}",
            "trust_level": self.trust_level
        }


# ============================================================
# IN-MEMORY STORES
# ============================================================

class DevelopmentStore:
    """Stores development journey data."""
    
    def __init__(self):
        self.goals: dict[str, DevelopmentGoal] = {}
        self.plans: dict[str, ActionPlan] = {}
        self.commitments: dict[str, Commitment] = {}
        self.user_model = UserModel()
        self.session_history: list[dict] = []
        self.insights: list[dict] = []
        self._seed_data()
    
    def _seed_data(self):
        """Seed with sample development journey."""
        
        # Sample goals
        goal1 = DevelopmentGoal(
            id="goal-1",
            title="Develop technical leadership",
            area=DevelopmentArea.LEADERSHIP,
            description="Grow from individual contributor to technical leader who influences architecture and mentors others",
            success_criteria=[
                "Lead design review for 2 major projects",
                "Mentor 2 junior engineers through a full project cycle",
                "Establish and document team coding standards"
            ],
            milestones=[
                {"name": "Shadow senior lead on design review", "target_date": "2025-01-15", "status": "completed"},
                {"name": "Lead first design review with support", "target_date": "2025-02-15", "status": "in_progress"},
                {"name": "Lead design review independently", "target_date": "2025-03-15", "status": "not_started"},
                {"name": "Begin mentoring relationship", "target_date": "2025-01-30", "status": "in_progress"}
            ],
            target_date="2025-06-30",
            current_progress=0.35,
            blockers=["Feeling imposter syndrome about leading reviews"],
            learnings=["Preparation is key - spent 2x time prepping first review"]
        )
        
        goal2 = DevelopmentGoal(
            id="goal-2",
            title="Improve stakeholder communication",
            area=DevelopmentArea.COMMUNICATION,
            description="Communicate more effectively with non-technical stakeholders, especially product and business",
            success_criteria=[
                "Regular 1:1s with key product partners",
                "Lead 3 cross-functional presentations",
                "Reduce technical jargon in written comms"
            ],
            milestones=[
                {"name": "Identify key stakeholders", "target_date": "2025-01-10", "status": "completed"},
                {"name": "Schedule regular syncs", "target_date": "2025-01-20", "status": "completed"},
                {"name": "First cross-functional presentation", "target_date": "2025-02-28", "status": "not_started"}
            ],
            target_date="2025-04-30",
            current_progress=0.25
        )
        
        self.goals["goal-1"] = goal1
        self.goals["goal-2"] = goal2
        
        # Sample action plan
        plan1 = ActionPlan(
            id="plan-1",
            goal_id="goal-1",
            steps=[
                {"description": "Review 3 past design docs as examples", "status": "completed", "due_date": "2025-01-10"},
                {"description": "Create design review checklist", "status": "completed", "due_date": "2025-01-15"},
                {"description": "Shadow Maria's next design review", "status": "completed", "due_date": "2025-01-20"},
                {"description": "Prepare presentation for auth service redesign", "status": "in_progress", "due_date": "2025-02-10"},
                {"description": "Dry run presentation with mentor", "status": "not_started", "due_date": "2025-02-12"},
                {"description": "Lead design review for auth service", "status": "not_started", "due_date": "2025-02-15"},
                {"description": "Collect feedback and reflect", "status": "not_started", "due_date": "2025-02-20"}
            ],
            created_at="2025-01-05",
            last_updated="2025-01-25",
            completion_rate=0.43
        )
        self.plans["plan-1"] = plan1
        
        # Sample commitments
        commitment1 = Commitment(
            id="commit-1",
            description="Complete auth service design presentation draft",
            due_date="2025-02-08",
            goal_id="goal-1",
            status="pending",
            check_in_dates=["2025-02-05"],
            notes=["Started outline, need to add performance considerations"]
        )
        self.commitments["commit-1"] = commitment1
        
        # Evolve user model based on "history"
        self.user_model.sessions_completed = 6
        self.user_model.total_commitments = 8
        self.user_model.commitments_completed = 6
        self.user_model.trust_level = 0.7
        self.user_model.current_motivation = 0.75
        self.user_model.current_confidence = 0.6
        self.user_model.strengths_demonstrated = [
            "Strong technical foundation",
            "Thorough preparation",
            "Receptive to feedback",
            "Consistent follow-through"
        ]
        self.user_model.challenges_observed = [
            "Imposter syndrome around leadership",
            "Tendency to over-prepare",
            "Difficulty delegating"
        ]
        self.user_model.successful_strategies = [
            "Breaking large tasks into small milestones",
            "Having a prep checklist",
            "Practicing with trusted peers first"
        ]
        
        # Sample insights
        self.insights = [
            {
                "date": "2025-01-20",
                "insight": "User shows strong progress when given structured frameworks",
                "evidence": "Completed all checklist-based tasks, struggled with open-ended ones"
            },
            {
                "date": "2025-01-25",
                "insight": "Confidence increases significantly after successful experiences",
                "evidence": "Motivation jumped after positive feedback on shadowing session"
            }
        ]
        
        # Session history
        self.session_history = [
            {
                "date": "2025-01-05",
                "phase": "discovery",
                "summary": "Initial discovery session. Identified key development goals.",
                "sentiment": "motivated"
            },
            {
                "date": "2025-01-12",
                "phase": "planning",
                "summary": "Created action plan for technical leadership goal.",
                "sentiment": "focused"
            },
            {
                "date": "2025-01-19",
                "phase": "execution",
                "summary": "Check-in on progress. Completed shadowing, feeling more confident.",
                "sentiment": "positive"
            },
            {
                "date": "2025-01-26",
                "phase": "execution",
                "summary": "Discussed imposter syndrome. Normalized feelings, identified coping strategies.",
                "sentiment": "supported"
            }
        ]


STORE = DevelopmentStore()


# ============================================================
# STATE DEFINITION
# ============================================================

class DevCoachState(TypedDict):
    messages: Annotated[list, add_messages]
    
    # Journey context
    current_phase: str
    active_goals: list[dict]
    active_plans: list[dict]
    pending_commitments: list[dict]
    
    # User understanding
    user_model: dict
    recent_insights: list[dict]
    
    # Planning
    generated_plan: dict
    plan_steps: list[dict]
    current_step: int
    
    # Adaptation
    detected_state: dict  # motivation, confidence, stress, engagement
    strategy_adjustments: list[str]
    
    # Response
    coaching_approach: str
    response_draft: str
    follow_up_actions: list[dict]
    
    # Learning
    new_insights: list[dict]
    model_updates: dict


# ============================================================
# PLAN-AND-EXECUTE PATTERN
# ============================================================

def assess_user_state(state: DevCoachState) -> dict:
    """Assess user's current emotional and motivational state."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    last_message = state["messages"][-1].content
    user_model = state.get("user_model", STORE.user_model.to_dict())
    session_history = STORE.session_history[-3:]
    
    prompt = f"""Assess the user's current state based on their message and history.

USER MODEL:
{json.dumps(user_model, indent=2)}

RECENT SESSION HISTORY:
{json.dumps(session_history, indent=2)}

CURRENT MESSAGE: {last_message}

Analyze and return JSON:
{{
    "motivation_level": 0.0-1.0,
    "confidence_level": 0.0-1.0,
    "stress_level": 0.0-1.0,
    "engagement_quality": "high|medium|low",
    "emotional_tone": "description of emotional state",
    "signals": ["specific signals detected"],
    "needs_detected": ["what they seem to need right now"],
    "recommended_approach": "supportive|challenging|exploratory|practical|celebratory"
}}

Consider:
- Changes from their baseline (user model)
- Explicit and implicit signals in their message
- Pattern from recent sessions"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "detected_state": parsed,
            "coaching_approach": parsed.get("recommended_approach", "supportive")
        }
    except:
        return {
            "detected_state": {"motivation_level": 0.7, "confidence_level": 0.6},
            "coaching_approach": "supportive"
        }


def determine_phase_and_intent(state: DevCoachState) -> dict:
    """Determine where we are in the coaching journey and what's needed."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    last_message = state["messages"][-1].content
    goals = [g.__dict__ if hasattr(g, '__dict__') else g for g in STORE.goals.values()]
    plans = [p.__dict__ if hasattr(p, '__dict__') else p for p in STORE.plans.values()]
    commitments = [c.__dict__ if hasattr(c, '__dict__') else c for c in STORE.commitments.values() if c.status == "pending"]
    
    prompt = f"""Determine the appropriate coaching phase and intent.

ACTIVE GOALS:
{json.dumps([{"title": g.title, "progress": g.current_progress, "blockers": g.blockers} for g in STORE.goals.values()], indent=2)}

ACTIVE PLANS:
{json.dumps([{"goal": p.goal_id, "completion": p.completion_rate, "next_steps": [s for s in p.steps if s["status"] != "completed"][:2]} for p in STORE.plans.values()], indent=2)}

PENDING COMMITMENTS:
{json.dumps([{"description": c.description, "due": c.due_date} for c in STORE.commitments.values() if c.status == "pending"], indent=2)}

USER MESSAGE: {last_message}

Determine:
{{
    "phase": "discovery|goal_setting|planning|execution|reflection|adaptation",
    "intent": "what they're asking for or need",
    "should_check_commitments": true/false,
    "should_review_progress": true/false,
    "should_create_plan": true/false,
    "should_adapt_approach": true/false,
    "primary_goal_focus": "goal-id or null",
    "reasoning": "why this phase/intent"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "current_phase": parsed.get("phase", "execution"),
            "active_goals": list(STORE.goals.values()),
            "active_plans": list(STORE.plans.values()),
            "pending_commitments": list(STORE.commitments.values())
        }
    except:
        return {"current_phase": "execution"}


def generate_coaching_plan(state: DevCoachState) -> dict:
    """Generate a plan for this coaching interaction."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    phase = state.get("current_phase", "execution")
    approach = state.get("coaching_approach", "supportive")
    detected_state = state.get("detected_state", {})
    user_model = STORE.user_model.to_dict()
    
    prompt = f"""Create a coaching plan for this interaction.

CURRENT PHASE: {phase}
RECOMMENDED APPROACH: {approach}
USER STATE: {json.dumps(detected_state, indent=2)}
USER MODEL: {json.dumps(user_model, indent=2)}

USER'S MESSAGE: {state["messages"][-1].content}

Generate a coaching plan:
{{
    "interaction_goal": "what we want to achieve in this interaction",
    "steps": [
        {{
            "action": "what to do",
            "purpose": "why this step",
            "technique": "coaching technique to use"
        }}
    ],
    "key_questions": ["powerful questions to ask"],
    "potential_pivots": ["if they respond X, pivot to Y"],
    "success_criteria": "how we know this interaction was successful",
    "avoid": ["things to avoid based on user model"]
}}

Consider:
- User's successful strategies: {user_model.get('successful_strategies', [])}
- Known challenges: {user_model.get('challenges', [])}
- Current confidence: {detected_state.get('confidence_level', 0.6)}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "generated_plan": parsed,
            "plan_steps": parsed.get("steps", []),
            "current_step": 0
        }
    except:
        return {"generated_plan": {}, "plan_steps": [], "current_step": 0}


def execute_coaching_step(state: DevCoachState) -> dict:
    """Execute the coaching plan and generate response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    plan = state.get("generated_plan", {})
    approach = state.get("coaching_approach", "supportive")
    detected_state = state.get("detected_state", {})
    phase = state.get("current_phase", "execution")
    user_model = STORE.user_model.to_dict()
    
    # Get relevant context based on phase
    context_additions = ""
    
    if phase in ["execution", "reflection"]:
        # Include progress data
        goals_progress = []
        for g in STORE.goals.values():
            goals_progress.append({
                "title": g.title,
                "progress": f"{g.current_progress:.0%}",
                "blockers": g.blockers,
                "learnings": g.learnings
            })
        context_additions += f"\nGOALS PROGRESS:\n{json.dumps(goals_progress, indent=2)}"
        
        # Include pending commitments
        pending = [c for c in STORE.commitments.values() if c.status == "pending"]
        if pending:
            context_additions += f"\n\nPENDING COMMITMENTS:\n"
            for c in pending:
                context_additions += f"- {c.description} (due: {c.due_date})\n"
    
    if phase == "planning":
        # Include plan status
        for p in STORE.plans.values():
            next_steps = [s for s in p.steps if s["status"] != "completed"][:3]
            context_additions += f"\nNEXT STEPS IN PLAN:\n{json.dumps(next_steps, indent=2)}"
    
    # Build the prompt
    prompt = f"""You are an adaptive development coach. Generate your response following this plan.

COACHING PLAN:
{json.dumps(plan, indent=2)}

USER MODEL (how they work best):
- Communication style: {user_model.get('communication_style')}
- Decision style: {user_model.get('decision_style')}
- Trust level: {user_model.get('trust_level')}
- Commitment completion rate: {user_model.get('commitment_rate')}

USER'S CURRENT STATE:
- Motivation: {detected_state.get('motivation_level', 0.7)}
- Confidence: {detected_state.get('confidence_level', 0.6)}
- Stress: {detected_state.get('stress_level', 0.4)}
- Emotional tone: {detected_state.get('emotional_tone', 'neutral')}

APPROACH: {approach}
{context_additions}

USER'S MESSAGE: {state["messages"][-1].content}

Generate your coaching response:
1. Match their energy and emotional state
2. Follow the planned steps but adapt naturally
3. Use questions from the plan strategically
4. Reference their progress and patterns where relevant
5. End with a clear next step or reflection prompt
6. If they've completed something, celebrate appropriately

Remember their successful strategies: {user_model.get('successful_strategies', [])}
Be mindful of their challenges: {user_model.get('challenges', [])}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"response_draft": response.content}


def extract_follow_ups(state: DevCoachState) -> dict:
    """Extract follow-up actions and commitments from the conversation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    user_message = state["messages"][-1].content
    response = state.get("response_draft", "")
    
    prompt = f"""Extract follow-up items from this coaching exchange.

USER: {user_message}
COACH: {response}

Return JSON:
{{
    "commitments": [
        {{
            "description": "what they committed to",
            "due_date": "YYYY-MM-DD or null",
            "goal_id": "related goal or null",
            "check_in_date": "when to follow up"
        }}
    ],
    "insights_about_user": [
        {{
            "insight": "what we learned about them",
            "evidence": "what indicated this",
            "category": "strength|challenge|preference|strategy"
        }}
    ],
    "progress_updates": [
        {{
            "goal_id": "which goal",
            "update_type": "milestone_completed|blocker_added|blocker_resolved|learning_added",
            "details": "specifics"
        }}
    ],
    "model_updates": {{
        "motivation_change": -0.1 to 0.1,
        "confidence_change": -0.1 to 0.1,
        "stress_change": -0.1 to 0.1,
        "new_strength": "or null",
        "new_challenge": "or null",
        "new_successful_strategy": "or null"
    }},
    "next_session_topics": ["topics to address next time"]
}}

Only include items explicitly mentioned or strongly implied."""

    response_llm = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response_llm.content)
        
        # Create commitments
        for c in parsed.get("commitments", []):
            if c.get("description"):
                commitment = Commitment(
                    id=str(uuid.uuid4()),
                    description=c["description"],
                    due_date=c.get("due_date", ""),
                    goal_id=c.get("goal_id"),
                    status="pending",
                    check_in_dates=[c.get("check_in_date")] if c.get("check_in_date") else []
                )
                STORE.commitments[commitment.id] = commitment
                STORE.user_model.total_commitments += 1
        
        # Update user model
        model_updates = parsed.get("model_updates", {})
        if model_updates.get("motivation_change"):
            STORE.user_model.current_motivation = max(0, min(1, 
                STORE.user_model.current_motivation + model_updates["motivation_change"]))
        if model_updates.get("confidence_change"):
            STORE.user_model.current_confidence = max(0, min(1,
                STORE.user_model.current_confidence + model_updates["confidence_change"]))
        if model_updates.get("new_strength"):
            STORE.user_model.strengths_demonstrated.append(model_updates["new_strength"])
        if model_updates.get("new_successful_strategy"):
            STORE.user_model.successful_strategies.append(model_updates["new_successful_strategy"])
        
        # Add insights
        for insight in parsed.get("insights_about_user", []):
            if insight.get("insight"):
                STORE.insights.append({
                    "date": datetime.now().isoformat(),
                    "insight": insight["insight"],
                    "evidence": insight.get("evidence", "")
                })
        
        # Update session count
        STORE.user_model.sessions_completed += 1
        
        return {
            "follow_up_actions": parsed.get("commitments", []),
            "new_insights": parsed.get("insights_about_user", []),
            "model_updates": model_updates
        }
    except:
        return {"follow_up_actions": [], "new_insights": [], "model_updates": {}}


def reflect_and_adapt(state: DevCoachState) -> dict:
    """Reflect on the interaction and adapt strategies."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    plan = state.get("generated_plan", {})
    response = state.get("response_draft", "")
    detected_state = state.get("detected_state", {})
    new_insights = state.get("new_insights", [])
    
    prompt = f"""Reflect on this coaching interaction and identify adaptations.

ORIGINAL PLAN:
{json.dumps(plan, indent=2)}

RESPONSE GIVEN:
{response[:500]}...

USER STATE DETECTED:
{json.dumps(detected_state, indent=2)}

NEW INSIGHTS:
{json.dumps(new_insights, indent=2)}

Analyze:
{{
    "plan_adherence": "how well did we follow the plan (high/medium/low)",
    "approach_effectiveness": "assessment of chosen approach",
    "what_worked": ["elements that seemed effective"],
    "what_to_improve": ["what could be better next time"],
    "strategy_adjustments": ["adjustments for future sessions"],
    "user_model_refinements": ["how to update our understanding"],
    "relationship_development": "how the coaching relationship evolved"
}}"""

    response_llm = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response_llm.content)
        return {"strategy_adjustments": parsed.get("strategy_adjustments", [])}
    except:
        return {"strategy_adjustments": []}


def finalize_response(state: DevCoachState) -> dict:
    """Finalize and enhance the response."""
    response = state.get("response_draft", "")
    follow_ups = state.get("follow_up_actions", [])
    
    # Add commitment tracking if any were made
    if follow_ups:
        response += "\n\n **I've noted these commitments:**\n"
        for f in follow_ups:
            due = f" (by {f['due_date']})" if f.get('due_date') else ""
            response += f"• {f['description']}{due}\n"
    
    # Add session context
    STORE.session_history.append({
        "date": datetime.now().isoformat(),
        "phase": state.get("current_phase", "execution"),
        "summary": f"Discussed: {state.get('generated_plan', {}).get('interaction_goal', 'coaching session')}",
        "sentiment": state.get("detected_state", {}).get("emotional_tone", "neutral")
    })
    
    return {"messages": [AIMessage(content=response)]}


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_dev_coach() -> StateGraph:
    """Build the adaptive development coach graph."""
    
    graph = StateGraph(DevCoachState)
    
    # Add nodes
    graph.add_node("assess_state", assess_user_state)
    graph.add_node("determine_phase", determine_phase_and_intent)
    graph.add_node("generate_plan", generate_coaching_plan)
    graph.add_node("execute", execute_coaching_step)
    graph.add_node("extract_follow_ups", extract_follow_ups)
    graph.add_node("reflect_adapt", reflect_and_adapt)
    graph.add_node("finalize", finalize_response)
    
    # Flow
    graph.add_edge(START, "assess_state")
    graph.add_edge("assess_state", "determine_phase")
    graph.add_edge("determine_phase", "generate_plan")
    graph.add_edge("generate_plan", "execute")
    graph.add_edge("execute", "extract_follow_ups")
    graph.add_edge("extract_follow_ups", "reflect_adapt")
    graph.add_edge("reflect_adapt", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ============================================================
# PROGRESS DASHBOARD
# ============================================================

def show_dashboard():
    """Display development progress dashboard."""
    print("\n" + "=" * 60)
    print("📊 DEVELOPMENT DASHBOARD")
    print("=" * 60)
    
    # Goals progress
    print("\n🎯 GOALS:")
    for goal in STORE.goals.values():
        bar_filled = int(goal.current_progress * 20)
        bar_empty = 20 - bar_filled
        progress_bar = "█" * bar_filled + "░" * bar_empty
        print(f"  {goal.title}")
        print(f"    [{progress_bar}] {goal.current_progress:.0%}")
        if goal.blockers:
            print(f"      Blockers: {', '.join(goal.blockers)}")
    
    # Pending commitments
    pending = [c for c in STORE.commitments.values() if c.status == "pending"]
    if pending:
        print("\n PENDING COMMITMENTS:")
        for c in pending:
            print(f"  • {c.description}")
            if c.due_date:
                print(f"    Due: {c.due_date}")
    
    # User model summary
    model = STORE.user_model
    print("\n📈 YOUR PATTERNS:")
    print(f"  Motivation: {'▓' * int(model.current_motivation * 10)}{'░' * (10 - int(model.current_motivation * 10))} {model.current_motivation:.0%}")
    print(f"  Confidence: {'▓' * int(model.current_confidence * 10)}{'░' * (10 - int(model.current_confidence * 10))} {model.current_confidence:.0%}")
    print(f"  Commitment rate: {model.commitment_completion_rate():.0%}")
    print(f"  Sessions: {model.sessions_completed}")
    
    # Recent insights
    if STORE.insights:
        print("\n RECENT INSIGHTS:")
        for insight in STORE.insights[-2:]:
            print(f"  • {insight['insight']}")
    
    print("=" * 60)
    print()


def show_journey():
    """Show the development journey timeline."""
    print("\n" + "=" * 60)
    print("🛤️  DEVELOPMENT JOURNEY")
    print("=" * 60)
    
    for session in STORE.session_history:
        date = session.get("date", "")[:10]
        phase = session.get("phase", "").upper()
        summary = session.get("summary", "")
        sentiment = session.get("sentiment", "")
        
        emoji = {"discovery": "", "planning": "", "execution": "🚀", "reflection": "🪞"}.get(session.get("phase", ""), "")
        print(f"\n{emoji} [{date}] {phase}")
        print(f"   {summary}")
        print(f"   Sentiment: {sentiment}")
    
    print("\n" + "=" * 60)
    print()


# ============================================================
# DEMO
# ============================================================

def main():
    coach = build_dev_coach()
    
    print("=" * 60)
    print("ADAPTIVE DEVELOPMENT COACH")
    print("=" * 60)
    print("\nAn intelligent coach that adapts to you over time.")
    print("\nFeatures:")
    print("  • Dynamic user model that evolves")
    print("  • Plan-and-execute coaching approach")
    print("  • Progress tracking and commitments")
    print("  • Personalized strategies based on your patterns")
    print("\nYou're working on:")
    for g in STORE.goals.values():
        print(f"  • {g.title} ({g.current_progress:.0%})")
    print("\nCommands: 'dashboard', 'journey', 'model', 'quit'\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nKeep pushing toward your goals! 🚀")
            break
        
        if user_input.lower() == 'dashboard':
            show_dashboard()
            continue
        
        if user_input.lower() == 'journey':
            show_journey()
            continue
        
        if user_input.lower() == 'model':
            print("\n YOUR USER MODEL:")
            print(json.dumps(STORE.user_model.to_dict(), indent=2))
            print()
            continue
        
        # Process coaching request
        state = {
            "messages": [HumanMessage(content=user_input)],
            "current_phase": "",
            "active_goals": [],
            "active_plans": [],
            "pending_commitments": [],
            "user_model": STORE.user_model.to_dict(),
            "recent_insights": STORE.insights[-5:],
            "generated_plan": {},
            "plan_steps": [],
            "current_step": 0,
            "detected_state": {},
            "strategy_adjustments": [],
            "coaching_approach": "",
            "response_draft": "",
            "follow_up_actions": [],
            "new_insights": [],
            "model_updates": {}
        }
        
        print("\n Assessing your state...")
        result = coach.invoke(state)
        
        # Show internal process (for demo)
        if result.get("detected_state"):
            state_info = result["detected_state"]
            print(f"   State: {state_info.get('emotional_tone', 'neutral')} | Approach: {result.get('coaching_approach', 'supportive')}")
        
        if result.get("current_phase"):
            print(f"   Phase: {result['current_phase']}")
        
        print()
        response = result["messages"][-1].content
        print(f"Coach: {response}\n")
        
        # Show any model updates
        if result.get("model_updates"):
            updates = result["model_updates"]
            if updates.get("confidence_change") and updates["confidence_change"] > 0:
                print("   📈 Your confidence is growing!")
            if updates.get("new_successful_strategy"):
                print(f"    New strategy noted: {updates['new_successful_strategy']}")
            print()


if __name__ == "__main__":
    main()
