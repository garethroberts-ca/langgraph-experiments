"""
14. Goal Lifecycle Coach
========================
Manages the complete goal lifecycle: creation, refinement, decomposition,
progress tracking, and completion. Implements OKR-style structure.

Key concepts:
- Goal decomposition into key results and actions
- Progress tracking with evidence
- Conflict detection between goals
- Timeline and priority management
"""

from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# GOAL DATA MODEL
# ============================================================

class GoalStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalCategory(str, Enum):
    PERFORMANCE = "performance"
    SKILL = "skill"
    CAREER = "career"
    WELLBEING = "wellbeing"
    RELATIONSHIP = "relationship"


@dataclass
class KeyResult:
    id: str
    description: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""
    due_date: Optional[str] = None
    
    @property
    def progress(self) -> float:
        if self.target_value == 0:
            return 0.0
        return min(self.current_value / self.target_value, 1.0)


@dataclass
class Goal:
    id: str
    title: str
    description: str
    category: GoalCategory
    status: GoalStatus = GoalStatus.DRAFT
    key_results: list[KeyResult] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    due_date: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    blockers: list[str] = field(default_factory=list)
    
    @property
    def progress(self) -> float:
        if not self.key_results:
            return 0.0
        return sum(kr.progress for kr in self.key_results) / len(self.key_results)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "progress": f"{self.progress:.0%}",
            "key_results": [
                {
                    "description": kr.description,
                    "progress": f"{kr.progress:.0%}",
                    "current": kr.current_value,
                    "target": kr.target_value,
                    "unit": kr.unit
                }
                for kr in self.key_results
            ],
            "actions": self.actions,
            "due_date": self.due_date,
            "blockers": self.blockers
        }


# ============================================================
# STATE DEFINITION
# ============================================================

class GoalCoachState(TypedDict):
    messages: Annotated[list, add_messages]
    goals: list[dict]  # Serialized Goal objects
    current_goal_id: Optional[str]
    intent: str  # create, refine, update_progress, review, decompose
    proposed_changes: dict
    conflicts: list[str]


# ============================================================
# GOAL STORE (In-memory for demo)
# ============================================================

class GoalStore:
    """Simple in-memory goal storage."""
    
    def __init__(self):
        self.goals: dict[str, Goal] = {}
        self._seed_sample_goals()
    
    def _seed_sample_goals(self):
        """Add sample goals for demonstration."""
        goal1 = Goal(
            id="goal-1",
            title="Improve presentation skills",
            description="Become more confident and effective in presenting to large groups",
            category=GoalCategory.SKILL,
            status=GoalStatus.ACTIVE,
            key_results=[
                KeyResult(
                    id="kr-1",
                    description="Deliver 5 presentations to groups of 10+",
                    target_value=5,
                    current_value=2,
                    unit="presentations"
                ),
                KeyResult(
                    id="kr-2", 
                    description="Average feedback score of 4+/5",
                    target_value=4.0,
                    current_value=3.5,
                    unit="score"
                )
            ],
            actions=["Join Toastmasters", "Practice with smaller groups first"],
            due_date="2025-03-31"
        )
        
        goal2 = Goal(
            id="goal-2",
            title="Build stronger stakeholder relationships",
            description="Develop better working relationships with key stakeholders",
            category=GoalCategory.RELATIONSHIP,
            status=GoalStatus.ACTIVE,
            key_results=[
                KeyResult(
                    id="kr-3",
                    description="Monthly 1:1s with 3 key stakeholders",
                    target_value=9,
                    current_value=4,
                    unit="meetings"
                )
            ],
            due_date="2025-03-31"
        )
        
        self.goals[goal1.id] = goal1
        self.goals[goal2.id] = goal2
    
    def get_all(self) -> list[Goal]:
        return list(self.goals.values())
    
    def get(self, goal_id: str) -> Optional[Goal]:
        return self.goals.get(goal_id)
    
    def save(self, goal: Goal):
        self.goals[goal.id] = goal
    
    def delete(self, goal_id: str):
        self.goals.pop(goal_id, None)


GOAL_STORE = GoalStore()


# ============================================================
# NODE FUNCTIONS
# ============================================================

def classify_intent(state: GoalCoachState) -> dict:
    """Classify user intent regarding goals."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    last_message = state["messages"][-1].content
    goals_summary = json.dumps([g.to_dict() for g in GOAL_STORE.get_all()], indent=2)
    
    prompt = f"""Classify the user's intent regarding their goals.

Current goals:
{goals_summary}

User message: {last_message}

Respond with JSON only:
{{
    "intent": "create|refine|update_progress|review|decompose|check_conflicts|general",
    "goal_id": "goal-id or null if creating new or general",
    "details": "brief extraction of relevant details"
}}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
        return {
            "intent": parsed.get("intent", "general"),
            "current_goal_id": parsed.get("goal_id")
        }
    except:
        return {"intent": "general", "current_goal_id": None}


def handle_create(state: GoalCoachState) -> dict:
    """Help user create a new goal with OKR structure."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    last_message = state["messages"][-1].content
    existing_goals = [g.to_dict() for g in GOAL_STORE.get_all()]
    
    prompt = f"""You are a goal-setting coach. Help the user create a well-structured goal.

Existing goals (check for potential conflicts or overlaps):
{json.dumps(existing_goals, indent=2)}

User's input: {last_message}

Guide the user to create a SMART goal with:
1. Clear, specific title
2. Meaningful description (why this matters)
3. 2-3 measurable Key Results
4. Concrete next actions
5. Realistic timeline

If the user's input is vague, ask clarifying questions.
If you have enough info, propose a structured goal and ask for confirmation.

Format proposed goals as:
**Goal**: [title]
**Category**: [performance/skill/career/wellbeing/relationship]
**Description**: [why this matters]
**Key Results**:
- KR1: [measurable outcome] (target: X)
- KR2: [measurable outcome] (target: Y)
**Next Actions**:
- Action 1
- Action 2
**Due Date**: [date]

Be encouraging but push for specificity."""

    response = llm.invoke([
        SystemMessage(content="You are an expert goal-setting coach."),
        HumanMessage(content=prompt)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}


def handle_review(state: GoalCoachState) -> dict:
    """Review progress on goals and provide coaching."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    goals = GOAL_STORE.get_all()
    goals_detail = json.dumps([g.to_dict() for g in goals], indent=2)
    
    prompt = f"""You are reviewing the user's goals as their coach.

Current goals and progress:
{goals_detail}

Provide a thoughtful progress review:
1. Celebrate wins and progress made
2. Identify goals that need attention (low progress, approaching deadlines)
3. Ask about blockers or challenges
4. Suggest adjustments if goals seem unrealistic
5. Offer one specific coaching question to deepen reflection

Be supportive but honest. Use their actual progress data."""

    response = llm.invoke([
        SystemMessage(content="You are an encouraging but honest goal coach."),
        HumanMessage(content=prompt)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}


def handle_update_progress(state: GoalCoachState) -> dict:
    """Handle progress updates on specific goals."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    goal_id = state.get("current_goal_id")
    goal = GOAL_STORE.get(goal_id) if goal_id else None
    last_message = state["messages"][-1].content
    
    if goal:
        goal_info = json.dumps(goal.to_dict(), indent=2)
        prompt = f"""The user wants to update progress on this goal:
{goal_info}

Their message: {last_message}

Help them:
1. Record specific progress on key results
2. Celebrate the progress made
3. Identify what's working
4. Plan the next step

If they mention completing something, acknowledge it enthusiastically.
Ask what enabled their progress (to reinforce good behaviors)."""
    else:
        goals = [g.to_dict() for g in GOAL_STORE.get_all()]
        prompt = f"""The user wants to update progress but we need to clarify which goal.

Available goals:
{json.dumps(goals, indent=2)}

Their message: {last_message}

Ask which goal they're updating and what progress they've made."""

    response = llm.invoke([
        SystemMessage(content="You are a supportive goal coach tracking progress."),
        HumanMessage(content=prompt)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}


def handle_decompose(state: GoalCoachState) -> dict:
    """Help break down a goal into smaller steps."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    goal_id = state.get("current_goal_id")
    goal = GOAL_STORE.get(goal_id) if goal_id else None
    
    if goal:
        prompt = f"""Help the user decompose this goal into manageable steps:

Goal: {goal.title}
Description: {goal.description}
Current Key Results: {json.dumps([kr.__dict__ for kr in goal.key_results], indent=2)}
Current Actions: {goal.actions}

Suggest:
1. Weekly milestones for the next month
2. Daily/weekly habits that support this goal
3. Potential obstacles and how to prepare for them
4. Quick wins they could achieve this week
5. How to make progress visible (tracking, accountability)

Make it actionable and not overwhelming."""
    else:
        prompt = "Ask the user which goal they'd like to break down into smaller steps."

    response = llm.invoke([
        SystemMessage(content="You help break big goals into achievable steps."),
        HumanMessage(content=prompt)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}


def check_conflicts(state: GoalCoachState) -> dict:
    """Analyze potential conflicts between goals."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    goals = [g.to_dict() for g in GOAL_STORE.get_all()]
    
    prompt = f"""Analyze these goals for potential conflicts or tensions:

Goals:
{json.dumps(goals, indent=2)}

Look for:
1. Time conflicts (too many goals with same deadline)
2. Resource conflicts (goals competing for same time/energy)
3. Philosophical tensions (goals that might work against each other)
4. Unrealistic total commitment

If conflicts exist, suggest:
- Which goals to prioritize
- How to sequence goals
- How to modify goals to reduce conflict

Be specific and practical."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "conflicts": []  # Could parse and store structured conflicts
    }


def handle_general(state: GoalCoachState) -> dict:
    """Handle general goal-related conversation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    goals = [g.to_dict() for g in GOAL_STORE.get_all()]
    last_message = state["messages"][-1].content
    
    prompt = f"""You are a goal coaching assistant. The user's goals:
{json.dumps(goals, indent=2)}

User message: {last_message}

Respond helpfully. You can:
- Answer questions about goal-setting
- Offer motivation and encouragement
- Suggest they create, review, or update goals
- Provide tips on goal achievement

Keep it conversational and supportive."""

    response = llm.invoke([
        SystemMessage(content="You are a helpful goal coach."),
        HumanMessage(content=prompt)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}


def route_by_intent(state: GoalCoachState) -> str:
    """Route to appropriate handler based on intent."""
    intent = state.get("intent", "general")
    
    routes = {
        "create": "create_goal",
        "refine": "create_goal",  # Similar flow
        "update_progress": "update_progress",
        "review": "review_goals",
        "decompose": "decompose_goal",
        "check_conflicts": "check_conflicts",
        "general": "general"
    }
    
    return routes.get(intent, "general")


# ============================================================
# BUILD GRAPH
# ============================================================

def build_goal_coach() -> StateGraph:
    """Build the goal lifecycle coaching graph."""
    
    graph = StateGraph(GoalCoachState)
    
    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("create_goal", handle_create)
    graph.add_node("review_goals", handle_review)
    graph.add_node("update_progress", handle_update_progress)
    graph.add_node("decompose_goal", handle_decompose)
    graph.add_node("check_conflicts", check_conflicts)
    graph.add_node("general", handle_general)
    
    # Add edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "create_goal": "create_goal",
            "review_goals": "review_goals",
            "update_progress": "update_progress",
            "decompose_goal": "decompose_goal",
            "check_conflicts": "check_conflicts",
            "general": "general"
        }
    )
    
    # All handlers end
    for node in ["create_goal", "review_goals", "update_progress", 
                 "decompose_goal", "check_conflicts", "general"]:
        graph.add_edge(node, END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    coach = build_goal_coach()
    
    print("=" * 60)
    print("GOAL LIFECYCLE COACH")
    print("=" * 60)
    print("\nThis coach helps you create, track, and achieve your goals.")
    print("Commands: 'review', 'new goal', 'update', 'quit'\n")
    
    # Show existing goals
    print("Your current goals:")
    for goal in GOAL_STORE.get_all():
        print(f"  • {goal.title} ({goal.progress:.0%} complete)")
    print()
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nKeep working on those goals! 🎯")
            break
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "goals": [],
            "current_goal_id": None,
            "intent": "",
            "proposed_changes": {},
            "conflicts": []
        }
        
        result = coach.invoke(state)
        response = result["messages"][-1].content
        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    main()
