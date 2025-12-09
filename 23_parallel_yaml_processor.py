"""
24. Ultimate HR Coaching Platform - Culture Amp Perform Edition
=================================================================
The most advanced LangGraph implementation combining ALL patterns:

ANTHROPIC PATTERNS:
├── Augmented LLM (retrieval, tools, memory)
├── Prompt Chaining (sequential refinement)
├── Routing (specialist dispatch)
├── Parallelization (sectioning + voting)
├── Orchestrator-Workers (dynamic delegation)
├── Evaluator-Optimizer (quality loop)
└── Autonomous Agent (think-act-observe)

CULTURE AMP PERFORM FEATURES:
├── Self-Reflections (guided self-assessment)
├── Performance Reviews (AI-assisted writing)
├── Goals/OKRs (setting and tracking)
├── 1-on-1 Meetings (agenda generation)
├── Continuous Feedback (SBI model)
├── Shoutouts (values-driven recognition)
├── Calibration (rating alignment)
└── Competencies (skills tracking)

LANGGRAPH FEATURES:
├── Send API (dynamic fan-out)
├── Automatic Fan-In (Annotated reducers)
├── Sub-Graphs (nested workflows)
├── Checkpointing (persistence)
├── Human-in-the-Loop (interrupts)
└── Streaming (real-time updates)

24. Ultimate HR Coaching Platform
=================================
The most advanced LangGraph implementation combining ALL patterns:

ANTHROPIC PATTERNS:
├── Augmented LLM (retrieval, tools, memory)
├── Prompt Chaining (sequential refinement)
├── Routing (specialist dispatch)
├── Parallelization (sectioning + voting)
├── Orchestrator-Workers (dynamic delegation)
├── Evaluator-Optimizer (quality loop)
└── Autonomous Agent (think-act-observe)

LANGGRAPH FEATURES:
├── Send API (dynamic fan-out)
├── Automatic Fan-In (Annotated reducers)
├── Sub-Graphs (nested workflows)
├── Checkpointing (persistence)
├── Human-in-the-Loop (interrupts)
└── Streaming (real-time updates)

OPTIMIZATION:
├── ThreadPoolExecutor (concurrent I/O)
├── Result Caching (hash-based)
├── Weighted Aggregation
├── Contradiction Detection
├── Graceful Degradation
└── Progress Tracking

USE CASES:
├── Interactive coaching conversations
├── Bulk YAML data processing
├── 360° feedback analysis
├── Promotion readiness assessment
├── Risk triage and escalation
└── Goal tracking and OKR analysis
"""

import asyncio
import json
import re
import time
import hashlib
import yaml
import threading
from typing import Annotated, TypedDict, Literal, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from operator import add
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# BANNER
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ULTIMATE HR COACHING PLATFORM                            ║
║                                                                              ║
║  Combining ALL Anthropic Agentic Patterns + LangGraph Advanced Features      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PlatformConfig:
    """Global platform configuration"""
    model: str = "gpt-4o-mini"
    max_parallel_workers: int = 8
    timeout_seconds: float = 45.0
    enable_caching: bool = True
    enable_human_in_loop: bool = False
    quality_threshold: float = 7.0
    max_optimization_iterations: int = 3
    debug_mode: bool = True


CONFIG = PlatformConfig()


# ============================================================
# UTILITIES
# ============================================================

def log(message: str, level: str = "INFO"):
    """Structured logging"""
    if CONFIG.debug_mode:
        icons = {"INFO": "", "SUCCESS": "", "ERROR": "", "WARN": "", "PARALLEL": "", "ROUTE": ""}
        print(f"  {icons.get(level, '•')} {message}")


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response"""
    try:
        return json.loads(text)
    except:
        pass
    
    patterns = [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'\{[\s\S]*\}']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str.strip())
            except:
                continue
    return {}


# ============================================================
# KNOWLEDGE BASE (Augmented LLM - Retrieval)
# ============================================================

class KnowledgeBase:
    """HR Knowledge base for retrieval augmentation"""

    POLICIES = {
        "promotion": {
            "criteria": ["Consistent high performance (2+ cycles)", "Leadership demonstration", "Cross-team impact", "Manager nomination"],
            "process": "Annual review cycle in Q4, mid-year check-ins available",
            "timeline": "Nominations due Oct 15, decisions announced Dec 1"
        },
        "feedback": {
            "model": "SBI - Situation, Behavior, Impact",
            "frequency": "Continuous, with formal reviews quarterly",
            "tools": ["Performance platform", "1:1 templates", "360 surveys"]
        },
        "goals": {
            "framework": "OKRs - Objectives and Key Results",
            "cadence": "Quarterly objectives, weekly check-ins",
            "alignment": "Individual → Team → Department → Company"
        },
        "conflict": {
            "steps": ["Listen to all parties", "Identify underlying needs", "Find common ground", "Document agreements"],
            "escalation": "Manager → HR BP → Employee Relations",
            "resources": ["Mediation services", "Conflict coaching", "Team facilitation"]
        },
        "wellbeing": {
            "resources": ["EAP: 1-800-EAP-HELP", "Mental health days", "Flexible work", "Wellness stipend"],
            "warning_signs": ["Exhaustion", "Cynicism", "Reduced efficacy", "Isolation"],
            "support": "Confidential counseling, manager training, peer support"
        },
        "compensation": {
            "philosophy": "Market competitive (50th-75th percentile)",
            "components": ["Base salary", "Annual bonus", "Equity grants", "Benefits"],
            "review": "Annual in March, off-cycle for promotions"
        }
    }

    @classmethod
    def search(cls, query: str) -> dict:
        """Search knowledge base"""
        query_lower = query.lower()
        for topic, content in cls.POLICIES.items():
            if topic in query_lower:
                return {"topic": topic, "content": content, "found": True}
        return {"topic": None, "content": {}, "found": False}

    @classmethod
    def get_all_topics(cls) -> list[str]:
        return list(cls.POLICIES.keys())


# ============================================================
# MEMORY SYSTEM (Augmented LLM - Memory)
# ============================================================

class ConversationMemory:
    """Persistent conversation memory with episodic and semantic storage"""

    def __init__(self):
        self._sessions: list[dict] = []
        self._user_profile: dict = {}
        self._key_insights: list[str] = []
        self._lock = threading.Lock()

    def add_session(self, summary: str, topics: list[str] = None):
        with self._lock:
            self._sessions.append({
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "topics": topics or []
            })

    def add_insight(self, insight: str):
        with self._lock:
            self._key_insights.append(insight)

    def update_profile(self, key: str, value: Any):
        with self._lock:
            self._user_profile[key] = value

    def get_context(self, max_sessions: int = 5) -> str:
        with self._lock:
            recent = self._sessions[-max_sessions:] if self._sessions else []
            context = "CONVERSATION HISTORY:\n"
            for s in recent:
                context += f"  - {s['timestamp'][:10]}: {s['summary'][:100]}\n"
            if self._key_insights:
                context += f"\nKEY INSIGHTS: {', '.join(self._key_insights[-5:])}"
            if self._user_profile:
                context += f"\nUSER PROFILE: {json.dumps(self._user_profile)}"
            return context


MEMORY = ConversationMemory()


# ============================================================
# CACHING SYSTEM
# ============================================================

class AnalysisCache:
    """Thread-safe LRU cache for analysis results"""

    def __init__(self, max_size: int = 100):
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._stats = {"hits": 0, "misses": 0}

    def _key(self, category: str, content: str) -> str:
        return hashlib.sha256(f"{category}:{content[:500]}".encode()).hexdigest()[:16]

    def get(self, category: str, content: str) -> Optional[dict]:
        key = self._key(category, content)
        with self._lock:
            if key in self._cache:
                self._stats["hits"] += 1
                return self._cache[key]
            self._stats["misses"] += 1
            return None

    def set(self, category: str, content: str, result: dict):
        key = self._key(category, content)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = result

    def stats(self) -> dict:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            return {**self._stats, "hit_rate": self._stats["hits"] / total if total > 0 else 0}


CACHE = AnalysisCache()


# ============================================================
# TOOLS (Augmented LLM - Tools)
# ============================================================

@tool
def search_knowledge(query: str) -> str:
    """Search HR knowledge base for policies and guidance."""
    result = KnowledgeBase.search(query)
    if result["found"]:
        return f"Found policy on '{result['topic']}': {json.dumps(result['content'])}"
    return "No specific policy found. Consider consulting HR directly."


@tool
def create_action_item(title: str, description: str, due_days: int = 7) -> str:
    """Create an action item for follow-up."""
    due_date = datetime.now().isoformat()[:10]
    return f" Action created: '{title}' - {description} (due in {due_days} days)"


@tool
def schedule_followup(topic: str, days: int = 7) -> str:
    """Schedule a follow-up coaching session."""
    return f" Follow-up scheduled in {days} days: {topic}"


@tool
def escalate_to_hr(reason: str, urgency: str = "normal") -> str:
    """Escalate issue to HR Business Partner."""
    return f" Escalated to HR BP ({urgency} priority): {reason}"


@tool
def log_coaching_note(note: str) -> str:
    """Log a confidential coaching note."""
    MEMORY.add_insight(note[:100])
    return f" Note logged confidentially"


TOOLS = [search_knowledge, create_action_item, schedule_followup, escalate_to_hr, log_coaching_note]


# ============================================================
# STATE DEFINITIONS
# ============================================================

class RequestType(str, Enum):
    CONVERSATION = "conversation"
    BULK_ANALYSIS = "bulk_analysis"
    PROMOTION_REVIEW = "promotion_review"
    RISK_TRIAGE = "risk_triage"
    ACTION_PLANNING = "action_planning"
    # Culture Amp Perform Features
    SELF_REFLECTION = "self_reflection"
    PERFORMANCE_REVIEW = "performance_review"
    GOAL_SETTING = "goal_setting"
    ONE_ON_ONE = "one_on_one"
    FEEDBACK_REQUEST = "feedback_request"
    FEEDBACK_WRITING = "feedback_writing"
    SHOUTOUT = "shoutout"
    CALIBRATION = "calibration"
    COMPETENCY_ASSESSMENT = "competency_assessment"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnalysisResult:
    """Result from parallel analyzer"""
    analyzer_id: str
    category: str
    analysis: str
    confidence: float
    insights: list[str]
    recommendations: list[str]
    risk_flags: list[str]
    score: float
    execution_time: float
    from_cache: bool = False


@dataclass
class VoteResult:
    """Result from voting perspective"""
    voter_id: str
    perspective: str
    decision: str
    confidence: float
    reasoning: str
    weight: float


# Main platform state
class PlatformState(TypedDict):
    """Unified state for the entire platform"""
    messages: Annotated[list, add_messages]

    # Request classification
    request_type: str
    request_complexity: str
    identified_topics: list[str]

    # Routing
    route: str
    route_confidence: float

    # Knowledge & Memory
    retrieved_knowledge: dict
    memory_context: str

    # Parallel processing
    chunks: list[dict]
    parallel_results: Annotated[list[dict], add]
    vote_results: list[dict]

    # Quality control
    draft_response: str
    quality_scores: dict
    optimization_iteration: int

    # Orchestration
    worker_assignments: list[dict]
    worker_outputs: list[dict]

    # Synthesis
    final_analysis: dict
    executive_summary: str
    action_items: list[str]
    risk_assessment: dict

    # Agent loop
    agent_plan: list[str]
    agent_observations: list[str]
    agent_actions_taken: list[str]
    agent_complete: bool

    # Action Planning (Orchestrator-Workers)
    planning_context: str
    worker_tasks: list[dict]
    worker_results: Annotated[list[dict], add]
    action_plan: dict

    # Metadata
    processing_stats: dict
    contradictions: list[dict]
    human_feedback: str

    # Culture Amp Perform State
    perform_context: dict  # Employee data, review cycle, etc.
    reflection_responses: list[dict]  # Self-reflection answers
    feedback_items: list[dict]  # Continuous feedback records
    goals: list[dict]  # OKRs and goals
    one_on_one_agenda: dict  # 1:1 meeting agenda
    shoutout_draft: dict  # Recognition shoutout
    calibration_data: dict  # Calibration ratings
    competency_scores: dict  # Skill assessments


# Chunk processing state
class ChunkState(TypedDict):
    """State for individual chunk processing"""
    chunk_id: str
    chunk_type: str
    chunk_data: Any
    analysis: str
    insights: list[str]
    recommendations: list[str]
    risk_flags: list[str]
    score: float


# ============================================================
# SPECIALIST COACHES (Routing Pattern)
# ============================================================

SPECIALIST_PROMPTS = {
    "career": """You are a CAREER DEVELOPMENT specialist coach.
Focus on: promotions, skill development, career paths, visibility, sponsorship, job transitions.
Be strategic, action-oriented, and help them create a concrete development plan.""",

    "conflict": """You are a CONFLICT RESOLUTION specialist coach.
Focus on: understanding perspectives, de-escalation, difficult conversations, team dynamics.
Stay neutral, help them see other viewpoints, and teach communication techniques.""",

    "wellbeing": """You are a WELLBEING & MENTAL HEALTH specialist coach.
Focus on: stress management, burnout prevention, work-life balance, boundaries, self-care.
Be warm and validating. Mention EAP resources when appropriate. Watch for crisis indicators.""",

    "feedback": """You are a FEEDBACK & PERFORMANCE specialist coach.
Focus on: giving effective feedback, receiving feedback, performance conversations, reviews.
Teach the SBI model (Situation-Behavior-Impact). Help them prepare for difficult conversations.""",

    "leadership": """You are a LEADERSHIP DEVELOPMENT specialist coach.
Focus on: team management, delegation, influence, executive presence, strategic thinking.
Help them grow from individual contributor to leader mindset.""",

    "goals": """You are a GOALS & PRODUCTIVITY specialist coach.
Focus on: OKRs, goal-setting, prioritization, time management, focus, execution.
Help them create SMART goals and overcome obstacles to achievement.""",

    "compensation": """You are a COMPENSATION & NEGOTIATION specialist coach.
Focus on: salary discussions, promotion negotiations, equity understanding, total rewards.
Help them prepare data-driven cases and practice conversations.""",

    "general": """You are a supportive HR coach.
Provide helpful, empathetic guidance on workplace topics.
If the topic seems specialized, suggest they might benefit from focused coaching."""
}


# ============================================================
# ANALYZERS (Parallelization Pattern - Sectioning)
# ============================================================

ANALYZERS = {
    "technical": {
        "prompt": "Analyze TECHNICAL SKILLS and capabilities. Focus on expertise, growth, and gaps.",
        "weight": 1.2
    },
    "leadership": {
        "prompt": "Analyze LEADERSHIP CAPABILITIES. Focus on influence, decision-making, mentoring.",
        "weight": 1.3
    },
    "communication": {
        "prompt": "Analyze COMMUNICATION SKILLS. Focus on clarity, stakeholder management, influence.",
        "weight": 1.0
    },
    "collaboration": {
        "prompt": "Analyze COLLABORATION & TEAMWORK. Focus on cross-team work, relationships.",
        "weight": 1.0
    },
    "delivery": {
        "prompt": "Analyze EXECUTION & DELIVERY. Focus on results, quality, reliability, impact.",
        "weight": 1.1
    },
    "growth": {
        "prompt": "Analyze GROWTH MINDSET. Focus on learning, adaptability, feedback receptiveness.",
        "weight": 0.9
    },
    "culture": {
        "prompt": "Analyze CULTURE FIT. Focus on values alignment, team contribution, integrity.",
        "weight": 0.8
    },
    "risk": {
        "prompt": "Analyze POTENTIAL RISKS. Focus on red flags, concerns, gaps, issues.",
        "weight": 1.4
    }
}


# ============================================================
# VOTERS (Parallelization Pattern - Voting)
# ============================================================

VOTERS = {
    "advocate": {
        "prompt": "As an ADVOCATE who champions growth, look for potential and achievements.",
        "weight": 0.8,
        "bias": "positive"
    },
    "skeptic": {
        "prompt": "As a SKEPTIC focused on standards, look for gaps and reasons to wait.",
        "weight": 1.2,
        "bias": "negative"
    },
    "balanced": {
        "prompt": "As a BALANCED JUDGE, weigh all evidence objectively.",
        "weight": 1.5,
        "bias": "neutral"
    },
    "peer": {
        "prompt": "As a PEER at target level, assess if they'd be an effective colleague.",
        "weight": 1.0,
        "bias": "peer"
    },
    "manager": {
        "prompt": "As a HIRING MANAGER, assess readiness for increased responsibility.",
        "weight": 1.3,
        "bias": "manager"
    }
}


# ============================================================
# WORKERS (Orchestrator-Workers Pattern)
# ============================================================

WORKERS = {
    "researcher": {
        "prompt": "You are a RESEARCH specialist. Gather relevant context and information.",
        "capabilities": ["knowledge_search", "data_gathering", "context_building"]
    },
    "analyst": {
        "prompt": "You are an ANALYSIS specialist. Identify patterns and extract insights.",
        "capabilities": ["pattern_recognition", "insight_extraction", "trend_analysis"]
    },
    "strategist": {
        "prompt": "You are a STRATEGY specialist. Develop actionable recommendations.",
        "capabilities": ["planning", "recommendation", "roadmap_creation"]
    },
    "communicator": {
        "prompt": "You are a COMMUNICATION specialist. Craft clear messages and talking points.",
        "capabilities": ["message_crafting", "talking_points", "presentation"]
    },
    "risk_assessor": {
        "prompt": "You are a RISK ASSESSMENT specialist. Identify and evaluate risks.",
        "capabilities": ["risk_identification", "impact_assessment", "mitigation_planning"]
    }
}


# ============================================================
# ACTION PLANNING WORKERS (Orchestrator-Workers Pattern)
# ============================================================

ACTION_PLAN_WORKERS = {
    "goal_setter": {
        "prompt": """You are a GOAL SETTING specialist.
Create SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound).
Break down high-level objectives into concrete, actionable goals.
Ensure each goal has clear success criteria.""",
        "output_schema": ["goal_title", "description", "success_criteria", "timeframe"]
    },
    "timeline_planner": {
        "prompt": """You are a TIMELINE & MILESTONE specialist.
Create realistic timelines with clear milestones.
Consider dependencies between tasks.
Build in buffer time for unexpected challenges.
Suggest cadence for check-ins and reviews.""",
        "output_schema": ["milestone", "target_date", "dependencies", "deliverables"]
    },
    "resource_identifier": {
        "prompt": """You are a RESOURCE & SUPPORT specialist.
Identify what resources, training, or support is needed.
Suggest mentors, courses, tools, or experiences.
Consider budget and time constraints.
Recommend internal vs external resources.""",
        "output_schema": ["resource_type", "description", "source", "estimated_cost", "priority"]
    },
    "obstacle_navigator": {
        "prompt": """You are an OBSTACLE & RISK specialist.
Anticipate potential blockers and challenges.
Create contingency plans for each risk.
Suggest strategies to overcome common pitfalls.
Identify early warning signs of derailment.""",
        "output_schema": ["obstacle", "likelihood", "impact", "mitigation_strategy", "contingency"]
    },
    "accountability_designer": {
        "prompt": """You are an ACCOUNTABILITY & MEASUREMENT specialist.
Design check-in schedules and progress tracking.
Create metrics and KPIs to measure progress.
Suggest accountability partners or structures.
Define how success will be celebrated.""",
        "output_schema": ["check_in_type", "frequency", "metrics", "accountability_partner", "review_criteria"]
    }
}


# ============================================================
# CULTURE AMP PERFORM - VALUES & COMPETENCIES
# ============================================================

# Culture Amp's Four Values
CULTURE_AMP_VALUES = {
    "have_the_courage_to_be_vulnerable": {
        "description": "Share openly, admit mistakes, ask for help, give and receive feedback honestly",
        "behaviors": [
            "Admits when they don't know something",
            "Asks for feedback proactively",
            "Shares failures as learning opportunities",
            "Creates psychological safety for others"
        ]
    },
    "trust_others_to_do_amazing_things": {
        "description": "Empower others, delegate with trust, support autonomy, believe in team capability",
        "behaviors": [
            "Delegates meaningfully without micromanaging",
            "Supports team decisions even when different from their own",
            "Gives others opportunities to stretch and grow",
            "Celebrates team successes"
        ]
    },
    "learn_faster_through_feedback": {
        "description": "Seek feedback continuously, act on it quickly, help others improve through feedback",
        "behaviors": [
            "Actively seeks feedback from multiple sources",
            "Responds to feedback with action, not defensiveness",
            "Provides timely, specific, actionable feedback to others",
            "Iterates quickly based on learnings"
        ]
    },
    "amplify_others": {
        "description": "Lift others up, share credit, mentor, advocate for peers, build inclusive environment",
        "behaviors": [
            "Publicly recognizes others' contributions",
            "Mentors and sponsors colleagues",
            "Creates space for diverse voices",
            "Shares knowledge and opportunities generously"
        ]
    }
}

# Competency Framework (aligned with Culture Amp's approach)
COMPETENCY_FRAMEWORK = {
    "delivers_results": {
        "description": "Achieves outcomes that drive business impact",
        "levels": {
            "developing": "Learning to deliver work with guidance",
            "performing": "Consistently delivers quality work on time",
            "exceeding": "Drives significant impact beyond immediate scope",
            "leading": "Sets the standard for delivery excellence"
        }
    },
    "builds_relationships": {
        "description": "Creates strong connections and collaboration",
        "levels": {
            "developing": "Building foundational working relationships",
            "performing": "Maintains effective cross-functional partnerships",
            "exceeding": "Trusted advisor across multiple teams",
            "leading": "Shapes organizational relationships and culture"
        }
    },
    "drives_innovation": {
        "description": "Brings new ideas and improves ways of working",
        "levels": {
            "developing": "Contributes ideas and open to experimentation",
            "performing": "Regularly introduces improvements",
            "exceeding": "Leads innovation initiatives",
            "leading": "Creates culture of innovation"
        }
    },
    "develops_self_and_others": {
        "description": "Continuous growth mindset and supporting team development",
        "levels": {
            "developing": "Actively learning and receptive to feedback",
            "performing": "Grows skills and helps peers",
            "exceeding": "Mentors others and drives team capability",
            "leading": "Builds organizational learning culture"
        }
    },
    "communicates_effectively": {
        "description": "Clear, compelling, and appropriate communication",
        "levels": {
            "developing": "Communicates clearly in routine situations",
            "performing": "Adapts communication style effectively",
            "exceeding": "Influences through communication",
            "leading": "Shapes organizational narrative"
        }
    }
}

# Self-Reflection Question Templates
SELF_REFLECTION_QUESTIONS = {
    "accomplishments": "What are you most proud of accomplishing this period? What impact did it have?",
    "challenges": "What challenges did you face? How did you handle them? What would you do differently?",
    "growth": "How have you grown? What new skills or capabilities have you developed?",
    "feedback_applied": "What feedback have you received and how have you applied it?",
    "goals_progress": "What progress have you made on your goals? What's blocking you?",
    "support_needed": "What support do you need from your manager or team?",
    "future_focus": "What do you want to focus on in the next period?"
}

# Performance Rating Scale (Culture Amp style)
PERFORMANCE_RATINGS = {
    "significantly_below": {
        "score": 1,
        "label": "Significantly Below Expectations",
        "description": "Performance requires significant improvement to meet role expectations"
    },
    "below": {
        "score": 2,
        "label": "Below Expectations",
        "description": "Performance partially meets expectations with clear development areas"
    },
    "meets": {
        "score": 3,
        "label": "Meets Expectations",
        "description": "Solid performance that meets role expectations"
    },
    "exceeds": {
        "score": 4,
        "label": "Exceeds Expectations",
        "description": "Strong performance that often exceeds expectations"
    },
    "significantly_exceeds": {
        "score": 5,
        "label": "Significantly Exceeds Expectations",
        "description": "Exceptional performance with outstanding impact"
    }
}


# ============================================================
# CULTURE AMP PERFORM WORKERS
# ============================================================

PERFORM_WORKERS = {
    # Self-Reflection Assistant
    "reflection_guide": {
        "prompt": """You are a SELF-REFLECTION GUIDE helping employees complete meaningful self-reflections.

Help the employee:
- Articulate their accomplishments with specific impact and metrics
- Reflect honestly on challenges and learnings
- Identify growth areas without being overly self-critical
- Connect their work to team and company goals
- Be specific and use the STAR format (Situation, Task, Action, Result)

Tone: Supportive, encouraging specificity, helping them showcase their best work.""",
        "output_schema": ["question", "suggested_response", "tips", "examples"]
    },
    
    # Manager Review Assistant
    "review_writer": {
        "prompt": """You are a PERFORMANCE REVIEW WRITING assistant for managers.

Help managers write:
- Balanced reviews with both strengths and development areas
- Specific, evidence-based feedback (not vague)
- Forward-looking development suggestions
- Fair and consistent assessments
- Ratings that align with the evidence presented

Use Culture Amp's recommended structure:
1. Summary of overall performance
2. Key accomplishments and impact
3. Strengths demonstrated
4. Areas for development
5. Goals and expectations for next period

Tone: Professional, constructive, specific, growth-oriented.""",
        "output_schema": ["section", "content", "suggestions", "rating_recommendation"]
    },
    
    # Feedback Improver (AI Suggestions feature)
    "feedback_improver": {
        "prompt": """You are a FEEDBACK IMPROVER that helps make feedback more effective.

Analyze feedback and suggest improvements to make it:
- More SPECIFIC (concrete examples, not generalities)
- More ACTIONABLE (clear next steps)
- More BALANCED (acknowledges what's working)
- More SBI-STRUCTURED (Situation-Behavior-Impact)
- More GROWTH-ORIENTED (future-focused)

Provide the improved version and explain what was changed and why.

Example SBI format:
- Situation: "In yesterday's client presentation..."
- Behavior: "You clearly explained the technical architecture..."
- Impact: "The client expressed confidence in our approach."

Tone: Helpful coach, non-judgmental, constructive.""",
        "output_schema": ["original", "improved", "changes_made", "effectiveness_score"]
    },
    
    # Goal/OKR Assistant
    "goal_architect": {
        "prompt": """You are a GOAL & OKR specialist helping set effective objectives.

Help create:
- Clear Objectives (qualitative, inspiring, ambitious)
- Measurable Key Results (quantitative, time-bound)
- Proper goal alignment (Individual → Team → Company)
- Stretch goals that are challenging but achievable (70% confidence)

OKR Best Practices:
- 3-5 objectives per quarter
- 2-4 key results per objective
- Key results should be measurable, not tasks
- Include a mix of committed (100% confidence) and stretch goals

Tone: Strategic, motivating, practical.""",
        "output_schema": ["objective", "key_results", "alignment", "confidence_level"]
    },
    
    # 1:1 Meeting Assistant
    "one_on_one_prep": {
        "prompt": """You are a 1:1 MEETING PREPARATION specialist.

Help prepare effective 1:1 agendas that include:
- Progress updates on goals and projects
- Blockers and support needed
- Feedback (both directions)
- Career development discussion
- Recognition and wins
- Personal check-in (wellbeing)

Suggest talking points and questions for both manager and employee.
Flag any concerning patterns that should be discussed.

Tone: Collaborative, structured, supportive.""",
        "output_schema": ["agenda_item", "talking_points", "questions", "time_allocation"]
    },
    
    # Shoutout/Recognition Writer
    "shoutout_crafter": {
        "prompt": """You are a RECOGNITION & SHOUTOUT specialist.

Help craft meaningful shoutouts that:
- Are specific about what the person did
- Explain the impact of their contribution
- Connect to company values when relevant
- Feel genuine and personal (not generic)
- Are appropriate for public sharing

Culture Amp Values to connect to:
- Have the courage to be vulnerable
- Trust others to do amazing things
- Learn faster through feedback
- Amplify others

Tone: Warm, genuine, celebratory.""",
        "output_schema": ["shoutout_text", "value_connection", "suggested_emoji", "visibility"]
    },
    
    # Calibration Assistant
    "calibration_analyst": {
        "prompt": """You are a CALIBRATION specialist helping managers align on fair ratings.

Analyze performance data to:
- Identify potential rating inconsistencies
- Flag possible bias (recency, halo effect, central tendency)
- Compare similar roles/levels for consistency
- Highlight evidence gaps
- Suggest calibration discussion points

Calibration questions to consider:
- Is this rating consistent with peers at the same level?
- Is there sufficient evidence to support this rating?
- Are we accounting for scope and complexity differences?
- Have we considered the full review period, not just recent events?

Tone: Objective, data-driven, fair.""",
        "output_schema": ["rating_analysis", "consistency_flags", "bias_alerts", "discussion_points"]
    },
    
    # Competency Assessor
    "competency_assessor": {
        "prompt": """You are a COMPETENCY ASSESSMENT specialist.

Evaluate competencies against the framework:
- Delivers Results
- Builds Relationships
- Drives Innovation
- Develops Self and Others
- Communicates Effectively

For each competency:
- Assess current level (Developing/Performing/Exceeding/Leading)
- Provide specific evidence for the assessment
- Identify 1-2 development actions
- Suggest resources or experiences for growth

Tone: Developmental, specific, actionable.""",
        "output_schema": ["competency", "current_level", "evidence", "development_actions", "resources"]
    },
    
    # Feedback Request Crafter
    "feedback_requester": {
        "prompt": """You are a FEEDBACK REQUEST specialist.

Help employees request feedback effectively by:
- Identifying the right people to ask
- Crafting specific questions (not "how am I doing?")
- Timing the request appropriately
- Making it easy for the responder

Good feedback questions:
- "What's one thing I could do differently to improve X?"
- "How effective was my communication in the Y project?"
- "What strengths should I leverage more?"

Tone: Humble, specific, growth-oriented.""",
        "output_schema": ["recipient", "context", "specific_questions", "timing_suggestion"]
    }
}


# ============================================================
# PARALLEL EXECUTION ENGINE
# ============================================================

class ParallelEngine:
    """Optimized parallel execution with ThreadPoolExecutor"""

    def __init__(self):
        self.llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)

    def run_analyzers(
        self,
        content: str,
        analyzers: dict = ANALYZERS
    ) -> tuple[list[AnalysisResult], dict]:
        """Run multiple analyzers in parallel"""

        log(f"Starting parallel analysis ({len(analyzers)} analyzers)", "PARALLEL")
        results = []
        stats = {"total": len(analyzers), "completed": 0, "cached": 0, "failed": 0}
        start_time = time.time()

        def run_single(analyzer_id: str, config: dict) -> Optional[AnalysisResult]:
            t0 = time.time()

            # Check cache
            if CONFIG.enable_caching:
                cached = CACHE.get(analyzer_id, content)
                if cached:
                    log(f"Cache hit: {analyzer_id}", "SUCCESS")
                    return AnalysisResult(
                        analyzer_id=analyzer_id, category=analyzer_id,
                        analysis=cached.get("analysis", ""),
                        confidence=cached.get("confidence", 0.7),
                        insights=cached.get("insights", []),
                        recommendations=cached.get("recommendations", []),
                        risk_flags=cached.get("risk_flags", []),
                        score=cached.get("score", 5),
                        execution_time=0.01, from_cache=True
                    )

            prompt = f"""{config['prompt']}

CONTENT TO ANALYZE:
{content[:3000]}

Return JSON:
{{"analysis": "detailed analysis", "score": 1-10, "confidence": 0.0-1.0,
"insights": ["insight1", "insight2"], "recommendations": ["rec1", "rec2"],
"risk_flags": []}}"""
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                data = extract_json(response.content)
                if not data:
                    raise ValueError("No JSON extracted")
                
                # Cache result
                if CONFIG.enable_caching:
                    CACHE.set(analyzer_id, content, data)
                
                elapsed = time.time() - t0
                log(f"Completed: {analyzer_id} ({elapsed:.1f}s)", "SUCCESS")
                
                return AnalysisResult(
                    analyzer_id=analyzer_id, category=analyzer_id,
                    analysis=data.get("analysis", ""),
                    confidence=data.get("confidence", 0.7),
                    insights=data.get("insights", []),
                    recommendations=data.get("recommendations", []),
                    risk_flags=data.get("risk_flags", []),
                    score=data.get("score", 5),
                    execution_time=elapsed
                )
            except Exception as e:
                log(f"Failed: {analyzer_id} - {e}", "ERROR")
                return None
        
        with ThreadPoolExecutor(max_workers=CONFIG.max_parallel_workers) as executor:
            futures = {
                executor.submit(run_single, aid, cfg): aid
                for aid, cfg in analyzers.items()
            }
            
            for future in as_completed(futures, timeout=CONFIG.timeout_seconds):
                result = future.result()
                if result:
                    results.append(result)
                    stats["completed"] += 1
                    if result.from_cache:
                        stats["cached"] += 1
                else:
                    stats["failed"] += 1
        
        stats["elapsed"] = time.time() - start_time
        stats["cache_stats"] = CACHE.stats()
        log(f"Analysis complete: {stats['completed']}/{stats['total']} in {stats['elapsed']:.1f}s", "SUCCESS")
        
        return results, stats
    
    def run_voters(
        self, 
        summary: str, 
        voters: dict = VOTERS
    ) -> tuple[list[VoteResult], dict]:
        """Run multiple voting perspectives in parallel"""
        
        log(f"Starting parallel voting ({len(voters)} perspectives)", "PARALLEL")
        results = []
        stats = {"total": len(voters), "completed": 0, "failed": 0}
        start_time = time.time()
        
        def run_single(voter_id: str, config: dict) -> Optional[VoteResult]:
            prompt = f"""{config['prompt']}

ANALYSIS SUMMARY:
{summary}

Assess readiness. Return JSON:
{{"decision": "ready|not_ready|needs_development", "confidence": 0.0-1.0,
"reasoning": "explanation", "key_strengths": [], "key_concerns": []}}"""
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                data = extract_json(response.content)
                if not data:
                    raise ValueError("No JSON")
                
                log(f"Vote: {voter_id} → {data.get('decision', '?')}", "SUCCESS")
                
                return VoteResult(
                    voter_id=voter_id,
                    perspective=config.get("bias", "neutral"),
                    decision=data.get("decision", "needs_development"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    weight=config.get("weight", 1.0)
                )
            except Exception as e:
                log(f"Vote failed: {voter_id} - {e}", "ERROR")
                return None
        
        with ThreadPoolExecutor(max_workers=CONFIG.max_parallel_workers) as executor:
            futures = {executor.submit(run_single, vid, cfg): vid for vid, cfg in voters.items()}
            for future in as_completed(futures, timeout=CONFIG.timeout_seconds):
                result = future.result()
                if result:
                    results.append(result)
                    stats["completed"] += 1
                else:
                    stats["failed"] += 1
        
        stats["elapsed"] = time.time() - start_time
        log(f"Voting complete: {stats['completed']}/{stats['total']} in {stats['elapsed']:.1f}s", "SUCCESS")
        
        return results, stats


PARALLEL_ENGINE = ParallelEngine()


# ============================================================
# SMART AGGREGATOR
# ============================================================

class SmartAggregator:
    """Intelligent aggregation with contradiction detection"""

    @staticmethod
    def detect_contradictions(results: list[AnalysisResult]) -> list[dict]:
        """Detect contradictions between analyses"""
        contradictions = []
        positive = ["strong", "excellent", "exceptional", "outstanding", "ready"]
        negative = ["weak", "lacking", "poor", "insufficient", "not ready", "gap"]

        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                r1_text = " ".join(r1.insights).lower()
                r2_text = " ".join(r2.insights).lower()

                r1_pos = any(w in r1_text for w in positive)
                r2_neg = any(w in r2_text for w in negative)

                if r1_pos and r2_neg:
                    contradictions.append({
                        "analyzers": [r1.analyzer_id, r2.analyzer_id],
                        "type": "sentiment_mismatch",
                        "description": f"{r1.analyzer_id} positive vs {r2.analyzer_id} negative"
                    })

        return contradictions

    @staticmethod
    def weighted_vote(votes: list[VoteResult]) -> dict:
        """Aggregate votes with confidence weighting"""
        if not votes:
            return {"decision": "insufficient_data", "confidence": 0, "consensus": "none"}

        scores = {"ready": 0.0, "not_ready": 0.0, "needs_development": 0.0}
        total_weight = 0.0

        for v in votes:
            eff_weight = v.weight * v.confidence
            scores[v.decision] = scores.get(v.decision, 0) + eff_weight
            total_weight += eff_weight

        if total_weight > 0:
            scores = {k: v / total_weight for k, v in scores.items()}

        winner = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

        return {
            "decision": winner,
            "confidence": scores[winner],
            "margin": margin,
            "consensus": "strong" if margin > 0.3 else "moderate" if margin > 0.15 else "weak",
            "breakdown": scores,
            "votes": [{"voter": v.voter_id, "decision": v.decision, "confidence": v.confidence} for v in votes]
        }


AGGREGATOR = SmartAggregator()


# ============================================================
# GRAPH NODES
# ============================================================

def classify_request(state: PlatformState) -> dict:
    """Classify incoming request and determine processing path"""

    log("Classifying request...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0)

    last_msg = state["messages"][-1].content if state["messages"] else ""

    # If request_type is already explicitly set (not empty), respect it
    if state.get("request_type") and state["request_type"] != "":
        log(f"Using pre-set request type: {state['request_type']}", "INFO")
        return {}

    # Check if it's YAML bulk data
    if last_msg.strip().startswith(("---", "metadata:", "departments:", "employees:")):
        return {
            "request_type": RequestType.BULK_ANALYSIS.value,
            "request_complexity": "high",
            "identified_topics": ["bulk_data"],
            "route": "bulk_processor"
        }

    # Check for action planning keywords
    action_keywords = ["action plan", "create a plan", "development plan", "transition plan",
                       "career plan", "90 day plan", "30 day plan", "roadmap", "step by step plan"]
    if any(kw in last_msg.lower() for kw in action_keywords):
        return {
            "request_type": RequestType.ACTION_PLANNING.value,
            "request_complexity": "high",
            "identified_topics": ["planning", "goals"],
            "route": "career"
        }

    # ═══ CULTURE AMP PERFORM KEYWORD DETECTION ═══
    msg_lower = last_msg.lower()

    # Self-Reflection
    reflection_keywords = ["self-reflection", "self reflection", "self assessment", "self-assessment",
                          "my accomplishments", "what i achieved", "reflect on my performance",
                          "help me write my self", "performance reflection"]
    if any(kw in msg_lower for kw in reflection_keywords):
        return {
            "request_type": RequestType.SELF_REFLECTION.value,
            "request_complexity": "medium",
            "identified_topics": ["performance", "reflection"],
            "route": "feedback"
        }

    # Performance Review Writing (Manager)
    review_keywords = ["write a review for", "performance review for", "review my direct report",
                      "manager review", "write feedback for", "assess my team member",
                      "draft a review", "help me rate"]
    if any(kw in msg_lower for kw in review_keywords):
        return {
            "request_type": RequestType.PERFORMANCE_REVIEW.value,
            "request_complexity": "high",
            "identified_topics": ["performance", "review", "feedback"],
            "route": "feedback"
        }

    # Goal Setting / OKRs
    goal_keywords = ["set goals", "okr", "objectives", "key results", "quarterly goals",
                    "goal setting", "set my objectives", "create okrs", "write goals",
                    "help me with my goals"]
    if any(kw in msg_lower for kw in goal_keywords):
        return {
            "request_type": RequestType.GOAL_SETTING.value,
            "request_complexity": "medium",
            "identified_topics": ["goals", "okrs"],
            "route": "goals"
        }

    # 1:1 Meeting Prep
    one_on_one_keywords = ["1:1", "one on one", "1-on-1", "meeting agenda", "1:1 agenda",
                          "prepare for my 1:1", "one-on-one", "talking points for",
                          "meeting with my manager", "meeting with my direct report"]
    if any(kw in msg_lower for kw in one_on_one_keywords):
        return {
            "request_type": RequestType.ONE_ON_ONE.value,
            "request_complexity": "medium",
            "identified_topics": ["meeting", "conversation"],
            "route": "feedback"
        }

    # Feedback Request
    feedback_request_keywords = ["ask for feedback", "request feedback", "get feedback from",
                                "how to ask for feedback", "feedback request"]
    if any(kw in msg_lower for kw in feedback_request_keywords):
        return {
            "request_type": RequestType.FEEDBACK_REQUEST.value,
            "request_complexity": "low",
            "identified_topics": ["feedback"],
            "route": "feedback"
        }

    # Feedback Writing/Improvement
    feedback_writing_keywords = ["improve this feedback", "make this feedback better",
                                "rewrite this feedback", "sbi feedback", "sbi format",
                                "help me give feedback", "write feedback"]
    if any(kw in msg_lower for kw in feedback_writing_keywords):
        return {
            "request_type": RequestType.FEEDBACK_WRITING.value,
            "request_complexity": "medium",
            "identified_topics": ["feedback", "communication"],
            "route": "feedback"
        }

    # Shoutout/Recognition
    shoutout_keywords = ["shoutout", "recognize", "recognition", "praise", "thank",
                        "appreciate", "celebrate", "kudos", "acknowledgment"]
    if any(kw in msg_lower for kw in shoutout_keywords):
        return {
            "request_type": RequestType.SHOUTOUT.value,
            "request_complexity": "low",
            "identified_topics": ["recognition"],
            "route": "general"
        }

    # Calibration
    calibration_keywords = ["calibration", "calibrate ratings", "rating consistency",
                           "performance ratings", "compare ratings", "fair rating"]
    if any(kw in msg_lower for kw in calibration_keywords):
        return {
            "request_type": RequestType.CALIBRATION.value,
            "request_complexity": "high",
            "identified_topics": ["performance", "calibration"],
            "route": "feedback"
        }

    # Competency Assessment
    competency_keywords = ["competency", "competencies", "skills assessment", "capability",
                          "assess my skills", "skill gaps", "development areas"]
    if any(kw in msg_lower for kw in competency_keywords):
        return {
            "request_type": RequestType.COMPETENCY_ASSESSMENT.value,
            "request_complexity": "medium",
            "identified_topics": ["skills", "development"],
            "route": "career"
        }

    prompt = f"""Classify this HR coaching request.

REQUEST: {last_msg[:500]}

Return JSON:
{{"type": "conversation|promotion_review|risk_triage|bulk_analysis",
"complexity": "low|medium|high",
"topics": ["topic1", "topic2"],
"route": "career|conflict|wellbeing|feedback|leadership|goals|compensation|general",
"route_confidence": 0.0-1.0}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    data = extract_json(response.content)
    
    log(f"Request type: {data.get('type', 'conversation')}, Route: {data.get('route', 'general')}", "ROUTE")
    
    return {
        "request_type": data.get("type", RequestType.CONVERSATION.value),
        "request_complexity": data.get("complexity", "medium"),
        "identified_topics": data.get("topics", []),
        "route": data.get("route", "general"),
        "route_confidence": data.get("route_confidence", 0.7)
    }


def retrieve_knowledge(state: PlatformState) -> dict:
    """Retrieve relevant knowledge (Augmented LLM - Retrieval)"""
    
    log("Retrieving knowledge...", "INFO")
    
    topics = state.get("identified_topics", [])
    retrieved = {}
    
    for topic in topics:
        result = KnowledgeBase.search(topic)
        if result["found"]:
            retrieved[topic] = result["content"]
    
    # Also search based on route
    route_result = KnowledgeBase.search(state.get("route", ""))
    if route_result["found"]:
        retrieved[state["route"]] = route_result["content"]
    
    # Get memory context
    memory_ctx = MEMORY.get_context()
    
    log(f"Retrieved {len(retrieved)} knowledge items", "SUCCESS")
    
    return {
        "retrieved_knowledge": retrieved,
        "memory_context": memory_ctx
    }


def route_to_specialist(state: PlatformState) -> str:
    """Route to appropriate specialist (Routing Pattern)"""
    
    route = state.get("route", "general")
    request_type = state.get("request_type", RequestType.CONVERSATION.value)
    
    # Route based on request type first
    if request_type == RequestType.BULK_ANALYSIS.value:
        return "bulk_processor"
    elif request_type == RequestType.PROMOTION_REVIEW.value:
        return "promotion_analyzer"
    elif request_type == RequestType.RISK_TRIAGE.value:
        return "risk_triager"
    
    # Then route based on topic
    if route in SPECIALIST_PROMPTS:
        return f"specialist_{route}"
    
    return "specialist_general"


def specialist_coach(state: PlatformState) -> dict:
    """Specialist coaching response with prompt chaining"""

    route = state.get("route", "general")
    specialist_prompt = SPECIALIST_PROMPTS.get(route, SPECIALIST_PROMPTS["general"])

    log(f"Specialist coach: {route}", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.7)

    # PROMPT CHAINING: Step 1 - Analyze
    user_msg = state["messages"][-1].content if state["messages"] else ""
    knowledge = json.dumps(state.get("retrieved_knowledge", {}), indent=2)
    memory = state.get("memory_context", "")

    analyze_prompt = f"""Analyze this coaching request:

REQUEST: {user_msg}

RELEVANT POLICIES:
{knowledge}

CONVERSATION HISTORY:
{memory}

Identify:
1. Core issue
2. Emotional state
3. Underlying needs
4. Key considerations"""

    analysis = llm.invoke([HumanMessage(content=analyze_prompt)]).content
    
    # PROMPT CHAINING: Step 2 - Generate response
    response_prompt = f"""{specialist_prompt}

ANALYSIS:
{analysis}

USER REQUEST:
{user_msg}

Provide a warm, helpful coaching response that:
1. Acknowledges their situation
2. Provides specific guidance
3. Suggests concrete next steps
4. Offers to follow up if needed"""

    response = llm.invoke([HumanMessage(content=response_prompt)]).content
    
    # Update memory
    MEMORY.add_session(f"Coaching on {route}: {user_msg[:50]}...", [route])
    
    return {
        "draft_response": response,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def evaluate_response(state: PlatformState) -> dict:
    """Evaluate response quality (Evaluator-Optimizer Pattern)"""
    
    log(f"Evaluating response (iteration {state.get('optimization_iteration', 0) + 1})", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.2)
    
    prompt = f"""Evaluate this coaching response:

USER REQUEST: {state["messages"][-1].content if state["messages"] else ""}

RESPONSE: {state.get("draft_response", "")}

Score 1-10 on:
- empathy: Acknowledges feelings?
- actionability: Clear next steps?
- accuracy: Sound advice?
- tone: Warm and professional?

Return JSON:
{{"empathy": 1-10, "actionability": 1-10, "accuracy": 1-10, "tone": 1-10,
"overall": 1-10, "feedback": "improvements needed", "pass": true/false}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    scores = extract_json(response.content)
    
    if not scores:
        scores = {"overall": 8, "pass": True, "feedback": ""}
    
    log(f"Quality score: {scores.get('overall', '?')}/10 (pass: {scores.get('pass', '?')})", "INFO")
    
    return {
        "quality_scores": scores,
        "optimization_iteration": state.get("optimization_iteration", 0) + 1
    }


def should_optimize(state: PlatformState) -> Literal["optimize", "finalize"]:
    """Decide if response needs optimization"""
    
    scores = state.get("quality_scores", {})
    iteration = state.get("optimization_iteration", 0)
    
    if scores.get("pass", True) or iteration >= CONFIG.max_optimization_iterations:
        return "finalize"
    return "optimize"


def optimize_response(state: PlatformState) -> dict:
    """Optimize response based on feedback"""
    
    log("Optimizing response...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.7)
    
    feedback = state.get("quality_scores", {}).get("feedback", "")
    
    prompt = f"""Improve this coaching response based on feedback:

ORIGINAL RESPONSE:
{state.get("draft_response", "")}

FEEDBACK:
{feedback}

Provide an improved response that addresses the feedback."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"draft_response": response.content}


def finalize_response(state: PlatformState) -> dict:
    """Finalize and return response"""
    
    log("Finalizing response", "SUCCESS")
    
    response = state.get("draft_response", "I'm here to help. Could you tell me more?")
    
    return {"messages": [AIMessage(content=response)]}


# ============================================================
# BULK PROCESSING (Send API + Sub-graphs)
# ============================================================

def parse_bulk_data(state: PlatformState) -> dict:
    """Parse bulk YAML data into chunks"""
    
    log("Parsing bulk data...", "INFO")
    
    content = state["messages"][-1].content if state["messages"] else ""
    
    try:
        parsed = yaml.safe_load(content)
        chunks = []
        
        for key, value in parsed.items():
            chunks.append({
                "chunk_id": f"chunk_{key}",
                "chunk_type": key,
                "chunk_data": {key: value}
            })
        
        log(f"Created {len(chunks)} chunks", "SUCCESS")
        
        return {
            "chunks": chunks,
            "processing_stats": {"total_chunks": len(chunks)}
        }
    except yaml.YAMLError as e:
        log(f"YAML parse error: {e}", "ERROR")
        return {"chunks": [], "processing_stats": {"error": str(e)}}


def dispatch_chunk_processors(state: PlatformState) -> list[Send]:
    """Dispatch chunks to parallel processors using Send API"""
    
    log(f"Dispatching {len(state['chunks'])} parallel chunk processors", "PARALLEL")
    
    sends = []
    for chunk in state["chunks"]:
        sends.append(Send("process_chunk", {
            "chunk_id": chunk["chunk_id"],
            "chunk_type": chunk["chunk_type"],
            "chunk_data": chunk["chunk_data"],
            "analysis": "",
            "insights": [],
            "recommendations": [],
            "risk_flags": [],
            "score": 0
        }))
    
    return sends


def process_chunk(state: ChunkState) -> dict:
    """Process individual chunk"""
    
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.3)
    
    prompts = {
        "departments": "Analyze organizational structure, headcount, budget allocation.",
        "employees": "Analyze performance distribution, promotion pipeline, retention risks.",
        "goals": "Analyze goal achievement, at-risk objectives, resource needs.",
        "compensation": "Analyze market competitiveness, equity, budget utilization.",
        "surveys": "Analyze engagement trends, concerns, action priorities.",
        "incidents": "Analyze patterns, systemic issues, risk mitigation.",
        "metadata": "Summarize key organizational statistics."
    }
    
    prompt = prompts.get(state["chunk_type"], "Analyze this data and extract insights.")
    
    full_prompt = f"""{prompt}

DATA:
```yaml
{yaml.dump(state['chunk_data'], default_flow_style=False)[:2000]}
```

Return JSON:
{{"analysis": "analysis", "insights": [], "recommendations": [], "risk_flags": [], "score": 1-10}}"""

    try:
        response = llm.invoke([HumanMessage(content=full_prompt)])
        data = extract_json(response.content)
        
        log(f"Chunk '{state['chunk_id']}' processed", "SUCCESS")
        
        return {
            "parallel_results": [{
                "chunk_id": state["chunk_id"],
                "chunk_type": state["chunk_type"],
                "analysis": data.get("analysis", response.content),
                "insights": data.get("insights", []),
                "recommendations": data.get("recommendations", []),
                "risk_flags": data.get("risk_flags", []),
                "score": data.get("score", 5)
            }]
        }
    except Exception as e:
        log(f"Chunk error: {e}", "ERROR")
        return {"parallel_results": []}


def synthesize_bulk_results(state: PlatformState) -> dict:
    """Synthesize all chunk results"""
    
    log(f"Synthesizing {len(state['parallel_results'])} chunk results", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    all_insights = []
    all_recommendations = []
    all_risks = []
    
    for r in state["parallel_results"]:
        all_insights.extend(r.get("insights", []))
        all_recommendations.extend(r.get("recommendations", []))
        all_risks.extend(r.get("risk_flags", []))
    
    summaries = "\n".join([
        f"**{r['chunk_type'].upper()}**: {r['analysis'][:200]}..."
        for r in state["parallel_results"]
    ])
    
    prompt = f"""Create an executive summary from this HR data analysis:

SECTION ANALYSES:
{summaries}

KEY INSIGHTS: {json.dumps(all_insights[:15])}
RECOMMENDATIONS: {json.dumps(all_recommendations[:10])}
RISKS: {json.dumps(all_risks[:10])}

Create:
1. Executive Summary (2-3 paragraphs)
2. Top 5 Action Items
3. Key Risks
4. Strategic Recommendations"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    stats = state.get("processing_stats", {})
    stats["chunks_processed"] = len(state["parallel_results"])
    stats["total_insights"] = len(all_insights)
    
    log("Synthesis complete", "SUCCESS")
    
    return {
        "executive_summary": response.content,
        "action_items": all_recommendations[:10],
        "final_analysis": {
            "by_section": {r["chunk_id"]: r for r in state["parallel_results"]},
            "all_insights": all_insights,
            "all_recommendations": all_recommendations,
            "all_risks": all_risks
        },
        "processing_stats": stats
    }


def format_bulk_output(state: PlatformState) -> dict:
    """Format bulk processing output"""
    
    stats = state.get("processing_stats", {})
    
    output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          HR DATA ANALYSIS REPORT                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 PROCESSING STATISTICS
   • Sections Processed: {stats.get('chunks_processed', 0)}
   • Insights Generated: {stats.get('total_insights', 0)}

{'═'*78}
 EXECUTIVE SUMMARY
{'═'*78}

{state.get('executive_summary', 'No summary generated.')}

{'═'*78}
🎯 TOP ACTION ITEMS
{'═'*78}

{chr(10).join(f"  {i+1}. {item}" for i, item in enumerate(state.get('action_items', [])[:10]))}

{'═'*78}
📊 SECTION DETAILS
{'═'*78}
"""
    
    for chunk_id, data in state.get("final_analysis", {}).get("by_section", {}).items():
        output += f"""
─── {data['chunk_type'].upper()} ───
{data['analysis'][:400]}...
"""
    
    return {"messages": [AIMessage(content=output)]}


# ============================================================
# PROMOTION ANALYZER (Full Pipeline)
# ============================================================

def run_promotion_analysis(state: PlatformState) -> dict:
    """Run full promotion analysis with parallelization + voting"""

    log("Running promotion analysis pipeline", "INFO")

    content = state["messages"][-1].content if state["messages"] else ""

    # Phase 1: Parallel sectioning analysis
    log("Phase 1: Parallel sectioning", "PARALLEL")
    analysis_results, analysis_stats = PARALLEL_ENGINE.run_analyzers(content)

    # Detect contradictions
    contradictions = AGGREGATOR.detect_contradictions(analysis_results)
    if contradictions:
        log(f"Found {len(contradictions)} contradictions", "WARN")

    # Create summary for voting
    summary = "\n".join([
        f"[{r.analyzer_id}]: {r.analysis[:150]}... (score: {r.score}/10)"
        for r in analysis_results
    ])

    # Phase 2: Parallel voting
    log("Phase 2: Parallel voting", "PARALLEL")
    vote_results, vote_stats = PARALLEL_ENGINE.run_voters(summary)

    # Phase 3: Aggregate
    log("Phase 3: Aggregation", "INFO")
    vote_aggregation = AGGREGATOR.weighted_vote(vote_results)

    return {
        "parallel_results": [
            {
                "analyzer_id": r.analyzer_id,
                "analysis": r.analysis,
                "insights": r.insights,
                "recommendations": r.recommendations,
                "risk_flags": r.risk_flags,
                "score": r.score
            }
            for r in analysis_results
        ],
        "vote_results": [
            {"voter_id": v.voter_id, "decision": v.decision, "confidence": v.confidence, "reasoning": v.reasoning}
            for v in vote_results
        ],
        "risk_assessment": vote_aggregation,
        "contradictions": contradictions,
        "processing_stats": {**analysis_stats, "voting": vote_stats}
    }


def format_promotion_output(state: PlatformState) -> dict:
    """Format promotion analysis output"""

    assessment = state.get("risk_assessment", {})
    stats = state.get("processing_stats", {})

    output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       PROMOTION READINESS ASSESSMENT                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 ANALYSIS STATISTICS
   • Analyzers Run: {len(state.get('parallel_results', []))}
   • Voters: {len(state.get('vote_results', []))}
   • Cache Hit Rate: {stats.get('cache_stats', {}).get('hit_rate', 0):.0%}
   • Contradictions: {len(state.get('contradictions', []))}

{'═'*78}
🗳️  VOTING RESULTS
{'═'*78}

   DECISION: {assessment.get('decision', 'N/A').upper()}
   CONFIDENCE: {assessment.get('confidence', 0):.0%}
   CONSENSUS: {assessment.get('consensus', 'N/A')}

   Breakdown:
"""
    
    for v in state.get("vote_results", []):
        output += f"     • {v['voter_id']}: {v['decision']} ({v['confidence']:.0%})\n"
    
    output += f"""
{'═'*78}
💪 ANALYSIS BY DIMENSION
{'═'*78}
"""
    
    for r in state.get("parallel_results", []):
        output += f"""
─── {r['analyzer_id'].upper()} (Score: {r['score']}/10) ───
{r['analysis'][:300]}...
Key Insights: {', '.join(r['insights'][:2]) if r['insights'] else 'None'}
"""
    
    if state.get("contradictions"):
        output += f"""
{'═'*78}
  CONTRADICTIONS DETECTED
{'═'*78}
"""
        for c in state["contradictions"]:
            output += f"   • {c['description']}\n"
    
    output += f"""
{'═'*78}
 RECOMMENDATION
{'═'*78}

Based on the weighted voting ({assessment.get('consensus', 'moderate')} consensus):

"""
    decision = assessment.get('decision', 'needs_development')
    if decision == "ready":
        output += " READY FOR PROMOTION - Proceed with nomination.\n"
    elif decision == "not_ready":
        output += " NOT YET READY - Create development plan and reassess in 6 months.\n"
    else:
        output += " NEEDS DEVELOPMENT - Address gaps before next review cycle.\n"
    
    return {"messages": [AIMessage(content=output)]}


# ============================================================
# ACTION PLANNING (Orchestrator-Workers Pattern)
# ============================================================

def orchestrator_analyze(state: PlatformState) -> dict:
    """Orchestrator analyzes the request and creates worker assignments"""
    
    log("Orchestrator analyzing request for action planning...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.3)
    
    # Get context from previous analysis or user request
    user_msg = state["messages"][-1].content if state["messages"] else ""
    prior_analysis = state.get("executive_summary", "")
    prior_recommendations = state.get("action_items", [])
    
    prompt = f"""You are an Action Planning Orchestrator.

Analyze this request and determine what kind of action plan is needed.

USER REQUEST:
{user_msg[:1500]}

{f"PRIOR ANALYSIS: {prior_analysis[:500]}" if prior_analysis else ""}
{f"PRIOR RECOMMENDATIONS: {json.dumps(prior_recommendations[:5])}" if prior_recommendations else ""}

Determine:
1. The main objective/goal to plan for
2. The scope (personal development, team initiative, career transition, etc.)
3. The timeframe (30 days, 90 days, 6 months, 1 year)
4. Which specialist workers should contribute

Available workers:
- goal_setter: Creates SMART goals with success criteria
- timeline_planner: Creates milestones and schedules
- resource_identifier: Identifies needed support and resources
- obstacle_navigator: Anticipates risks and creates contingencies
- accountability_designer: Designs tracking and check-ins

Return JSON:
{{
    "objective": "main goal being planned",
    "scope": "personal|team|career|project",
    "timeframe": "30_days|90_days|6_months|1_year",
    "context_summary": "brief context for workers",
    "worker_assignments": [
        {{"worker": "worker_id", "specific_task": "what this worker should focus on", "priority": 1-5}}
    ]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    data = extract_json(response.content)
    
    if not data:
        data = {
            "objective": user_msg[:100],
            "scope": "personal",
            "timeframe": "90_days",
            "context_summary": user_msg[:200],
            "worker_assignments": [
                {"worker": w, "specific_task": "Analyze and contribute", "priority": i+1}
                for i, w in enumerate(ACTION_PLAN_WORKERS.keys())
            ]
        }
    
    log(f"Orchestrator assigned {len(data.get('worker_assignments', []))} workers", "SUCCESS")
    
    return {
        "planning_context": json.dumps({
            "objective": data.get("objective", ""),
            "scope": data.get("scope", "personal"),
            "timeframe": data.get("timeframe", "90_days"),
            "context": data.get("context_summary", "")
        }),
        "worker_tasks": data.get("worker_assignments", [])
    }


def dispatch_action_workers(state: PlatformState) -> list[Send]:
    """Dispatch workers in parallel using Send API"""
    
    log(f"Dispatching {len(state['worker_tasks'])} action planning workers", "PARALLEL")
    
    sends = []
    context = json.loads(state.get("planning_context", "{}"))
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    for task in state["worker_tasks"]:
        worker_id = task.get("worker", "goal_setter")
        worker_config = ACTION_PLAN_WORKERS.get(worker_id, ACTION_PLAN_WORKERS["goal_setter"])
        
        sends.append(Send("execute_action_worker", {
            "worker_id": worker_id,
            "worker_prompt": worker_config["prompt"],
            "specific_task": task.get("specific_task", ""),
            "planning_context": context,
            "user_input": user_input,
            "priority": task.get("priority", 3)
        }))
    
    return sends


# Worker execution state
class ActionWorkerState(TypedDict):
    worker_id: str
    worker_prompt: str
    specific_task: str
    planning_context: dict
    user_input: str
    priority: int


def execute_action_worker(state: ActionWorkerState) -> dict:
    """Execute a single action planning worker"""
    
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.6)
    
    context = state["planning_context"]
    
    prompt = f"""{state['worker_prompt']}

CONTEXT:
- Objective: {context.get('objective', 'Not specified')}
- Scope: {context.get('scope', 'personal')}
- Timeframe: {context.get('timeframe', '90 days')}
- Additional Context: {context.get('context', '')}

SPECIFIC TASK:
{state['specific_task']}

USER INPUT:
{state['user_input'][:1000]}

Provide your specialist contribution. Return JSON:
{{
    "worker_id": "{state['worker_id']}",
    "contribution_type": "goals|timeline|resources|obstacles|accountability",
    "items": [
        {{"title": "item title", "description": "details", "priority": "high|medium|low", "details": {{}}}}
    ],
    "key_recommendations": ["rec1", "rec2"],
    "dependencies": ["any dependencies on other workers' output"]
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
        
        if not data:
            data = {
                "worker_id": state["worker_id"],
                "contribution_type": "general",
                "items": [{"title": "See analysis", "description": response.content[:500], "priority": "medium"}],
                "key_recommendations": [],
                "dependencies": []
            }
        
        log(f"Worker '{state['worker_id']}' completed with {len(data.get('items', []))} items", "SUCCESS")
        
        return {
            "worker_results": [{
                "worker_id": state["worker_id"],
                "contribution_type": data.get("contribution_type", "general"),
                "items": data.get("items", []),
                "recommendations": data.get("key_recommendations", []),
                "dependencies": data.get("dependencies", []),
                "priority": state["priority"]
            }]
        }
    except Exception as e:
        log(f"Worker '{state['worker_id']}' failed: {e}", "ERROR")
        return {"worker_results": []}


def synthesize_action_plan(state: PlatformState) -> dict:
    """Synthesize all worker outputs into cohesive action plan"""
    
    log(f"Synthesizing action plan from {len(state['worker_results'])} worker outputs", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    context = json.loads(state.get("planning_context", "{}"))
    
    # Organize worker outputs
    goals = []
    timeline = []
    resources = []
    obstacles = []
    accountability = []
    all_recommendations = []
    
    for result in state["worker_results"]:
        worker_id = result.get("worker_id", "")
        items = result.get("items", [])
        all_recommendations.extend(result.get("recommendations", []))
        
        if "goal" in worker_id:
            goals.extend(items)
        elif "timeline" in worker_id:
            timeline.extend(items)
        elif "resource" in worker_id:
            resources.extend(items)
        elif "obstacle" in worker_id:
            obstacles.extend(items)
        elif "accountability" in worker_id:
            accountability.extend(items)
    
    # Generate executive summary
    worker_summaries = "\n".join([
        f"**{r['worker_id'].upper()}**: {len(r.get('items', []))} items, Recommendations: {r.get('recommendations', [])[:2]}"
        for r in state["worker_results"]
    ])
    
    prompt = f"""Create an executive summary for this action plan.

OBJECTIVE: {context.get('objective', 'Not specified')}
TIMEFRAME: {context.get('timeframe', '90 days')}

WORKER CONTRIBUTIONS:
{worker_summaries}

GOALS: {json.dumps(goals[:5], indent=2)}
TIMELINE: {json.dumps(timeline[:5], indent=2)}
RESOURCES: {json.dumps(resources[:3], indent=2)}
OBSTACLES: {json.dumps(obstacles[:3], indent=2)}

Create a 2-3 paragraph executive summary that:
1. States the main objective and approach
2. Highlights the key milestones and success criteria
3. Notes critical resources and risks to manage
4. Emphasizes the accountability structure"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    action_plan = {
        "objective": context.get("objective", ""),
        "timeframe": context.get("timeframe", "90_days"),
        "scope": context.get("scope", "personal"),
        "executive_summary": response.content,
        "goals": goals,
        "timeline": timeline,
        "resources": resources,
        "obstacles": obstacles,
        "accountability": accountability,
        "recommendations": all_recommendations,
        "worker_count": len(state["worker_results"])
    }
    
    log("Action plan synthesis complete", "SUCCESS")
    
    return {
        "action_plan": action_plan,
        "executive_summary": response.content,
        "action_items": [g.get("title", "") for g in goals[:10]]
    }


def format_action_plan_output(state: PlatformState) -> dict:
    """Format the action plan for display"""
    
    plan = state.get("action_plan", {})
    
    output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ACTION PLAN                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📎 OBJECTIVE: {plan.get('objective', 'Not specified')}
⏱️  TIMEFRAME: {plan.get('timeframe', '90 days').replace('_', ' ')}
📊 SCOPE: {plan.get('scope', 'personal').title()}

{'═'*78}
 EXECUTIVE SUMMARY
{'═'*78}

{plan.get('executive_summary', 'No summary generated.')}

{'═'*78}
🎯 SMART GOALS
{'═'*78}
"""
    
    for i, goal in enumerate(plan.get('goals', [])[:5], 1):
        output += f"""
{i}. {goal.get('title', 'Goal')}
   {goal.get('description', '')[:200]}
   Priority: {goal.get('priority', 'medium').upper()}
"""
    
    output += f"""
{'═'*78}
 TIMELINE & MILESTONES
{'═'*78}
"""
    
    for item in plan.get('timeline', [])[:5]:
        output += f"\n   • {item.get('title', 'Milestone')}: {item.get('description', '')[:100]}"
    
    output += f"""

{'═'*78}
📚 RESOURCES NEEDED
{'═'*78}
"""
    
    for item in plan.get('resources', [])[:5]:
        output += f"\n   • {item.get('title', 'Resource')}: {item.get('description', '')[:80]}"
    
    output += f"""

{'═'*78}
  POTENTIAL OBSTACLES & MITIGATIONS
{'═'*78}
"""
    
    for item in plan.get('obstacles', [])[:5]:
        output += f"\n   • {item.get('title', 'Risk')}: {item.get('description', '')[:80]}"
    
    output += f"""

{'═'*78}
 ACCOUNTABILITY & CHECK-INS
{'═'*78}
"""
    
    for item in plan.get('accountability', [])[:5]:
        output += f"\n   • {item.get('title', 'Check-in')}: {item.get('description', '')[:80]}"
    
    output += f"""

{'═'*78}
 KEY RECOMMENDATIONS
{'═'*78}
"""
    
    for i, rec in enumerate(plan.get('recommendations', [])[:8], 1):
        output += f"\n   {i}. {rec}"
    
    output += f"""

{'═'*78}
📊 PLAN STATISTICS
{'═'*78}
   • Workers Consulted: {plan.get('worker_count', 0)}
   • Goals Defined: {len(plan.get('goals', []))}
   • Milestones: {len(plan.get('timeline', []))}
   • Resources Identified: {len(plan.get('resources', []))}
   • Risks Mapped: {len(plan.get('obstacles', []))}
"""
    
    return {"messages": [AIMessage(content=output)]}


# ============================================================
# AUTONOMOUS AGENT (Agent Pattern)
# ============================================================

def agent_think(state: PlatformState) -> dict:
    """Agent thinks about next action"""
    
    log("Agent thinking...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    prompt = f"""You are an autonomous HR coaching agent.

GOAL: Help with: {state["messages"][-1].content if state["messages"] else "unknown"}

OBSERVATIONS: {state.get("agent_observations", [])}
ACTIONS TAKEN: {state.get("agent_actions_taken", [])}

Available actions:
- search_knowledge(topic): Search HR policies
- analyze_situation(description): Analyze a workplace situation
- create_recommendation(topic): Create actionable recommendations
- respond(message): Send final response (use when ready)

Decide next action. Return JSON:
{{"thinking": "reasoning", "action": "action_name", "params": {{}}, "is_final": true/false}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    data = extract_json(response.content)
    
    log(f"Agent decided: {data.get('action', 'respond')}", "INFO")
    
    return {"agent_plan": [json.dumps(data)]}


def agent_act(state: PlatformState) -> dict:
    """Agent executes action"""
    
    try:
        plan = json.loads(state["agent_plan"][-1]) if state.get("agent_plan") else {}
    except:
        plan = {"action": "respond", "is_final": True}
    
    action = plan.get("action", "respond")
    params = plan.get("params", {})
    
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    if action == "search_knowledge":
        result = KnowledgeBase.search(params.get("topic", ""))
        observation = f"Search result: {json.dumps(result)}"
    elif action == "analyze_situation":
        response = llm.invoke([HumanMessage(content=f"Analyze: {params.get('description', '')}")])
        observation = f"Analysis: {response.content[:200]}"
    elif action == "create_recommendation":
        response = llm.invoke([HumanMessage(content=f"Create recommendations for: {params.get('topic', '')}")])
        observation = f"Recommendations: {response.content[:200]}"
    elif action == "respond":
        log("Agent completing with response", "SUCCESS")
        response = llm.invoke([HumanMessage(content=f"""
Generate helpful coaching response.

Goal: {state["messages"][-1].content if state["messages"] else ""}
Information gathered: {state.get("agent_observations", [])}

Provide warm, actionable response.""")])
        
        return {
            "messages": [AIMessage(content=response.content)],
            "agent_complete": True
        }
    else:
        observation = "Unknown action"
    
    return {
        "agent_observations": state.get("agent_observations", []) + [observation],
        "agent_actions_taken": state.get("agent_actions_taken", []) + [action],
        "agent_complete": plan.get("is_final", False)
    }


def should_agent_continue(state: PlatformState) -> Literal["think", "done"]:
    """Check if agent should continue"""
    if state.get("agent_complete") or len(state.get("agent_actions_taken", [])) >= 5:
        return "done"
    return "think"


def agent_done(state: PlatformState) -> dict:
    """Ensure agent has response"""
    if not any(isinstance(m, AIMessage) for m in state.get("messages", [])):
        return {"messages": [AIMessage(content="I've gathered information. How can I help further?")]}
    return {}


# ============================================================
# CULTURE AMP PERFORM FEATURES
# ============================================================

def perform_self_reflection(state: PlatformState) -> dict:
    """Guide employee through self-reflection process"""
    
    log("Starting self-reflection guidance...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.7)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    # Determine which reflection questions to focus on
    prompt = f"""You are a SELF-REFLECTION GUIDE helping an employee complete their performance self-reflection.

The employee said: "{user_msg[:1000]}"

SELF-REFLECTION QUESTIONS TO COVER:
{json.dumps(SELF_REFLECTION_QUESTIONS, indent=2)}

COMPETENCY FRAMEWORK:
{json.dumps({k: v['description'] for k, v in COMPETENCY_FRAMEWORK.items()}, indent=2)}

Your job:
1. Help them articulate their accomplishments with specific impact and metrics
2. Use the STAR format (Situation, Task, Action, Result)
3. Be encouraging but push for specificity
4. Help them connect their work to team/company goals
5. Suggest how to discuss challenges honestly without being overly self-critical

Provide:
- Suggested responses for 2-3 key reflection questions
- Tips for making their responses more impactful
- Example phrases they can use

Be warm, supportive, and help them showcase their best work."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_review_writing(state: PlatformState) -> dict:
    """Help managers write effective performance reviews"""
    
    log("Starting performance review writing assistance...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.6)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a PERFORMANCE REVIEW WRITING assistant helping a manager write an effective review.

The manager's input: "{user_msg[:1500]}"

PERFORMANCE RATING SCALE:
{json.dumps({k: v['label'] for k, v in PERFORMANCE_RATINGS.items()}, indent=2)}

COMPETENCY FRAMEWORK:
{json.dumps({k: v['description'] for k, v in COMPETENCY_FRAMEWORK.items()}, indent=2)}

CULTURE AMP VALUES:
{json.dumps({k: v['description'] for k, v in CULTURE_AMP_VALUES.items()}, indent=2)}

Help the manager write a review that:
1. Has a clear overall summary
2. Includes specific accomplishments with impact
3. Balances strengths and development areas
4. Provides actionable feedback using SBI (Situation-Behavior-Impact)
5. Suggests a rating that aligns with evidence
6. Includes forward-looking development suggestions

Structure your response as a draft review with sections:
- Overall Performance Summary
- Key Accomplishments & Impact
- Strengths Demonstrated
- Areas for Development
- Goals for Next Period
- Recommended Rating (with justification)

Be specific, balanced, and growth-oriented."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_goal_setting(state: PlatformState) -> dict:
    """Help set effective goals and OKRs"""
    
    log("Starting goal/OKR setting assistance...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.6)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a GOAL & OKR specialist helping create effective objectives.

User's context: "{user_msg[:1000]}"

OKR BEST PRACTICES:
- 3-5 objectives per quarter
- 2-4 key results per objective
- Key results should be measurable, not tasks
- Include committed (100% confidence) and stretch goals (70% confidence)
- Goals should align: Individual → Team → Company

GOOD KEY RESULT EXAMPLES:
- "Increase customer NPS from 40 to 55"
- "Reduce time-to-resolution from 48h to 24h"
- "Ship 3 features that drive 20% user engagement increase"

BAD KEY RESULT EXAMPLES (these are tasks, not results):
- "Launch new feature"
- "Have weekly meetings"
- "Write documentation"

Create 2-3 well-structured OKRs with:
1. Clear, inspiring Objective (qualitative)
2. 2-3 measurable Key Results (quantitative)
3. Alignment note (how it supports team/company goals)
4. Confidence level (committed vs stretch)

Format nicely for review and discussion."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_one_on_one(state: PlatformState) -> dict:
    """Generate 1:1 meeting agenda and talking points"""
    
    log("Generating 1:1 meeting agenda...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.6)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a 1:1 MEETING PREPARATION specialist.

Context from user: "{user_msg[:1000]}"

Create a structured 1:1 meeting agenda that includes:

 AGENDA STRUCTURE (30-45 min meeting):

1. CHECK-IN (5 min)
   - Personal/wellbeing check
   - Energy level and workload

2. PROGRESS & WINS (10 min)
   - Updates on goals and projects
   - Celebrate achievements
   - What they're proud of

3. CHALLENGES & SUPPORT (10 min)
   - Blockers and obstacles
   - Support needed from manager
   - Resource requests

4. FEEDBACK EXCHANGE (10 min)
   - Feedback for the employee
   - Feedback for the manager
   - Team dynamics

5. DEVELOPMENT & GROWTH (5 min)
   - Career conversation
   - Learning opportunities
   - Stretch assignments

6. ACTION ITEMS (5 min)
   - Clear next steps
   - Who does what by when

For each section, provide:
- Suggested talking points
- Good questions to ask
- Things to watch for

Make it practical and ready to use."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_feedback_request(state: PlatformState) -> dict:
    """Help craft effective feedback requests"""
    
    log("Crafting feedback request...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.6)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a FEEDBACK REQUEST specialist.

User's context: "{user_msg[:1000]}"

Help them request feedback effectively.

 BAD FEEDBACK REQUESTS:
- "How am I doing?" (too vague)
- "Any feedback?" (too open-ended)
- "What should I improve?" (puts burden on responder)

 GOOD FEEDBACK REQUESTS:
- "What's one thing I could do differently to improve [specific area]?"
- "How effective was my communication in the [specific project]?"
- "In [situation], what would you suggest I do differently?"
- "What strengths should I leverage more?"

Provide:
1. Suggested people to ask (based on context)
2. 3-4 specific, well-crafted questions
3. Best timing for the request
4. Template message they can send

Make it humble, specific, and growth-oriented."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_feedback_writing(state: PlatformState) -> dict:
    """Help write and improve feedback using SBI model"""
    
    log("Improving feedback with SBI model...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a FEEDBACK IMPROVER helping make feedback more effective.

User's draft feedback: "{user_msg[:1500]}"

SBI MODEL (Situation-Behavior-Impact):
- SITUATION: When and where did this happen?
- BEHAVIOR: What specifically did they do? (observable, not interpretation)
- IMPACT: What was the result or effect?

EXAMPLE TRANSFORMATION:
 Original: "You're not a good communicator."
 Improved: "In yesterday's client presentation (Situation), you spoke quickly and used technical jargon without explaining terms (Behavior). The client looked confused and asked several clarifying questions (Impact). For next time, I suggest slowing down and checking for understanding."

Analyze the feedback and provide:

1. EFFECTIVENESS SCORE (1-10)
   - Specificity: Is it concrete?
   - Actionability: Clear next steps?
   - Balance: Acknowledges positives?
   - SBI Structure: Follows the model?

2. IMPROVED VERSION
   - Rewritten using SBI
   - More specific and actionable
   - Balanced and growth-oriented

3. WHAT WAS CHANGED
   - Specific improvements made
   - Why these changes help

4. DELIVERY TIPS
   - How to deliver this feedback effectively"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_shoutout(state: PlatformState) -> dict:
    """Help craft meaningful recognition shoutouts"""
    
    log("Crafting recognition shoutout...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.7)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a RECOGNITION specialist helping craft meaningful shoutouts.

User's context: "{user_msg[:1000]}"

CULTURE AMP VALUES:
{json.dumps({k: v['description'] for k, v in CULTURE_AMP_VALUES.items()}, indent=2)}

 GENERIC SHOUTOUTS (avoid):
- "Great job!"
- "Thanks for your hard work!"
- "You're awesome!"

 MEANINGFUL SHOUTOUTS:
- Specific about what the person did
- Explains the impact of their contribution
- Connects to company values when relevant
- Feels genuine and personal

EXAMPLE:
🌟 Shoutout to @Sarah!

I want to recognize Sarah for her incredible work leading the Q3 customer migration. When we hit an unexpected data issue at 2am, Sarah rallied the team, kept everyone calm, and found a creative solution that got us back on track by morning. The client thanked us personally for the smooth experience.

This is a perfect example of "Learn faster through feedback" - Sarah quickly adapted when things went wrong and turned a potential disaster into a success.

Thank you for being such a rock for the team! 🙌

Provide:
1. Draft shoutout message (ready to post)
2. Which Culture Amp value it connects to
3. Suggested emoji/tone
4. Visibility recommendation (team/department/company)"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_calibration(state: PlatformState) -> dict:
    """Support manager calibration discussions"""
    
    log("Starting calibration analysis...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.4)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a CALIBRATION specialist helping managers align on fair ratings.

Performance data to analyze: "{user_msg[:2000]}"

PERFORMANCE RATING SCALE:
{json.dumps({k: {'label': v['label'], 'description': v['description']} for k, v in PERFORMANCE_RATINGS.items()}, indent=2)}

COMMON BIASES TO WATCH FOR:
- Recency bias: Overweighting recent events
- Halo effect: One strength colors entire review
- Central tendency: Rating everyone "meets expectations"
- Leniency/strictness: Consistently high or low rater
- Similar-to-me bias: Favoring those like yourself

CALIBRATION QUESTIONS:
1. Is this rating consistent with peers at the same level?
2. Is there sufficient evidence to support this rating?
3. Are we accounting for scope and complexity differences?
4. Have we considered the full review period?
5. Are we conflating potential with current performance?

Provide:

1. RATING ANALYSIS
   - Proposed rating and evidence summary
   - Comparison to level expectations

2. CONSISTENCY FLAGS
   - Any potential inconsistencies with peers
   - Evidence gaps to address

3. BIAS ALERTS
   - Potential biases to discuss
   - Questions to explore

4. DISCUSSION POINTS
   - Key topics for calibration meeting
   - Suggested talking points

5. RECOMMENDATION
   - Final rating recommendation
   - Confidence level
   - Any caveats

Be objective, data-driven, and fair."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


def perform_competency_assessment(state: PlatformState) -> dict:
    """Assess competencies against framework"""
    
    log("Running competency assessment...", "INFO")
    llm = ChatOpenAI(model=CONFIG.model, temperature=0.5)
    
    user_msg = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are a COMPETENCY ASSESSMENT specialist.

Employee information: "{user_msg[:1500]}"

COMPETENCY FRAMEWORK:
{json.dumps(COMPETENCY_FRAMEWORK, indent=2)}

LEVELS:
- Developing: Learning with guidance
- Performing: Consistently meets expectations
- Exceeding: Regularly exceeds expectations
- Leading: Sets the standard for others

For each competency, provide:

1. DELIVERS RESULTS
   - Current Level: [Developing/Performing/Exceeding/Leading]
   - Evidence: [Specific examples supporting the assessment]
   - Development Actions: [1-2 specific actions to grow]
   - Resources: [Training, experiences, mentors suggested]

2. BUILDS RELATIONSHIPS
   [Same structure]

3. DRIVES INNOVATION
   [Same structure]

4. DEVELOPS SELF AND OTHERS
   [Same structure]

5. COMMUNICATES EFFECTIVELY
   [Same structure]

SUMMARY:
- Overall Strengths: [Top 2 competencies]
- Priority Development Areas: [Top 2 areas to focus on]
- 90-Day Development Focus: [Specific recommendation]

Be developmental, specific, and actionable."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_response": response.content,
        "quality_scores": {},
        "optimization_iteration": 0
    }


# ============================================================
# BUILD THE ULTIMATE GRAPH
# ============================================================

def build_ultimate_platform():
    """Build the complete platform graph"""
    
    graph = StateGraph(PlatformState)
    
    # ─── Core Nodes ───
    graph.add_node("classify", classify_request)
    graph.add_node("retrieve", retrieve_knowledge)
    
    # ─── Specialist Coaches (Routing) ───
    for route in SPECIALIST_PROMPTS.keys():
        graph.add_node(f"specialist_{route}", specialist_coach)
    
    # ─── Quality Control (Evaluator-Optimizer) ───
    graph.add_node("evaluate", evaluate_response)
    graph.add_node("optimize", optimize_response)
    graph.add_node("finalize", finalize_response)
    
    # ─── Bulk Processing (Send API) ───
    graph.add_node("parse_bulk", parse_bulk_data)
    graph.add_node("process_chunk", process_chunk)
    graph.add_node("synthesize_bulk", synthesize_bulk_results)
    graph.add_node("format_bulk", format_bulk_output)
    
    # ─── Promotion Analysis ───
    graph.add_node("promotion_analyze", run_promotion_analysis)
    graph.add_node("format_promotion", format_promotion_output)
    
    # ─── Action Planning (Orchestrator-Workers) ───
    graph.add_node("orchestrator", orchestrator_analyze)
    graph.add_node("execute_action_worker", execute_action_worker)
    graph.add_node("synthesize_plan", synthesize_action_plan)
    graph.add_node("format_plan", format_action_plan_output)
    
    # ─── Autonomous Agent ───
    graph.add_node("agent_think", agent_think)
    graph.add_node("agent_act", agent_act)
    graph.add_node("agent_done", agent_done)
    
    # ─── Culture Amp Perform Features ───
    graph.add_node("perform_self_reflection", perform_self_reflection)
    graph.add_node("perform_review_writing", perform_review_writing)
    graph.add_node("perform_goal_setting", perform_goal_setting)
    graph.add_node("perform_one_on_one", perform_one_on_one)
    graph.add_node("perform_feedback_request", perform_feedback_request)
    graph.add_node("perform_feedback_writing", perform_feedback_writing)
    graph.add_node("perform_shoutout", perform_shoutout)
    graph.add_node("perform_calibration", perform_calibration)
    graph.add_node("perform_competency", perform_competency_assessment)
    
    # ═══ EDGES ═══
    
    # Start → Classify → Retrieve
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    
    # Routing based on request type
    def main_router(state: PlatformState) -> str:
        req_type = state.get("request_type", "conversation")
        route = state.get("route", "general")
        
        if req_type == RequestType.BULK_ANALYSIS.value:
            return "parse_bulk"
        elif req_type == RequestType.PROMOTION_REVIEW.value:
            return "promotion_analyze"
        elif req_type == RequestType.ACTION_PLANNING.value:
            return "orchestrator"
        elif req_type == RequestType.RISK_TRIAGE.value:
            return "agent_think"  # Use autonomous agent for complex triage
        # Culture Amp Perform Routes
        elif req_type == RequestType.SELF_REFLECTION.value:
            return "perform_self_reflection"
        elif req_type == RequestType.PERFORMANCE_REVIEW.value:
            return "perform_review_writing"
        elif req_type == RequestType.GOAL_SETTING.value:
            return "perform_goal_setting"
        elif req_type == RequestType.ONE_ON_ONE.value:
            return "perform_one_on_one"
        elif req_type == RequestType.FEEDBACK_REQUEST.value:
            return "perform_feedback_request"
        elif req_type == RequestType.FEEDBACK_WRITING.value:
            return "perform_feedback_writing"
        elif req_type == RequestType.SHOUTOUT.value:
            return "perform_shoutout"
        elif req_type == RequestType.CALIBRATION.value:
            return "perform_calibration"
        elif req_type == RequestType.COMPETENCY_ASSESSMENT.value:
            return "perform_competency"
        else:
            return f"specialist_{route}" if route in SPECIALIST_PROMPTS else "specialist_general"
    
    graph.add_conditional_edges("retrieve", main_router, {
        "parse_bulk": "parse_bulk",
        "promotion_analyze": "promotion_analyze",
        "orchestrator": "orchestrator",
        "agent_think": "agent_think",
        # Culture Amp Perform nodes
        "perform_self_reflection": "perform_self_reflection",
        "perform_review_writing": "perform_review_writing",
        "perform_goal_setting": "perform_goal_setting",
        "perform_one_on_one": "perform_one_on_one",
        "perform_feedback_request": "perform_feedback_request",
        "perform_feedback_writing": "perform_feedback_writing",
        "perform_shoutout": "perform_shoutout",
        "perform_calibration": "perform_calibration",
        "perform_competency": "perform_competency",
        **{f"specialist_{r}": f"specialist_{r}" for r in SPECIALIST_PROMPTS.keys()}
    })
    
    # Specialist → Evaluate → Optimize/Finalize
    for route in SPECIALIST_PROMPTS.keys():
        graph.add_edge(f"specialist_{route}", "evaluate")
    
    # Culture Amp Perform → Evaluate (quality control for all Perform features)
    perform_nodes = [
        "perform_self_reflection", "perform_review_writing", "perform_goal_setting",
        "perform_one_on_one", "perform_feedback_request", "perform_feedback_writing",
        "perform_shoutout", "perform_calibration", "perform_competency"
    ]
    for node in perform_nodes:
        graph.add_edge(node, "evaluate")
    
    graph.add_conditional_edges("evaluate", should_optimize, {
        "optimize": "optimize",
        "finalize": "finalize"
    })
    graph.add_edge("optimize", "evaluate")
    graph.add_edge("finalize", END)
    
    # Bulk Processing: Parse → Fan-out → Synthesize → Format
    graph.add_conditional_edges("parse_bulk", dispatch_chunk_processors, ["process_chunk"])
    graph.add_edge("process_chunk", "synthesize_bulk")
    graph.add_edge("synthesize_bulk", "format_bulk")
    graph.add_edge("format_bulk", END)
    
    # Promotion: Analyze → Format
    graph.add_edge("promotion_analyze", "format_promotion")
    graph.add_edge("format_promotion", END)
    
    # Action Planning: Orchestrator → Workers (parallel) → Synthesize → Format
    graph.add_conditional_edges("orchestrator", dispatch_action_workers, ["execute_action_worker"])
    graph.add_edge("execute_action_worker", "synthesize_plan")
    graph.add_edge("synthesize_plan", "format_plan")
    graph.add_edge("format_plan", END)
    
    # Agent: Think → Act → Continue/Done
    graph.add_edge("agent_think", "agent_act")
    graph.add_conditional_edges("agent_act", should_agent_continue, {
        "think": "agent_think",
        "done": "agent_done"
    })
    graph.add_edge("agent_done", END)
    
    # Compile with memory checkpointing
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# MAIN
# ============================================================

def main():
    """Demo the ultimate platform"""
    
    platform = build_ultimate_platform()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CULTURE AMP PERFORM - ULTIMATE HR COACHING PLATFORM                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COACHING                                                                    ║
║    1. Conversation  - Interactive coaching with specialist routing           ║
║    2. Promotion     - Full 360° analysis with parallel voting                ║
║    3. Action Plan   - Orchestrator-Workers for comprehensive planning        ║
║                                                                              ║
║  CULTURE AMP PERFORM                                                         ║
║    4. Self-Reflection  - Guided self-assessment with STAR format             ║
║    5. Goal Setting     - OKR creation with alignment                         ║
║    6. 1:1 Agenda       - Meeting preparation with talking points             ║
║    7. Feedback Writing - SBI-structured feedback improvement                 ║
║    8. Shoutout         - Values-driven recognition                           ║
║    9. Calibration      - Rating consistency analysis                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Demo 1: Conversation Coaching
    print("\n" + "═" * 78)
    print("DEMO 1: CONVERSATION COACHING (Routing + Evaluator-Optimizer)")
    print("═" * 78)
    
    config = {"configurable": {"thread_id": "demo-1"}}
    
    result = platform.invoke({
        "messages": [HumanMessage(content="I'm feeling burned out and thinking about quitting. My manager doesn't seem to notice.")],
        "request_type": "",
        "request_complexity": "",
        "identified_topics": [],
        "route": "",
        "route_confidence": 0.0,
        "retrieved_knowledge": {},
        "memory_context": "",
        "chunks": [],
        "parallel_results": [],
        "vote_results": [],
        "draft_response": "",
        "quality_scores": {},
        "optimization_iteration": 0,
        "worker_assignments": [],
        "worker_outputs": [],
        "final_analysis": {},
        "executive_summary": "",
        "action_items": [],
        "risk_assessment": {},
        "agent_plan": [],
        "agent_observations": [],
        "agent_actions_taken": [],
        "agent_complete": False,
        "planning_context": "",
        "worker_tasks": [],
        "worker_results": [],
        "action_plan": {},
        "processing_stats": {},
        "contradictions": [],
        "human_feedback": "",
        "perform_context": {},
        "reflection_responses": [],
        "feedback_items": [],
        "goals": [],
        "one_on_one_agenda": {},
        "shoutout_draft": {},
        "calibration_data": {},
        "competency_scores": {}
    }, config)
    
    print(result["messages"][-1].content)
    
    # Demo 2: Promotion Analysis
    print("\n" + "═" * 78)
    print("DEMO 2: PROMOTION ANALYSIS (Parallel Sectioning + Voting)")
    print("═" * 78)
    
    promotion_feedback = """
    360 FEEDBACK FOR PROMOTION REVIEW

    Manager: "Strong technical contributor, delivered major migration project.
    Needs to improve executive presence and cross-org influence."

    Peers: "Go-to person for hard problems. Could delegate more.
    Sometimes too detailed in reviews."

    Direct Reports: "Great mentor, explains things clearly.
    Would like more regular check-ins."

    Self: "Grown in system design. Want to expand influence beyond team."

    Metrics: 4 major projects, top 10% code review velocity, led 3 incidents.
    """
    
    config2 = {"configurable": {"thread_id": "demo-2"}}
    
    result2 = platform.invoke({
        "messages": [HumanMessage(content=promotion_feedback)],
        "request_type": RequestType.PROMOTION_REVIEW.value,
        "request_complexity": "high",
        "identified_topics": ["promotion"],
        "route": "career",
        "route_confidence": 0.9,
        "retrieved_knowledge": {},
        "memory_context": "",
        "chunks": [],
        "parallel_results": [],
        "vote_results": [],
        "draft_response": "",
        "quality_scores": {},
        "optimization_iteration": 0,
        "worker_assignments": [],
        "worker_outputs": [],
        "final_analysis": {},
        "executive_summary": "",
        "action_items": [],
        "risk_assessment": {},
        "agent_plan": [],
        "agent_observations": [],
        "agent_actions_taken": [],
        "agent_complete": False,
        "planning_context": "",
        "worker_tasks": [],
        "worker_results": [],
        "action_plan": {},
        "processing_stats": {},
        "contradictions": [],
        "human_feedback": "",
        "perform_context": {},
        "reflection_responses": [],
        "feedback_items": [],
        "goals": [],
        "one_on_one_agenda": {},
        "shoutout_draft": {},
        "calibration_data": {},
        "competency_scores": {}
    }, config2)
    
    print(result2["messages"][-1].content)
    
    # Demo 3: Action Planning
    print("\n" + "═" * 78)
    print("DEMO 3: ACTION PLANNING (Orchestrator-Workers Pattern)")
    print("═" * 78)
    
    config3 = {"configurable": {"thread_id": "demo-3"}}
    
    result3 = platform.invoke({
        "messages": [HumanMessage(content="""I want to transition from Senior Engineer to Engineering Manager.
I've been in my current role for 4 years with strong technical performance.
I've mentored 2 junior engineers but have no formal management experience.
I want to make this transition within the next 6-12 months.""")],
        "request_type": RequestType.ACTION_PLANNING.value,
        "request_complexity": "high",
        "identified_topics": ["career", "leadership"],
        "route": "career",
        "route_confidence": 0.9,
        "retrieved_knowledge": {},
        "memory_context": "",
        "chunks": [],
        "parallel_results": [],
        "vote_results": [],
        "draft_response": "",
        "quality_scores": {},
        "optimization_iteration": 0,
        "worker_assignments": [],
        "worker_outputs": [],
        "final_analysis": {},
        "executive_summary": "",
        "action_items": [],
        "risk_assessment": {},
        "agent_plan": [],
        "agent_observations": [],
        "agent_actions_taken": [],
        "agent_complete": False,
        "planning_context": "",
        "worker_tasks": [],
        "worker_results": [],
        "action_plan": {},
        "processing_stats": {},
        "contradictions": [],
        "human_feedback": "",
        "perform_context": {},
        "reflection_responses": [],
        "feedback_items": [],
        "goals": [],
        "one_on_one_agenda": {},
        "shoutout_draft": {},
        "calibration_data": {},
        "competency_scores": {}
    }, config3)
    
    print(result3["messages"][-1].content)
    
    # Demo 4: Culture Amp Perform - Feedback Writing
    print("\n" + "═" * 78)
    print("DEMO 4: CULTURE AMP PERFORM - FEEDBACK IMPROVEMENT (SBI Model)")
    print("═" * 78)
    
    config4 = {"configurable": {"thread_id": "demo-4"}}
    
    result4 = platform.invoke({
        "messages": [HumanMessage(content="""Help me improve this feedback I want to give:

"John needs to be better at communication. He's always late to meetings and doesn't
speak up enough. He also needs to work on his attitude."

I want to make this more constructive and actionable.""")],
        "request_type": RequestType.FEEDBACK_WRITING.value,
        "request_complexity": "medium",
        "identified_topics": ["feedback"],
        "route": "feedback",
        "route_confidence": 0.9,
        "retrieved_knowledge": {},
        "memory_context": "",
        "chunks": [],
        "parallel_results": [],
        "vote_results": [],
        "draft_response": "",
        "quality_scores": {},
        "optimization_iteration": 0,
        "worker_assignments": [],
        "worker_outputs": [],
        "final_analysis": {},
        "executive_summary": "",
        "action_items": [],
        "risk_assessment": {},
        "agent_plan": [],
        "agent_observations": [],
        "agent_actions_taken": [],
        "agent_complete": False,
        "planning_context": "",
        "worker_tasks": [],
        "worker_results": [],
        "action_plan": {},
        "processing_stats": {},
        "contradictions": [],
        "human_feedback": "",
        "perform_context": {},
        "reflection_responses": [],
        "feedback_items": [],
        "goals": [],
        "one_on_one_agenda": {},
        "shoutout_draft": {},
        "calibration_data": {},
        "competency_scores": {}
    }, config4)
    
    print(result4["messages"][-1].content)
    
    print("\n" + "═" * 78)
    print("PLATFORM CAPABILITIES SUMMARY")
    print("═" * 78)
    print("""
╭──────────────────────────────────────────────────────────────────────────────╮
│ ANTHROPIC PATTERNS IMPLEMENTED                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  Augmented LLM      - KnowledgeBase retrieval, Tools, ConversationMemory   │
│  Prompt Chaining    - Analyze → Generate → Refine in specialist coaches    │
│  Routing            - 8 specialist coaches based on topic classification   │
│  Parallelization    - ThreadPoolExecutor + sectioning + weighted voting    │
│  Orchestrator       - Worker delegation for complex tasks                  │
│  Evaluator-Optimizer- Quality scoring loop with configurable threshold     │
│  Autonomous Agent   - Think-Act-Observe loop for risk triage               │
├──────────────────────────────────────────────────────────────────────────────┤
│ CULTURE AMP PERFORM FEATURES                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  Self-Reflections   - STAR format guided self-assessment                   │
│  Performance Review - AI-assisted review writing for managers              │
│  Goals/OKRs         - Objective setting with alignment & key results       │
│  1-on-1 Meetings    - Agenda generation with talking points                │
│  Feedback Request   - Help crafting effective feedback questions           │
│  Feedback Writing   - SBI model improvement suggestions                    │
│  Shoutouts          - Values-driven recognition with Culture Amp values    │
│  Calibration        - Rating consistency & bias detection                  │
│  Competencies       - Skills assessment against framework                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ LANGGRAPH FEATURES                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  Send API           - Dynamic fan-out for bulk YAML processing             │
│  Automatic Fan-In   - Annotated[list, add] for collecting parallel results │
│  Sub-Graphs         - Chunk processor as nested workflow                   │
│  Checkpointing      - MemorySaver for conversation persistence             │
│  Conditional Edges  - Dynamic routing based on request classification      │
├──────────────────────────────────────────────────────────────────────────────┤
│ OPTIMIZATIONS                                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  Caching            - Hash-based result cache with hit rate tracking       │
│  Weighted Voting    - Confidence × expertise weight aggregation            │
│  Contradiction Detect- Cross-analyzer sentiment comparison                 │
│  Graceful Degrade   - Timeout handling, partial results                    │
│  Progress Tracking  - Real-time logging of parallel execution              │
╰──────────────────────────────────────────────────────────────────────────────╯
""")
    
    print(f"\n📊 Cache Statistics: {CACHE.stats()}")


if __name__ == "__main__":
    main()
