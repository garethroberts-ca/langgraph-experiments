#!/usr/bin/env python3
"""
LangGraph Example 37: Culture Amp EVOLVE Coaching Model
========================================================

This example implements Culture Amp's EVOLVE coaching framework as a
LangGraph workflow. EVOLVE is a structured approach to employee development
and performance coaching conversations.

EVOLVE stands for:
- Explore: Understand the current situation and context
- Vision: Define the desired future state and aspirations
- Options: Generate possible paths and strategies
- Learn: Identify skills, knowledge, and resources needed
- Validate: Check alignment and commitment
- Execute: Create actionable plans and next steps

This implementation demonstrates how LangGraph can model coaching
conversations, track progress through stages, and provide AI-assisted
guidance at each phase.

Author: LangGraph Examples
"""

import json
from datetime import datetime
from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# =============================================================================
# Helper Utilities
# =============================================================================

def print_banner(title: str) -> None:
    """Print a formatted section banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def log(message: str, indent: int = 0) -> None:
    """Print a log message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}> {message}")


def reduce_list(left: list | None, right: list | None) -> list:
    """Merge two lists, handling None values."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


# =============================================================================
# EVOLVE Model State Definition
# =============================================================================

class EVOLVEState(TypedDict):
    """State for the EVOLVE coaching conversation."""
    
    # Session metadata
    employee_name: str
    coach_name: str
    session_date: str
    session_topic: str
    
    # Stage completion tracking
    current_stage: str
    completed_stages: Annotated[list[str], reduce_list]
    
    # Explore stage outputs
    current_situation: str
    challenges: list[str]
    strengths: list[str]
    context_notes: str
    
    # Vision stage outputs
    desired_outcome: str
    success_indicators: list[str]
    timeframe: str
    motivation: str
    
    # Options stage outputs
    possible_approaches: list[dict]
    preferred_option: dict
    risks_and_mitigations: list[dict]
    
    # Learn stage outputs
    skills_to_develop: list[str]
    resources_needed: list[str]
    learning_plan: str
    support_required: str
    
    # Validate stage outputs
    alignment_score: int
    commitment_level: str
    potential_blockers: list[str]
    stakeholder_buy_in: str
    
    # Execute stage outputs
    action_items: list[dict]
    milestones: list[dict]
    check_in_schedule: str
    accountability_partner: str
    
    # AI-generated insights
    ai_insights: Annotated[list[str], reduce_list]
    coaching_notes: Annotated[list[str], reduce_list]
    
    # Session summary
    session_summary: str


# =============================================================================
# Demo 1: Full EVOLVE Coaching Session
# =============================================================================

def demo_full_evolve_session():
    """
    Demonstrate a complete EVOLVE coaching session workflow.
    
    Each stage of the EVOLVE model is implemented as a node in the graph,
    with AI assistance providing insights and recommendations.
    """
    print_banner("Demo 1: Full EVOLVE Coaching Session")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def explore_stage(state: EVOLVEState) -> dict:
        """
        EXPLORE: Understand the current situation and context.
        
        This stage focuses on:
        - Understanding where the employee is now
        - Identifying current challenges and obstacles
        - Recognising existing strengths and resources
        """
        log("EXPLORE: Understanding current situation...")
        
        # Simulate exploration conversation
        current_situation = (
            f"{state['employee_name']} is currently working as a senior developer "
            f"and has expressed interest in moving into a technical leadership role. "
            f"They have strong technical skills but limited experience with "
            f"people management and stakeholder communication."
        )
        
        challenges = [
            "Limited experience leading team meetings",
            "Tendency to focus on technical details over strategic thinking",
            "Discomfort with giving constructive feedback to peers",
            "Time management when balancing coding and leadership tasks"
        ]
        
        strengths = [
            "Deep technical expertise and credibility with the team",
            "Strong problem-solving abilities",
            "Respected by colleagues for mentoring junior developers",
            "Excellent written communication skills"
        ]
        
        # Get AI insights on the exploration
        explore_prompt = f"""As a coaching assistant, provide one brief insight about this employee's situation:

Employee: {state['employee_name']}
Topic: {state['session_topic']}
Situation: {current_situation}
Challenges: {', '.join(challenges)}
Strengths: {', '.join(strengths)}

Provide a single coaching insight (2-3 sentences) that connects their strengths to their challenges."""

        response = llm.invoke(explore_prompt)
        ai_insight = response.content.strip()
        
        log(f"Current situation assessed", indent=1)
        log(f"Identified {len(challenges)} challenges and {len(strengths)} strengths", indent=1)
        
        return {
            "current_stage": "explore_complete",
            "completed_stages": ["explore"],
            "current_situation": current_situation,
            "challenges": challenges,
            "strengths": strengths,
            "context_notes": "Employee shows high motivation and self-awareness",
            "ai_insights": [f"[Explore] {ai_insight}"]
        }
    
    def vision_stage(state: EVOLVEState) -> dict:
        """
        VISION: Define the desired future state and aspirations.
        
        This stage focuses on:
        - Articulating what success looks like
        - Setting meaningful goals
        - Understanding underlying motivations
        """
        log("VISION: Defining desired future state...")
        
        desired_outcome = (
            f"{state['employee_name']} wants to be recognised as an effective "
            f"technical leader who can guide team direction, mentor others, "
            f"and communicate effectively with both technical and non-technical "
            f"stakeholders whilst maintaining technical credibility."
        )
        
        success_indicators = [
            "Successfully leading weekly team planning sessions",
            "Receiving positive feedback on 1:1 conversations with direct reports",
            "Contributing to technical strategy discussions at leadership level",
            "Balancing hands-on coding with leadership responsibilities effectively"
        ]
        
        timeframe = "6 months"
        
        motivation = (
            "Driven by desire to have greater impact on product direction "
            "and to help grow the next generation of developers on the team."
        )
        
        # Get AI insights on the vision
        vision_prompt = f"""As a coaching assistant, provide one brief insight about this employee's vision:

Employee: {state['employee_name']}
Desired Outcome: {desired_outcome}
Success Indicators: {', '.join(success_indicators)}
Motivation: {motivation}

Provide a single coaching observation (2-3 sentences) about the clarity and achievability of this vision."""

        response = llm.invoke(vision_prompt)
        ai_insight = response.content.strip()
        
        log(f"Vision articulated with {len(success_indicators)} success indicators", indent=1)
        log(f"Timeframe: {timeframe}", indent=1)
        
        return {
            "current_stage": "vision_complete",
            "completed_stages": ["vision"],
            "desired_outcome": desired_outcome,
            "success_indicators": success_indicators,
            "timeframe": timeframe,
            "motivation": motivation,
            "ai_insights": [f"[Vision] {ai_insight}"]
        }
    
    def options_stage(state: EVOLVEState) -> dict:
        """
        OPTIONS: Generate possible paths and strategies.
        
        This stage focuses on:
        - Brainstorming multiple approaches
        - Evaluating pros and cons
        - Selecting preferred path forward
        """
        log("OPTIONS: Generating possible approaches...")
        
        possible_approaches = [
            {
                "name": "Gradual Transition",
                "description": "Take on leadership responsibilities incrementally whilst maintaining current role",
                "pros": ["Lower risk", "Time to develop skills", "Maintains technical credibility"],
                "cons": ["Slower progression", "May create role ambiguity"]
            },
            {
                "name": "Formal Leadership Programme",
                "description": "Enrol in company's leadership development programme with mentorship",
                "pros": ["Structured learning", "Networking opportunities", "Visible commitment"],
                "cons": ["Time investment", "May need manager approval"]
            },
            {
                "name": "Shadow and Learn",
                "description": "Shadow existing technical leads and gradually take on their responsibilities",
                "pros": ["Practical experience", "Direct mentorship", "Real-world learning"],
                "cons": ["Dependent on others' availability", "Less structured"]
            }
        ]
        
        # Select preferred option based on context
        preferred_option = possible_approaches[0]  # Gradual Transition
        
        risks_and_mitigations = [
            {
                "risk": "Spreading too thin between technical and leadership work",
                "mitigation": "Set clear boundaries and protected time for each type of work"
            },
            {
                "risk": "Team perceiving shift as abandoning technical contributions",
                "mitigation": "Communicate openly about transition and maintain code reviews"
            }
        ]
        
        # Get AI insights on options
        options_prompt = f"""As a coaching assistant, provide one brief recommendation:

Employee: {state['employee_name']}
Current situation: {state['current_situation']}
Options considered: {', '.join([o['name'] for o in possible_approaches])}
Preferred option: {preferred_option['name']}

Provide a single coaching recommendation (2-3 sentences) about this choice."""

        response = llm.invoke(options_prompt)
        ai_insight = response.content.strip()
        
        log(f"Generated {len(possible_approaches)} possible approaches", indent=1)
        log(f"Preferred option: {preferred_option['name']}", indent=1)
        
        return {
            "current_stage": "options_complete",
            "completed_stages": ["options"],
            "possible_approaches": possible_approaches,
            "preferred_option": preferred_option,
            "risks_and_mitigations": risks_and_mitigations,
            "ai_insights": [f"[Options] {ai_insight}"]
        }
    
    def learn_stage(state: EVOLVEState) -> dict:
        """
        LEARN: Identify skills, knowledge, and resources needed.
        
        This stage focuses on:
        - Gap analysis between current and desired state
        - Identifying learning opportunities
        - Planning skill development
        """
        log("LEARN: Identifying development needs...")
        
        skills_to_develop = [
            "Facilitation and meeting management",
            "Giving and receiving feedback effectively",
            "Strategic thinking and prioritisation",
            "Stakeholder management and communication",
            "Delegation and empowerment"
        ]
        
        resources_needed = [
            "Access to leadership development courses",
            "Regular mentorship sessions with current tech lead",
            "Books on technical leadership (e.g., 'The Manager's Path')",
            "Opportunities to lead small projects or initiatives"
        ]
        
        learning_plan = (
            "Month 1-2: Focus on facilitation skills through leading team retrospectives. "
            "Month 3-4: Work on feedback skills through structured 1:1s with junior developers. "
            "Month 5-6: Develop strategic thinking by participating in planning sessions."
        )
        
        support_required = (
            "Weekly check-ins with manager, access to leadership coaching, "
            "protected time for learning activities, and feedback from peers."
        )
        
        # Get AI insights on learning needs
        learn_prompt = f"""As a coaching assistant, provide one brief learning insight:

Employee: {state['employee_name']}
Skills to develop: {', '.join(skills_to_develop)}
Preferred approach: {state['preferred_option']['name']}

Provide a single coaching insight (2-3 sentences) about prioritising these learning needs."""

        response = llm.invoke(learn_prompt)
        ai_insight = response.content.strip()
        
        log(f"Identified {len(skills_to_develop)} skills to develop", indent=1)
        log(f"Identified {len(resources_needed)} resources needed", indent=1)
        
        return {
            "current_stage": "learn_complete",
            "completed_stages": ["learn"],
            "skills_to_develop": skills_to_develop,
            "resources_needed": resources_needed,
            "learning_plan": learning_plan,
            "support_required": support_required,
            "ai_insights": [f"[Learn] {ai_insight}"]
        }
    
    def validate_stage(state: EVOLVEState) -> dict:
        """
        VALIDATE: Check alignment and commitment.
        
        This stage focuses on:
        - Confirming alignment with goals and values
        - Assessing commitment level
        - Identifying potential blockers
        """
        log("VALIDATE: Checking alignment and commitment...")
        
        # Alignment assessment (1-10 scale)
        alignment_score = 8
        
        commitment_level = "High"
        
        potential_blockers = [
            "Upcoming major project deadline may limit initial availability",
            "Need to ensure manager supports the transition plan",
            "May face resistance from some team members initially"
        ]
        
        stakeholder_buy_in = (
            "Manager is supportive of development goals. Team lead has agreed "
            "to provide mentorship. HR has confirmed access to leadership programme."
        )
        
        # Get AI insights on validation
        validate_prompt = f"""As a coaching assistant, assess this development plan:

Employee: {state['employee_name']}
Alignment score: {alignment_score}/10
Commitment level: {commitment_level}
Potential blockers: {', '.join(potential_blockers)}

Provide a single coaching observation (2-3 sentences) about readiness to proceed."""

        response = llm.invoke(validate_prompt)
        ai_insight = response.content.strip()
        
        log(f"Alignment score: {alignment_score}/10", indent=1)
        log(f"Commitment level: {commitment_level}", indent=1)
        log(f"Identified {len(potential_blockers)} potential blockers", indent=1)
        
        return {
            "current_stage": "validate_complete",
            "completed_stages": ["validate"],
            "alignment_score": alignment_score,
            "commitment_level": commitment_level,
            "potential_blockers": potential_blockers,
            "stakeholder_buy_in": stakeholder_buy_in,
            "ai_insights": [f"[Validate] {ai_insight}"]
        }
    
    def execute_stage(state: EVOLVEState) -> dict:
        """
        EXECUTE: Create actionable plans and next steps.
        
        This stage focuses on:
        - Defining specific action items
        - Setting milestones and deadlines
        - Establishing accountability
        """
        log("EXECUTE: Creating action plan...")
        
        action_items = [
            {
                "action": "Schedule meeting with manager to discuss transition plan",
                "due_date": "This week",
                "status": "pending",
                "owner": state["employee_name"]
            },
            {
                "action": "Volunteer to lead next team retrospective",
                "due_date": "Next sprint",
                "status": "pending",
                "owner": state["employee_name"]
            },
            {
                "action": "Set up fortnightly mentorship sessions with tech lead",
                "due_date": "Within 2 weeks",
                "status": "pending",
                "owner": state["employee_name"]
            },
            {
                "action": "Enrol in facilitation skills workshop",
                "due_date": "This month",
                "status": "pending",
                "owner": state["employee_name"]
            },
            {
                "action": "Begin reading 'The Manager's Path'",
                "due_date": "Ongoing",
                "status": "pending",
                "owner": state["employee_name"]
            }
        ]
        
        milestones = [
            {
                "milestone": "Complete first team retrospective facilitation",
                "target_date": "End of Month 1",
                "success_criteria": "Positive feedback from at least 3 team members"
            },
            {
                "milestone": "Conduct structured 1:1s with 2 junior developers",
                "target_date": "End of Month 2",
                "success_criteria": "Both report feeling supported and clear on goals"
            },
            {
                "milestone": "Present technical recommendation to leadership",
                "target_date": "End of Month 4",
                "success_criteria": "Recommendation accepted or constructive feedback received"
            },
            {
                "milestone": "Complete leadership development programme module",
                "target_date": "End of Month 6",
                "success_criteria": "Certification or completion confirmation"
            }
        ]
        
        check_in_schedule = "Fortnightly coaching sessions, monthly progress review with manager"
        
        accountability_partner = "Current tech lead (Sarah) and direct manager (Michael)"
        
        # Get AI insights on execution plan
        execute_prompt = f"""As a coaching assistant, evaluate this action plan:

Employee: {state['employee_name']}
Action items: {len(action_items)} items defined
Milestones: {len(milestones)} milestones set
Timeframe: {state['timeframe']}

Provide a single coaching observation (2-3 sentences) about the plan's effectiveness."""

        response = llm.invoke(execute_prompt)
        ai_insight = response.content.strip()
        
        log(f"Created {len(action_items)} action items", indent=1)
        log(f"Defined {len(milestones)} milestones", indent=1)
        
        return {
            "current_stage": "execute_complete",
            "completed_stages": ["execute"],
            "action_items": action_items,
            "milestones": milestones,
            "check_in_schedule": check_in_schedule,
            "accountability_partner": accountability_partner,
            "ai_insights": [f"[Execute] {ai_insight}"]
        }
    
    def generate_summary(state: EVOLVEState) -> dict:
        """Generate comprehensive session summary."""
        log("Generating session summary...")
        
        summary_prompt = f"""Create a brief coaching session summary for:

Employee: {state['employee_name']}
Session Topic: {state['session_topic']}
Date: {state['session_date']}

Key Points:
- Current Situation: {state['current_situation'][:200]}...
- Vision: {state['desired_outcome'][:200]}...
- Chosen Approach: {state['preferred_option']['name']}
- Commitment Level: {state['commitment_level']}
- Action Items: {len(state['action_items'])} defined
- Milestones: {len(state['milestones'])} set

Provide a 3-4 sentence executive summary of this coaching session."""

        response = llm.invoke(summary_prompt)
        
        return {
            "current_stage": "complete",
            "session_summary": response.content.strip(),
            "coaching_notes": [f"Session completed successfully on {state['session_date']}"]
        }
    
    # Build the EVOLVE graph
    builder = StateGraph(EVOLVEState)
    
    # Add nodes for each EVOLVE stage
    builder.add_node("explore", explore_stage)
    builder.add_node("vision", vision_stage)
    builder.add_node("options", options_stage)
    builder.add_node("learn", learn_stage)
    builder.add_node("validate", validate_stage)
    builder.add_node("execute", execute_stage)
    builder.add_node("summarise", generate_summary)
    
    # Define the linear flow through EVOLVE stages
    builder.add_edge(START, "explore")
    builder.add_edge("explore", "vision")
    builder.add_edge("vision", "options")
    builder.add_edge("options", "learn")
    builder.add_edge("learn", "validate")
    builder.add_edge("validate", "execute")
    builder.add_edge("execute", "summarise")
    builder.add_edge("summarise", END)
    
    graph = builder.compile()
    
    # Run the coaching session
    initial_state = {
        "employee_name": "Alex Chen",
        "coach_name": "Jordan Smith",
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "session_topic": "Transition to Technical Leadership",
        "current_stage": "not_started",
        "completed_stages": [],
        "current_situation": "",
        "challenges": [],
        "strengths": [],
        "context_notes": "",
        "desired_outcome": "",
        "success_indicators": [],
        "timeframe": "",
        "motivation": "",
        "possible_approaches": [],
        "preferred_option": {},
        "risks_and_mitigations": [],
        "skills_to_develop": [],
        "resources_needed": [],
        "learning_plan": "",
        "support_required": "",
        "alignment_score": 0,
        "commitment_level": "",
        "potential_blockers": [],
        "stakeholder_buy_in": "",
        "action_items": [],
        "milestones": [],
        "check_in_schedule": "",
        "accountability_partner": "",
        "ai_insights": [],
        "coaching_notes": [],
        "session_summary": ""
    }
    
    result = graph.invoke(initial_state)
    
    # Display session results
    log("\n" + "=" * 60)
    log("EVOLVE COACHING SESSION COMPLETE")
    log("=" * 60)
    
    log(f"\nEmployee: {result['employee_name']}")
    log(f"Topic: {result['session_topic']}")
    log(f"Date: {result['session_date']}")
    
    log("\nStages Completed:")
    for stage in result['completed_stages']:
        log(f"  [{stage.upper()}]", indent=1)
    
    log("\nAI Insights:")
    for insight in result['ai_insights']:
        log(f"  {insight[:100]}...", indent=1)
    
    log("\nAction Items:")
    for item in result['action_items'][:3]:
        log(f"  - {item['action']} (Due: {item['due_date']})", indent=1)
    
    log("\nSession Summary:")
    log(f"  {result['session_summary']}", indent=1)


# =============================================================================
# Demo 2: EVOLVE with Conditional Routing
# =============================================================================

def demo_evolve_with_routing():
    """
    Demonstrate EVOLVE with conditional routing based on validation results.
    
    If alignment is low, the workflow routes back to earlier stages
    for refinement.
    """
    print_banner("Demo 2: EVOLVE with Conditional Routing")
    
    class RoutedEVOLVEState(TypedDict):
        employee_name: str
        session_topic: str
        current_stage: str
        alignment_score: int
        iteration_count: int
        refinement_notes: Annotated[list[str], reduce_list]
        final_plan_approved: bool
    
    def explore(state: RoutedEVOLVEState) -> dict:
        log(f"EXPLORE: Iteration {state.get('iteration_count', 1)}")
        return {"current_stage": "explore_done"}
    
    def vision(state: RoutedEVOLVEState) -> dict:
        log("VISION: Defining goals")
        return {"current_stage": "vision_done"}
    
    def options(state: RoutedEVOLVEState) -> dict:
        log("OPTIONS: Generating approaches")
        return {"current_stage": "options_done"}
    
    def learn(state: RoutedEVOLVEState) -> dict:
        log("LEARN: Identifying development needs")
        return {"current_stage": "learn_done"}
    
    def validate(state: RoutedEVOLVEState) -> dict:
        log("VALIDATE: Checking alignment")
        
        # Simulate alignment check - lower on first iteration
        iteration = state.get("iteration_count", 1)
        score = 5 if iteration == 1 else 8
        
        log(f"Alignment score: {score}/10", indent=1)
        
        return {
            "current_stage": "validate_done",
            "alignment_score": score
        }
    
    def refine(state: RoutedEVOLVEState) -> dict:
        log("REFINE: Adjusting plan based on feedback")
        
        iteration = state.get("iteration_count", 1)
        
        return {
            "iteration_count": iteration + 1,
            "refinement_notes": [f"Iteration {iteration}: Adjusted goals for better alignment"],
            "current_stage": "refine_done"
        }
    
    def execute(state: RoutedEVOLVEState) -> dict:
        log("EXECUTE: Finalising action plan")
        return {
            "current_stage": "complete",
            "final_plan_approved": True
        }
    
    def route_after_validation(state: RoutedEVOLVEState) -> Literal["refine", "execute"]:
        """Route based on alignment score."""
        if state["alignment_score"] < 7:
            log("Alignment below threshold - routing to refinement", indent=1)
            return "refine"
        else:
            log("Alignment acceptable - proceeding to execution", indent=1)
            return "execute"
    
    # Build graph with conditional routing
    builder = StateGraph(RoutedEVOLVEState)
    
    builder.add_node("explore", explore)
    builder.add_node("vision", vision)
    builder.add_node("options", options)
    builder.add_node("learn", learn)
    builder.add_node("validate", validate)
    builder.add_node("refine", refine)
    builder.add_node("execute", execute)
    
    builder.add_edge(START, "explore")
    builder.add_edge("explore", "vision")
    builder.add_edge("vision", "options")
    builder.add_edge("options", "learn")
    builder.add_edge("learn", "validate")
    builder.add_conditional_edges("validate", route_after_validation)
    builder.add_edge("refine", "vision")  # Loop back to vision for refinement
    builder.add_edge("execute", END)
    
    graph = builder.compile()
    
    result = graph.invoke({
        "employee_name": "Sam Wilson",
        "session_topic": "Career Progression",
        "current_stage": "not_started",
        "alignment_score": 0,
        "iteration_count": 1,
        "refinement_notes": [],
        "final_plan_approved": False
    })
    
    log("\nRouting Results:")
    log(f"Final alignment score: {result['alignment_score']}/10")
    log(f"Total iterations: {result['iteration_count']}")
    log(f"Plan approved: {result['final_plan_approved']}")
    
    if result['refinement_notes']:
        log("\nRefinement History:")
        for note in result['refinement_notes']:
            log(f"  - {note}", indent=1)


# =============================================================================
# Demo 3: EVOLVE Progress Tracker
# =============================================================================

def demo_evolve_progress_tracker():
    """
    Demonstrate an EVOLVE progress tracking system for multiple employees.
    
    This shows how to manage and track coaching progress across
    an organisation.
    """
    print_banner("Demo 3: EVOLVE Progress Tracker")
    
    class ProgressState(TypedDict):
        employees: list[dict]
        current_employee_index: int
        completed_assessments: list[dict]
        organisation_summary: dict
    
    def assess_employee(state: ProgressState) -> dict:
        """Assess current employee's EVOLVE progress."""
        idx = state["current_employee_index"]
        employee = state["employees"][idx]
        
        log(f"Assessing: {employee['name']}")
        
        # Calculate progress metrics
        stages_completed = employee.get("completed_stages", [])
        total_stages = 6
        progress_pct = len(stages_completed) / total_stages * 100
        
        assessment = {
            "employee_name": employee["name"],
            "progress_percentage": progress_pct,
            "stages_completed": stages_completed,
            "current_stage": employee.get("current_stage", "not_started"),
            "last_session_date": employee.get("last_session", "N/A"),
            "action_items_pending": employee.get("pending_actions", 0),
            "risk_level": "low" if progress_pct > 50 else "medium" if progress_pct > 25 else "high"
        }
        
        log(f"Progress: {progress_pct:.0f}% ({len(stages_completed)}/{total_stages} stages)", indent=1)
        log(f"Risk level: {assessment['risk_level']}", indent=1)
        
        return {
            "completed_assessments": [assessment],
            "current_employee_index": idx + 1
        }
    
    def should_continue(state: ProgressState) -> Literal["assess", "summarise"]:
        """Check if there are more employees to assess."""
        if state["current_employee_index"] < len(state["employees"]):
            return "assess"
        return "summarise"
    
    def generate_org_summary(state: ProgressState) -> dict:
        """Generate organisation-wide summary."""
        log("Generating organisation summary...")
        
        assessments = state["completed_assessments"]
        
        total_employees = len(assessments)
        avg_progress = sum(a["progress_percentage"] for a in assessments) / total_employees
        
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for a in assessments:
            risk_counts[a["risk_level"]] += 1
        
        summary = {
            "total_employees_tracked": total_employees,
            "average_progress": f"{avg_progress:.1f}%",
            "risk_distribution": risk_counts,
            "employees_requiring_attention": [
                a["employee_name"] for a in assessments if a["risk_level"] == "high"
            ],
            "assessment_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        return {"organisation_summary": summary}
    
    # Build progress tracker graph
    builder = StateGraph(ProgressState)
    
    builder.add_node("assess", assess_employee)
    builder.add_node("summarise", generate_org_summary)
    
    builder.add_edge(START, "assess")
    builder.add_conditional_edges("assess", should_continue)
    builder.add_edge("summarise", END)
    
    graph = builder.compile()
    
    # Sample employee data
    employees = [
        {
            "name": "Alice Johnson",
            "completed_stages": ["explore", "vision", "options", "learn"],
            "current_stage": "validate",
            "last_session": "2024-03-15",
            "pending_actions": 2
        },
        {
            "name": "Bob Smith",
            "completed_stages": ["explore"],
            "current_stage": "vision",
            "last_session": "2024-03-01",
            "pending_actions": 5
        },
        {
            "name": "Carol Davis",
            "completed_stages": ["explore", "vision", "options", "learn", "validate", "execute"],
            "current_stage": "complete",
            "last_session": "2024-03-18",
            "pending_actions": 0
        },
        {
            "name": "David Lee",
            "completed_stages": ["explore", "vision"],
            "current_stage": "options",
            "last_session": "2024-03-10",
            "pending_actions": 3
        }
    ]
    
    result = graph.invoke({
        "employees": employees,
        "current_employee_index": 0,
        "completed_assessments": [],
        "organisation_summary": {}
    })
    
    log("\nORGANISATION SUMMARY")
    log("=" * 40)
    summary = result["organisation_summary"]
    log(f"Total employees tracked: {summary['total_employees_tracked']}")
    log(f"Average progress: {summary['average_progress']}")
    log(f"Risk distribution: {summary['risk_distribution']}")
    
    if summary["employees_requiring_attention"]:
        log("\nEmployees requiring attention:")
        for name in summary["employees_requiring_attention"]:
            log(f"  - {name}", indent=1)


# =============================================================================
# Demo 4: EVOLVE Coaching Conversation Simulator
# =============================================================================

def demo_evolve_conversation():
    """
    Simulate an interactive EVOLVE coaching conversation using LLM.
    
    This demonstrates how to create AI-powered coaching dialogues
    that follow the EVOLVE framework.
    """
    print_banner("Demo 4: EVOLVE Coaching Conversation")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    class ConversationState(TypedDict):
        employee_context: str
        current_stage: str
        conversation_history: Annotated[list[dict], reduce_list]
        stage_outputs: dict
    
    def generate_coaching_prompt(stage: str, context: str) -> str:
        """Generate stage-specific coaching prompt."""
        prompts = {
            "explore": f"""You are a skilled coach conducting an EVOLVE coaching session.
You are in the EXPLORE stage - understanding the current situation.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Asks about current challenges
2. Explores existing strengths
3. Summarises understanding

Format as:
Coach: [question]
Employee: [response]
Coach: [follow-up or summary]""",

            "vision": f"""You are conducting the VISION stage of EVOLVE coaching.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Asks about desired future state
2. Explores what success looks like
3. Clarifies motivation

Format as:
Coach: [question]
Employee: [response]
Coach: [follow-up or summary]""",

            "options": f"""You are conducting the OPTIONS stage of EVOLVE coaching.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Brainstorms possible approaches
2. Evaluates pros and cons
3. Helps select preferred path

Format as:
Coach: [question]
Employee: [response]
Coach: [follow-up or summary]""",

            "learn": f"""You are conducting the LEARN stage of EVOLVE coaching.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Identifies skills gaps
2. Explores learning resources
3. Plans development activities

Format as:
Coach: [question]
Employee: [response]
Coach: [follow-up or summary]""",

            "validate": f"""You are conducting the VALIDATE stage of EVOLVE coaching.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Checks commitment level
2. Identifies potential blockers
3. Confirms readiness to proceed

Format as:
Coach: [question]
Employee: [response]
Coach: [follow-up or summary]""",

            "execute": f"""You are conducting the EXECUTE stage of EVOLVE coaching.

Employee context: {context}

Generate a brief coaching exchange (2-3 turns) that:
1. Defines specific action items
2. Sets timelines
3. Establishes accountability

Format as:
Coach: [question]
Employee: [response]
Coach: [summary and next steps]"""
        }
        return prompts.get(stage, prompts["explore"])
    
    def conduct_stage(state: ConversationState, stage: str) -> dict:
        """Conduct a single EVOLVE stage conversation."""
        log(f"{stage.upper()}: Conducting coaching conversation...")
        
        prompt = generate_coaching_prompt(stage, state["employee_context"])
        response = llm.invoke(prompt)
        
        conversation = response.content.strip()
        
        # Parse and display conversation
        lines = conversation.split("\n")
        for line in lines[:6]:  # Limit display
            if line.strip():
                log(line, indent=1)
        
        return {
            "current_stage": f"{stage}_complete",
            "conversation_history": [{"stage": stage, "dialogue": conversation}],
            "stage_outputs": {**state.get("stage_outputs", {}), stage: "completed"}
        }
    
    def explore_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "explore")
    
    def vision_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "vision")
    
    def options_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "options")
    
    def learn_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "learn")
    
    def validate_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "validate")
    
    def execute_conversation(state: ConversationState) -> dict:
        return conduct_stage(state, "execute")
    
    # Build conversation graph
    builder = StateGraph(ConversationState)
    
    builder.add_node("explore", explore_conversation)
    builder.add_node("vision", vision_conversation)
    builder.add_node("options", options_conversation)
    builder.add_node("learn", learn_conversation)
    builder.add_node("validate", validate_conversation)
    builder.add_node("execute", execute_conversation)
    
    builder.add_edge(START, "explore")
    builder.add_edge("explore", "vision")
    builder.add_edge("vision", "options")
    builder.add_edge("options", "learn")
    builder.add_edge("learn", "validate")
    builder.add_edge("validate", "execute")
    builder.add_edge("execute", END)
    
    graph = builder.compile()
    
    result = graph.invoke({
        "employee_context": (
            "Mid-level software engineer with 4 years experience, "
            "interested in becoming a team lead. Strong technical skills, "
            "needs to develop people management and communication abilities."
        ),
        "current_stage": "not_started",
        "conversation_history": [],
        "stage_outputs": {}
    })
    
    log("\nConversation Summary:")
    log(f"Stages completed: {len(result['conversation_history'])}")
    log(f"Stage outputs: {list(result['stage_outputs'].keys())}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all EVOLVE coaching model demonstrations."""
    print("\n" + "=" * 70)
    print(" Culture Amp EVOLVE Coaching Model ".center(70))
    print(" Explore - Vision - Options - Learn - Validate - Execute ".center(70))
    print("=" * 70)
    
    demo_full_evolve_session()
    demo_evolve_with_routing()
    demo_evolve_progress_tracker()
    demo_evolve_conversation()
    
    print("\n" + "=" * 70)
    print(" All EVOLVE Demonstrations Complete ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
