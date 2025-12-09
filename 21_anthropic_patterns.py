"""
21. Anthropic's Agentic Patterns
================================
Implementation of all 7 patterns from Anthropic's "Building Effective Agents" blog.
Each pattern is demonstrated in an HR coaching context.

Patterns:
1. Augmented LLM - Base building block with retrieval, tools, memory
2. Prompt Chaining - Sequential steps where each uses previous output
3. Routing - Classify input and direct to specialized handlers
4. Parallelization - Run multiple LLM calls in parallel (sectioning/voting)
5. Orchestrator-Workers - Central LLM delegates to workers dynamically
6. Evaluator-Optimizer - One LLM generates, another evaluates/improves
7. Autonomous Agent - Full agentic loop with environment feedback

Reference: https://www.anthropic.com/engineering/building-effective-agents
"""

import asyncio
import json
from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool


print("=" * 70)
print("ANTHROPIC'S AGENTIC PATTERNS - HR COACHING IMPLEMENTATIONS")
print("=" * 70)


# ============================================================
# PATTERN 1: AUGMENTED LLM
# ============================================================
# The basic building block - an LLM enhanced with:
# - Retrieval (knowledge/context)
# - Tools (actions)
# - Memory (state persistence)
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 1: AUGMENTED LLM")
print("=" * 70)
print("""
The foundational building block. An LLM enhanced with:
- Retrieval: Access to knowledge bases and context
- Tools: Ability to take actions
- Memory: Persistence across interactions
""")

# Knowledge base (retrieval)
HR_KNOWLEDGE = {
    "promotion": "Promotion criteria: 1) Consistent high performance 2) Leadership demonstration 3) Cross-team impact 4) Manager nomination during review cycle",
    "feedback": "Effective feedback is: Specific, Timely, Actionable, and Balanced. Use SBI model: Situation-Behavior-Impact",
    "goals": "SMART goals: Specific, Measurable, Achievable, Relevant, Time-bound. Align with OKR framework.",
    "conflict": "Conflict resolution steps: 1) Listen to all parties 2) Identify underlying needs 3) Find common ground 4) Agree on next steps",
    "wellbeing": "Burnout warning signs: exhaustion, cynicism, reduced efficacy. Resources: EAP (1-800-EAP-HELP), manager 1:1, HR support"
}

# Tools
@tool
def search_hr_knowledge(topic: str) -> str:
    """Search HR knowledge base for policies and guidance."""
    for key, value in HR_KNOWLEDGE.items():
        if key in topic.lower():
            return value
    return "No specific policy found. Please consult HR directly."

@tool
def create_action_item(description: str, due_date: str = None) -> str:
    """Create an action item for the employee."""
    return f" Action created: {description}" + (f" (due: {due_date})" if due_date else "")

@tool
def schedule_followup(topic: str, days_from_now: int = 7) -> str:
    """Schedule a follow-up coaching session."""
    return f" Follow-up scheduled in {days_from_now} days to discuss: {topic}"

# Memory store
class CoachingMemory:
    def __init__(self):
        self.sessions = []
        self.user_context = {}
    
    def add_session(self, summary: str):
        self.sessions.append({"date": datetime.now().isoformat(), "summary": summary})
    
    def get_context(self) -> str:
        if not self.sessions:
            return "No prior sessions."
        return "\n".join([f"- {s['date'][:10]}: {s['summary']}" for s in self.sessions[-3:]])

MEMORY = CoachingMemory()


class AugmentedLLMState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_knowledge: str
    tool_results: list[str]


def pattern1_augmented_llm():
    """Demonstrate Pattern 1: Augmented LLM"""
    
    def retrieve_and_respond(state: AugmentedLLMState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        user_msg = state["messages"][-1].content
        
        # RETRIEVAL: Search knowledge base
        retrieved = ""
        for key in HR_KNOWLEDGE:
            if key in user_msg.lower():
                retrieved = HR_KNOWLEDGE[key]
                break
        
        # MEMORY: Get prior context
        memory_context = MEMORY.get_context()
        
        prompt = f"""You are an HR coach with access to:

RETRIEVED KNOWLEDGE: {retrieved or 'No specific policy found'}
PRIOR SESSIONS: {memory_context}

User question: {user_msg}

Provide helpful coaching. Reference the knowledge when relevant.
If appropriate, suggest creating an action item or scheduling follow-up."""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # MEMORY: Store this interaction
        MEMORY.add_session(f"Discussed: {user_msg[:50]}...")
        
        return {"messages": [response], "retrieved_knowledge": retrieved}
    
    graph = StateGraph(AugmentedLLMState)
    graph.add_node("coach", retrieve_and_respond)
    graph.add_edge(START, "coach")
    graph.add_edge("coach", END)
    return graph.compile()


# Demo Pattern 1
def demo_pattern1():
    coach = pattern1_augmented_llm()
    result = coach.invoke({
        "messages": [HumanMessage(content="How do I get promoted here?")],
        "retrieved_knowledge": "",
        "tool_results": []
    })
    print(f"User: How do I get promoted here?")
    print(f"Coach: {result['messages'][-1].content[:300]}...")
    print(f"[Retrieved: {result['retrieved_knowledge'][:100]}...]")


# ============================================================
# PATTERN 2: PROMPT CHAINING
# ============================================================
# Decompose task into sequential steps where each LLM call
# processes the output of the previous one.
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 2: PROMPT CHAINING")
print("=" * 70)
print("""
Sequential processing where each step builds on the previous.
Example: Analyze situation → Generate advice → Create action plan
""")


class PromptChainState(TypedDict):
    messages: Annotated[list, add_messages]
    situation_analysis: str
    coaching_advice: str
    action_plan: str


def pattern2_prompt_chaining():
    """Demonstrate Pattern 2: Prompt Chaining"""
    
    def step1_analyze(state: PromptChainState) -> dict:
        """Step 1: Analyze the situation"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        user_msg = state["messages"][-1].content
        
        prompt = f"""Analyze this workplace situation. Identify:
1. The core issue
2. Stakeholders involved
3. Emotional state of the person
4. Underlying needs

Situation: {user_msg}

Provide a structured analysis (2-3 sentences per point)."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"situation_analysis": response.content}
    
    def step2_advise(state: PromptChainState) -> dict:
        """Step 2: Generate coaching advice based on analysis"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        prompt = f"""Based on this situation analysis, provide coaching advice.

ANALYSIS:
{state['situation_analysis']}

Provide:
1. Validation of their feelings
2. Reframe/perspective shift
3. Specific recommendations
4. Potential pitfalls to avoid

Be warm and supportive but practical."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"coaching_advice": response.content}
    
    def step3_action_plan(state: PromptChainState) -> dict:
        """Step 3: Create concrete action plan"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        
        prompt = f"""Create a concrete action plan based on this advice.

ADVICE:
{state['coaching_advice']}

Create 3-5 specific, actionable steps with:
- Clear description
- Timeline (this week, next 2 weeks, this month)
- Success indicator

Format as a numbered list."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"action_plan": response.content}
    
    def step4_synthesize(state: PromptChainState) -> dict:
        """Step 4: Combine into final response"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        prompt = f"""Synthesize this into a cohesive coaching response.

ANALYSIS: {state['situation_analysis'][:300]}...
ADVICE: {state['coaching_advice'][:300]}...
ACTION PLAN: {state['action_plan']}

Create a warm, supportive response that:
1. Briefly acknowledges their situation
2. Provides key insight
3. Shares the action plan
4. Ends with encouragement"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    
    graph = StateGraph(PromptChainState)
    graph.add_node("analyze", step1_analyze)
    graph.add_node("advise", step2_advise)
    graph.add_node("plan", step3_action_plan)
    graph.add_node("synthesize", step4_synthesize)
    
    # Chain: analyze → advise → plan → synthesize
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "advise")
    graph.add_edge("advise", "plan")
    graph.add_edge("plan", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def demo_pattern2():
    coach = pattern2_prompt_chaining()
    result = coach.invoke({
        "messages": [HumanMessage(content="My manager never gives me feedback and I don't know if I'm doing well")],
        "situation_analysis": "",
        "coaching_advice": "",
        "action_plan": ""
    })
    print(f"User: My manager never gives me feedback...")
    print(f"\n[Step 1 - Analysis]: {result['situation_analysis'][:150]}...")
    print(f"\n[Step 2 - Advice]: {result['coaching_advice'][:150]}...")
    print(f"\n[Step 3 - Plan]: {result['action_plan'][:150]}...")
    print(f"\nFinal Response: {result['messages'][-1].content[:300]}...")


# ============================================================
# PATTERN 3: ROUTING
# ============================================================
# Classify input and direct to specialized handlers.
# Each handler is optimized for its specific domain.
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 3: ROUTING")
print("=" * 70)
print("""
Classify input and route to specialized handlers.
Example: Career questions → Career coach, Conflict → Conflict coach
""")


class RoutingState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    confidence: float


def pattern3_routing():
    """Demonstrate Pattern 3: Routing"""
    
    def classify_route(state: RoutingState) -> dict:
        """Classify the input to determine routing"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        user_msg = state["messages"][-1].content
        
        prompt = f"""Classify this HR coaching request into ONE category.

Categories:
- career: Promotion, career growth, skill development, job transitions
- conflict: Interpersonal issues, difficult conversations, team dynamics
- wellbeing: Stress, burnout, work-life balance, mental health
- feedback: Giving/receiving feedback, performance reviews
- general: Other coaching topics

Message: {user_msg}

Return JSON only: {{"route": "category", "confidence": 0.0-1.0}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            parsed = json.loads(response.content)
            return {"route": parsed["route"], "confidence": parsed.get("confidence", 0.8)}
        except:
            return {"route": "general", "confidence": 0.5}
    
    def career_coach(state: RoutingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""You are a CAREER DEVELOPMENT specialist coach.

Focus on: promotions, skill building, career paths, visibility, sponsorship.
Be strategic and action-oriented.

User: {state["messages"][-1].content}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=f"[Career Coach]\n{response.content}")]}
    
    def conflict_coach(state: RoutingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""You are a CONFLICT RESOLUTION specialist coach.

Focus on: understanding perspectives, de-escalation, difficult conversations.
Stay neutral and help them see other viewpoints.

User: {state["messages"][-1].content}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=f"[Conflict Coach]\n{response.content}")]}
    
    def wellbeing_coach(state: RoutingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""You are a WELLBEING specialist coach.

Focus on: stress management, boundaries, burnout prevention, self-care.
Be warm, validating, and mention EAP if appropriate.

User: {state["messages"][-1].content}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=f"[Wellbeing Coach]\n{response.content}")]}
    
    def feedback_coach(state: RoutingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""You are a FEEDBACK specialist coach.

Focus on: giving effective feedback, receiving feedback, performance conversations.
Teach SBI model (Situation-Behavior-Impact).

User: {state["messages"][-1].content}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=f"[Feedback Coach]\n{response.content}")]}
    
    def general_coach(state: RoutingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""You are a supportive HR coach.

Provide helpful guidance on the user's question.

User: {state["messages"][-1].content}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=f"[General Coach]\n{response.content}")]}
    
    def route_to_specialist(state: RoutingState) -> str:
        return state["route"]
    
    graph = StateGraph(RoutingState)
    
    graph.add_node("classify", classify_route)
    graph.add_node("career", career_coach)
    graph.add_node("conflict", conflict_coach)
    graph.add_node("wellbeing", wellbeing_coach)
    graph.add_node("feedback", feedback_coach)
    graph.add_node("general", general_coach)
    
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_to_specialist, {
        "career": "career",
        "conflict": "conflict",
        "wellbeing": "wellbeing",
        "feedback": "feedback",
        "general": "general"
    })
    
    for node in ["career", "conflict", "wellbeing", "feedback", "general"]:
        graph.add_edge(node, END)
    
    return graph.compile()


def demo_pattern3():
    coach = pattern3_routing()
    
    queries = [
        "How do I get promoted?",
        "My coworker and I keep clashing",
        "I'm completely burned out"
    ]
    
    for q in queries:
        result = coach.invoke({"messages": [HumanMessage(content=q)], "route": "", "confidence": 0.0})
        print(f"\nUser: {q}")
        print(f"Route: {result['route']} (confidence: {result['confidence']:.0%})")
        print(f"Response: {result['messages'][-1].content[:200]}...")


# ============================================================
# PATTERN 4: PARALLELIZATION
# ============================================================
# Run multiple LLM calls in parallel, then aggregate.
# Two variants: Sectioning (divide work) and Voting (multiple perspectives)
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 4: PARALLELIZATION")
print("=" * 70)
print("""
Run multiple LLM calls in parallel:
- Sectioning: Divide task into parts, process simultaneously
- Voting: Get multiple perspectives, aggregate for confidence
""")


class ParallelState(TypedDict):
    messages: Annotated[list, add_messages]
    # Sectioning outputs
    strengths_analysis: str
    growth_analysis: str
    action_analysis: str
    # Voting outputs
    votes: list[dict]
    consensus: str


def pattern4_parallelization():
    """Demonstrate Pattern 4: Parallelization (both sectioning and voting)"""
    
    # SECTIONING: Analyze feedback from multiple angles simultaneously
    def analyze_strengths(state: ParallelState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        prompt = f"""Analyze ONLY the STRENGTHS mentioned in this feedback.
        
Feedback: {state["messages"][-1].content}

List 2-3 key strengths with evidence."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"strengths_analysis": response.content}
    
    def analyze_growth(state: ParallelState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        prompt = f"""Analyze ONLY the GROWTH AREAS mentioned in this feedback.
        
Feedback: {state["messages"][-1].content}

List 2-3 development opportunities with evidence."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"growth_analysis": response.content}
    
    def analyze_actions(state: ParallelState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        prompt = f"""Extract ACTIONABLE ITEMS from this feedback.
        
Feedback: {state["messages"][-1].content}

List specific actions the person should take."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"action_analysis": response.content}
    
    def synthesize_sections(state: ParallelState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""Synthesize this feedback analysis into coaching guidance.

STRENGTHS:
{state['strengths_analysis']}

GROWTH AREAS:
{state['growth_analysis']}

ACTIONS:
{state['action_analysis']}

Create a balanced, encouraging summary."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    
    graph = StateGraph(ParallelState)
    
    # Fan-out to parallel analysis
    graph.add_node("strengths", analyze_strengths)
    graph.add_node("growth", analyze_growth)
    graph.add_node("actions", analyze_actions)
    graph.add_node("synthesize", synthesize_sections)
    
    # Parallel execution (all three run from START)
    graph.add_edge(START, "strengths")
    graph.add_edge(START, "growth")
    graph.add_edge(START, "actions")
    
    # Fan-in to synthesis
    graph.add_edge("strengths", "synthesize")
    graph.add_edge("growth", "synthesize")
    graph.add_edge("actions", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def pattern4_voting():
    """Demonstrate voting variant - multiple perspectives on same question"""
    
    class VotingState(TypedDict):
        messages: Annotated[list, add_messages]
        vote_1: dict
        vote_2: dict
        vote_3: dict
        final_assessment: str
    
    def vote_perspective_1(state: VotingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        prompt = f"""As an OPTIMISTIC coach, assess: Is this person ready for promotion?

Info: {state["messages"][-1].content}

Return JSON: {{"ready": true/false, "confidence": 0-1, "reasoning": "brief"}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            return {"vote_1": json.loads(response.content)}
        except:
            return {"vote_1": {"ready": True, "confidence": 0.5, "reasoning": "uncertain"}}
    
    def vote_perspective_2(state: VotingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        prompt = f"""As a CRITICAL coach, assess: Is this person ready for promotion?

Info: {state["messages"][-1].content}

Return JSON: {{"ready": true/false, "confidence": 0-1, "reasoning": "brief"}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            return {"vote_2": json.loads(response.content)}
        except:
            return {"vote_2": {"ready": False, "confidence": 0.5, "reasoning": "uncertain"}}
    
    def vote_perspective_3(state: VotingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        prompt = f"""As a BALANCED coach, assess: Is this person ready for promotion?

Info: {state["messages"][-1].content}

Return JSON: {{"ready": true/false, "confidence": 0-1, "reasoning": "brief"}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            return {"vote_3": json.loads(response.content)}
        except:
            return {"vote_3": {"ready": True, "confidence": 0.6, "reasoning": "balanced view"}}
    
    def aggregate_votes(state: VotingState) -> dict:
        votes = [state["vote_1"], state["vote_2"], state["vote_3"]]
        ready_votes = sum(1 for v in votes if v.get("ready"))
        avg_confidence = sum(v.get("confidence", 0.5) for v in votes) / 3
        
        consensus = f"""VOTING RESULTS:
- Ready votes: {ready_votes}/3
- Average confidence: {avg_confidence:.0%}
- Perspectives:
  1. Optimistic: {votes[0]}
  2. Critical: {votes[1]}
  3. Balanced: {votes[2]}

CONSENSUS: {"Ready with development areas" if ready_votes >= 2 else "Not yet ready - needs more development"}"""
        
        return {"final_assessment": consensus, "messages": [AIMessage(content=consensus)]}
    
    graph = StateGraph(VotingState)
    graph.add_node("vote1", vote_perspective_1)
    graph.add_node("vote2", vote_perspective_2)
    graph.add_node("vote3", vote_perspective_3)
    graph.add_node("aggregate", aggregate_votes)
    
    graph.add_edge(START, "vote1")
    graph.add_edge(START, "vote2")
    graph.add_edge(START, "vote3")
    graph.add_edge("vote1", "aggregate")
    graph.add_edge("vote2", "aggregate")
    graph.add_edge("vote3", "aggregate")
    graph.add_edge("aggregate", END)
    
    return graph.compile()


def demo_pattern4():
    # Sectioning demo
    coach = pattern4_parallelization()
    feedback = "Sarah delivers high-quality work but struggles with delegation. She's technically strong but needs to develop leadership presence. Consider presenting at team meetings more."
    result = coach.invoke({
        "messages": [HumanMessage(content=feedback)],
        "strengths_analysis": "", "growth_analysis": "", "action_analysis": "",
        "votes": [], "consensus": ""
    })
    print(f"[SECTIONING] Feedback: {feedback[:50]}...")
    print(f"Strengths: {result['strengths_analysis'][:100]}...")
    print(f"Growth: {result['growth_analysis'][:100]}...")
    print(f"Final: {result['messages'][-1].content[:200]}...")
    
    # Voting demo
    voting = pattern4_voting()
    result = voting.invoke({
        "messages": [HumanMessage(content="3 years in role, good reviews, led 2 projects, mentor to 1 junior")],
        "vote_1": {}, "vote_2": {}, "vote_3": {}, "final_assessment": ""
    })
    print(f"\n[VOTING] Promotion readiness assessment:")
    print(result["final_assessment"])


# ============================================================
# PATTERN 5: ORCHESTRATOR-WORKERS
# ============================================================
# Central LLM dynamically breaks down tasks and delegates to
# worker LLMs, then synthesizes results.
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 5: ORCHESTRATOR-WORKERS")
print("=" * 70)
print("""
Central orchestrator dynamically delegates to specialized workers.
Unlike parallelization, the subtasks are determined at runtime.
""")


class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    task_breakdown: list[dict]
    worker_results: list[str]
    final_synthesis: str


def pattern5_orchestrator_workers():
    """Demonstrate Pattern 5: Orchestrator-Workers"""
    
    def orchestrator_plan(state: OrchestratorState) -> dict:
        """Orchestrator analyzes task and creates worker assignments"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        user_msg = state["messages"][-1].content
        
        prompt = f"""You are a coaching orchestrator. Break down this request into subtasks.

Request: {user_msg}

Available workers:
- research_worker: Gathers relevant information and context
- analysis_worker: Analyzes situations and identifies patterns
- strategy_worker: Develops actionable strategies
- communication_worker: Crafts messages and talking points

Return JSON:
{{
    "subtasks": [
        {{"worker": "worker_name", "task": "specific task", "priority": 1-3}}
    ]
}}

Create 2-4 subtasks. Be specific about what each worker should do."""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            parsed = json.loads(response.content)
            return {"task_breakdown": parsed.get("subtasks", [])}
        except:
            return {"task_breakdown": [{"worker": "analysis_worker", "task": user_msg, "priority": 1}]}
    
    def execute_workers(state: OrchestratorState) -> dict:
        """Execute worker tasks based on orchestrator's plan"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
        
        worker_prompts = {
            "research_worker": "You are a research specialist. Gather relevant information and context for:",
            "analysis_worker": "You are an analysis specialist. Analyze this situation and identify key patterns:",
            "strategy_worker": "You are a strategy specialist. Develop actionable strategies for:",
            "communication_worker": "You are a communication specialist. Craft effective messages for:"
        }
        
        results = []
        for subtask in state["task_breakdown"]:
            worker = subtask.get("worker", "analysis_worker")
            task = subtask.get("task", "")
            
            prompt = f"""{worker_prompts.get(worker, 'Help with:')}

Task: {task}
Context: {state["messages"][-1].content}

Provide a focused, helpful response (2-3 paragraphs max)."""

            response = llm.invoke([HumanMessage(content=prompt)])
            results.append(f"[{worker}]: {response.content}")
        
        return {"worker_results": results}
    
    def synthesize_results(state: OrchestratorState) -> dict:
        """Orchestrator synthesizes worker outputs"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        worker_outputs = "\n\n".join(state["worker_results"])
        
        prompt = f"""Synthesize these worker outputs into a cohesive coaching response.

WORKER OUTPUTS:
{worker_outputs}

Create a unified response that:
1. Addresses the original request
2. Incorporates insights from each worker
3. Provides clear next steps
4. Is warm and supportive in tone"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)], "final_synthesis": response.content}
    
    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", orchestrator_plan)
    graph.add_node("execute", execute_workers)
    graph.add_node("synthesize", synthesize_results)
    
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def demo_pattern5():
    orchestrator = pattern5_orchestrator_workers()
    result = orchestrator.invoke({
        "messages": [HumanMessage(content="I need to have a difficult conversation with my manager about workload and ask for a raise")],
        "task_breakdown": [],
        "worker_results": [],
        "final_synthesis": ""
    })
    print(f"User: Difficult conversation about workload + raise")
    print(f"\nTask breakdown: {json.dumps(result['task_breakdown'], indent=2)}")
    print(f"\nWorker results: {len(result['worker_results'])} outputs")
    print(f"\nFinal synthesis: {result['final_synthesis'][:400]}...")


# ============================================================
# PATTERN 6: EVALUATOR-OPTIMIZER
# ============================================================
# One LLM generates output, another evaluates it.
# Loop until quality threshold is met.
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 6: EVALUATOR-OPTIMIZER")
print("=" * 70)
print("""
Generator creates output, Evaluator scores it.
Loop continues until quality threshold is met.
""")


class EvaluatorState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    evaluation: dict
    iteration: int
    max_iterations: int


def pattern6_evaluator_optimizer():
    """Demonstrate Pattern 6: Evaluator-Optimizer"""
    
    def generate(state: EvaluatorState) -> dict:
        """Generate coaching response"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        user_msg = state["messages"][-1].content
        prev_feedback = state.get("evaluation", {}).get("feedback", "")
        
        prompt = f"""You are an HR coach. Respond to this query.

Query: {user_msg}

{"Previous feedback to incorporate: " + prev_feedback if prev_feedback else ""}

Provide a helpful, empathetic coaching response."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"draft": response.content, "iteration": state.get("iteration", 0) + 1}
    
    def evaluate(state: EvaluatorState) -> dict:
        """Evaluate the coaching response quality"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        prompt = f"""Evaluate this coaching response on a scale of 0-10.

QUERY: {state["messages"][-1].content}
RESPONSE: {state["draft"]}

Score these dimensions:
- empathy: Does it acknowledge feelings?
- actionability: Does it provide clear next steps?
- accuracy: Is the advice sound?
- tone: Is it appropriately warm and professional?

Return JSON:
{{
    "scores": {{"empathy": 0-10, "actionability": 0-10, "accuracy": 0-10, "tone": 0-10}},
    "overall": 0-10,
    "feedback": "specific improvements needed",
    "pass": true if overall >= 7
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            return {"evaluation": json.loads(response.content)}
        except:
            return {"evaluation": {"overall": 8, "pass": True, "feedback": ""}}
    
    def should_continue(state: EvaluatorState) -> Literal["generate", "finalize"]:
        """Decide if more iterations needed"""
        evaluation = state.get("evaluation", {})
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)
        
        if evaluation.get("pass", False) or iteration >= max_iter:
            return "finalize"
        return "generate"
    
    def finalize(state: EvaluatorState) -> dict:
        """Return final response"""
        return {"messages": [AIMessage(content=state["draft"])]}
    
    graph = StateGraph(EvaluatorState)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("finalize", finalize)
    
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {
        "generate": "generate",
        "finalize": "finalize"
    })
    graph.add_edge("finalize", END)
    
    return graph.compile()


def demo_pattern6():
    coach = pattern6_evaluator_optimizer()
    result = coach.invoke({
        "messages": [HumanMessage(content="I'm feeling overwhelmed and thinking about quitting")],
        "draft": "",
        "evaluation": {},
        "iteration": 0,
        "max_iterations": 3
    })
    print(f"User: I'm feeling overwhelmed and thinking about quitting")
    print(f"\nIterations: {result['iteration']}")
    print(f"Evaluation: {result['evaluation']}")
    print(f"\nFinal response: {result['messages'][-1].content[:300]}...")


# ============================================================
# PATTERN 7: AUTONOMOUS AGENT
# ============================================================
# Full agentic loop where LLM dynamically directs its own
# process and tool usage based on environment feedback.
# ============================================================

print("\n" + "=" * 70)
print("PATTERN 7: AUTONOMOUS AGENT")
print("=" * 70)
print("""
The agent loop: Think → Act → Observe → Repeat
Agent autonomously decides actions based on environment feedback.
""")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_goal: str
    observations: list[str]
    actions_taken: list[str]
    plan: list[str]
    step: int
    complete: bool


def pattern7_autonomous_agent():
    """Demonstrate Pattern 7: Autonomous Agent"""
    
    # Simulated environment for the agent
    class Environment:
        def __init__(self):
            self.knowledge = HR_KNOWLEDGE.copy()
            self.calendar = {"next_week": ["Team meeting Monday", "1:1 Wednesday"]}
            self.todos = []
        
        def search(self, query: str) -> str:
            for k, v in self.knowledge.items():
                if k in query.lower():
                    return f"Found: {v}"
            return "No results found"
        
        def check_calendar(self, period: str) -> str:
            return f"Calendar for {period}: {self.calendar.get(period, 'No events')}"
        
        def add_todo(self, item: str) -> str:
            self.todos.append(item)
            return f"Added to-do: {item}"
        
        def get_todos(self) -> str:
            return f"Current to-dos: {self.todos}" if self.todos else "No to-dos"
    
    ENV = Environment()
    
    def think(state: AgentState) -> dict:
        """Agent thinks about what to do next"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        
        prompt = f"""You are an autonomous HR coaching agent.

GOAL: {state['current_goal']}
OBSERVATIONS SO FAR: {state['observations']}
ACTIONS TAKEN: {state['actions_taken']}
CURRENT STEP: {state['step']}

Available actions:
- search(query): Search HR knowledge base
- check_calendar(period): Check calendar availability
- add_todo(item): Add an action item
- respond(message): Send final response to user

Decide your next action. Return JSON:
{{
    "thinking": "your reasoning",
    "action": "action_name",
    "params": {{"param": "value"}},
    "is_final": true/false
}}

If you have enough information, use respond() with is_final=true."""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            return {"plan": [response.content]}
        except:
            return {"plan": ["respond"]}
    
    def act(state: AgentState) -> dict:
        """Agent takes action and observes result"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        try:
            decision = json.loads(state["plan"][-1])
        except:
            decision = {"action": "respond", "params": {"message": "Let me help you with that."}, "is_final": True}
        
        action = decision.get("action", "respond")
        params = decision.get("params", {})
        
        # Execute action
        if action == "search":
            result = ENV.search(params.get("query", ""))
        elif action == "check_calendar":
            result = ENV.check_calendar(params.get("period", "next_week"))
        elif action == "add_todo":
            result = ENV.add_todo(params.get("item", "Follow up"))
        elif action == "respond":
            # Generate final response
            response = llm.invoke([HumanMessage(content=f"""
Generate a helpful coaching response.

Goal: {state['current_goal']}
Information gathered: {state['observations']}

Provide a warm, actionable response.""")])
            return {
                "messages": [AIMessage(content=response.content)],
                "complete": True,
                "step": state["step"] + 1
            }
        else:
            result = "Unknown action"
        
        return {
            "observations": state["observations"] + [f"{action}: {result}"],
            "actions_taken": state["actions_taken"] + [action],
            "step": state["step"] + 1,
            "complete": decision.get("is_final", False)
        }
    
    def should_continue(state: AgentState) -> Literal["think", "end"]:
        """Check if agent should continue or stop"""
        if state.get("complete", False) or state.get("step", 0) >= 5:
            return "end"
        return "think"
    
    def end_agent(state: AgentState) -> dict:
        """Ensure we have a final message"""
        if not any(isinstance(m, AIMessage) for m in state.get("messages", [])):
            return {"messages": [AIMessage(content="I've gathered information to help you. Let me know if you have questions.")]}
        return {}
    
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("end", end_agent)
    
    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_conditional_edges("act", should_continue, {
        "think": "think",
        "end": "end"
    })
    graph.add_edge("end", END)
    
    return graph.compile()


def demo_pattern7():
    agent = pattern7_autonomous_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content="Help me prepare for asking my manager for a promotion")],
        "current_goal": "Help user prepare for promotion conversation with manager",
        "observations": [],
        "actions_taken": [],
        "plan": [],
        "step": 0,
        "complete": False
    })
    print(f"User: Help me prepare for asking my manager for a promotion")
    print(f"\nActions taken: {result['actions_taken']}")
    print(f"Observations: {result['observations']}")
    print(f"Steps: {result['step']}")
    print(f"\nFinal response: {result['messages'][-1].content[:300]}...")


# ============================================================
# MAIN DEMO
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("RUNNING ALL PATTERN DEMOS")
    print("=" * 70)
    
    demos = [
        ("Pattern 1: Augmented LLM", demo_pattern1),
        ("Pattern 2: Prompt Chaining", demo_pattern2),
        ("Pattern 3: Routing", demo_pattern3),
        ("Pattern 4: Parallelization", demo_pattern4),
        ("Pattern 5: Orchestrator-Workers", demo_pattern5),
        ("Pattern 6: Evaluator-Optimizer", demo_pattern6),
        ("Pattern 7: Autonomous Agent", demo_pattern7),
    ]
    
    for name, demo_fn in demos:
        print(f"\n{'='*70}")
        print(f"DEMO: {name}")
        print("=" * 70)
        try:
            demo_fn()
        except Exception as e:
            print(f"Error in {name}: {e}")
        print()
    
    print("\n" + "=" * 70)
    print("ALL PATTERNS DEMONSTRATED!")
    print("=" * 70)
    print("""
Summary of Anthropic's Agentic Patterns:

1. AUGMENTED LLM - Foundation with retrieval, tools, memory
2. PROMPT CHAINING - Sequential steps, each builds on previous
3. ROUTING - Classify and direct to specialized handlers
4. PARALLELIZATION - Concurrent execution (sectioning/voting)
5. ORCHESTRATOR-WORKERS - Dynamic delegation and synthesis
6. EVALUATOR-OPTIMIZER - Generate-evaluate-refine loop
7. AUTONOMOUS AGENT - Think-act-observe loop

Key principles:
- Start simple, add complexity only when needed
- Each pattern has specific use cases
- Combine patterns for sophisticated systems
""")


if __name__ == "__main__":
    main()
