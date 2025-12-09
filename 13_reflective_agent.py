"""
Script 13: Self-Reflective Agent with Evaluation and Iterative Refinement
Demonstrates: Reflection patterns, critique agents, self-evaluation, 
              iterative improvement, confidence calibration, learning from feedback
"""

from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# EVALUATION FRAMEWORK
# ============================================================

class QualityDimension(str, Enum):
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    ACTIONABILITY = "actionability"
    EMPATHY = "empathy"
    COMPLETENESS = "completeness"
    SAFETY = "safety"


@dataclass
class CritiqueResult:
    """Result from a critique evaluation."""
    dimension: QualityDimension
    score: float  # 0-1
    issues: list[str]
    suggestions: list[str]
    confidence: float


@dataclass
class ReflectionResult:
    """Result from self-reflection."""
    original_reasoning: str
    identified_gaps: list[str]
    alternative_perspectives: list[str]
    revised_approach: str
    confidence_adjustment: float


@dataclass 
class RefinementIteration:
    """Tracks a single refinement iteration."""
    iteration: int
    original_response: str
    critiques: list[CritiqueResult]
    reflection: ReflectionResult
    refined_response: str
    improvement_score: float


class ReflectiveState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    
    # Generation
    current_response: str
    response_reasoning: str
    confidence: float
    
    # Evaluation
    critiques: list[dict]
    aggregate_quality_score: float
    quality_threshold: float
    
    # Reflection
    reflection: dict
    identified_improvements: list[str]
    
    # Iteration
    iteration_count: int
    max_iterations: int
    iteration_history: list[dict]
    
    # Final
    final_response: str
    improvement_summary: str


# ============================================================
# CRITIC AGENTS
# ============================================================

CRITIC_CONFIGS = {
    QualityDimension.ACCURACY: {
        "prompt": """You are an Accuracy Critic evaluating HR coaching responses.

Evaluate for:
- Factual correctness of HR information
- Alignment with HR best practices
- Appropriate caveats for uncertain information
- No hallucinated policies or procedures

Score 0-1 where 1 is perfectly accurate.""",
        "weight": 0.25
    },
    
    QualityDimension.RELEVANCE: {
        "prompt": """You are a Relevance Critic evaluating HR coaching responses.

Evaluate for:
- Direct addressing of user's question/concern
- Appropriate level of detail for the context
- Focus on what matters most to the user
- No tangential or unnecessary information

Score 0-1 where 1 is perfectly relevant.""",
        "weight": 0.20
    },
    
    QualityDimension.ACTIONABILITY: {
        "prompt": """You are an Actionability Critic evaluating HR coaching responses.

Evaluate for:
- Clear, concrete next steps
- Realistic and achievable recommendations
- Specific timelines and milestones where appropriate
- Resources or tools mentioned when relevant

Score 0-1 where 1 is highly actionable.""",
        "weight": 0.20
    },
    
    QualityDimension.EMPATHY: {
        "prompt": """You are an Empathy Critic evaluating HR coaching responses.

Evaluate for:
- Acknowledgment of emotions and challenges
- Supportive and encouraging tone
- Non-judgmental language
- Appropriate validation without enabling

Score 0-1 where 1 is highly empathetic.""",
        "weight": 0.15
    },
    
    QualityDimension.COMPLETENESS: {
        "prompt": """You are a Completeness Critic evaluating HR coaching responses.

Evaluate for:
- All aspects of the question addressed
- Important considerations not overlooked
- Follow-up questions anticipated
- Appropriate scope (not too brief or too verbose)

Score 0-1 where 1 is appropriately complete.""",
        "weight": 0.10
    },
    
    QualityDimension.SAFETY: {
        "prompt": """You are a Safety Critic evaluating HR coaching responses.

Evaluate for:
- No harmful advice given
- Appropriate disclaimers for legal/medical/clinical topics
- Escalation paths mentioned when needed
- No potential for misuse or misinterpretation

Score 0-1 where 1 is completely safe.""",
        "weight": 0.10
    }
}


def create_critic_evaluator(dimension: QualityDimension):
    """Create a critic evaluator for a specific dimension."""
    
    config = CRITIC_CONFIGS[dimension]
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    def evaluate(user_query: str, response: str) -> CritiqueResult:
        prompt = f"""{config['prompt']}

User Query:
"{user_query}"

Response to Evaluate:
"{response}"

Provide your evaluation as JSON:
{{
    "score": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "confidence": 0.0-1.0
}}"""
        
        result = llm.invoke([HumanMessage(content=prompt)])
        
        try:
            parsed = json.loads(result.content)
            return CritiqueResult(
                dimension=dimension,
                score=parsed.get("score", 0.5),
                issues=parsed.get("issues", []),
                suggestions=parsed.get("suggestions", []),
                confidence=parsed.get("confidence", 0.7)
            )
        except json.JSONDecodeError:
            # Parse manually if JSON fails
            score_match = re.search(r'"score":\s*([\d.]+)', result.content)
            return CritiqueResult(
                dimension=dimension,
                score=float(score_match.group(1)) if score_match else 0.5,
                issues=[result.content],
                suggestions=[],
                confidence=0.5
            )
    
    return evaluate


# ============================================================
# REFLECTION ENGINE
# ============================================================

class ReflectionEngine:
    """Engine for agent self-reflection."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    def reflect(self, 
                original_query: str,
                original_response: str,
                original_reasoning: str,
                critiques: list[CritiqueResult]) -> ReflectionResult:
        """Perform deep reflection on response and critiques."""
        
        critique_summary = "\n".join([
            f"- {c.dimension.value}: {c.score:.2f} - Issues: {c.issues}"
            for c in critiques
        ])
        
        reflection_prompt = f"""Reflect deeply on this coaching interaction.

Original Query: "{original_query}"

Your Response: "{original_response}"

Your Reasoning: "{original_reasoning}"

Critiques Received:
{critique_summary}

Perform thorough self-reflection:

1. GAPS: What did I miss or get wrong? What assumptions did I make?

2. PERSPECTIVES: What alternative viewpoints should I consider?
   - From the employee's perspective
   - From a manager's perspective
   - From HR's perspective
   - From an organizational perspective

3. REVISED APPROACH: How should I approach this differently?

4. CONFIDENCE: How should I adjust my confidence? (increase/decrease/maintain and by how much)

Respond as JSON:
{{
    "identified_gaps": ["gap1", "gap2"],
    "alternative_perspectives": ["perspective1", "perspective2"],
    "revised_approach": "description of how to improve",
    "confidence_adjustment": -0.1 to +0.1
}}"""
        
        result = self.llm.invoke([HumanMessage(content=reflection_prompt)])
        
        try:
            parsed = json.loads(result.content)
            return ReflectionResult(
                original_reasoning=original_reasoning,
                identified_gaps=parsed.get("identified_gaps", []),
                alternative_perspectives=parsed.get("alternative_perspectives", []),
                revised_approach=parsed.get("revised_approach", ""),
                confidence_adjustment=parsed.get("confidence_adjustment", 0.0)
            )
        except json.JSONDecodeError:
            return ReflectionResult(
                original_reasoning=original_reasoning,
                identified_gaps=["Failed to parse reflection"],
                alternative_perspectives=[],
                revised_approach=result.content,
                confidence_adjustment=0.0
            )


# ============================================================
# GRAPH NODES
# ============================================================

def generate_initial_response(state: ReflectiveState) -> dict:
    """Generate initial coaching response with reasoning."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""You are an HR coaching assistant. Respond to this query:

"{user_query}"

Think through your response carefully:
1. What is the user really asking/needing?
2. What relevant HR knowledge applies?
3. What are the key considerations?
4. What actionable guidance can I provide?

Respond as JSON:
{{
    "reasoning": "Your step-by-step thinking",
    "response": "Your actual coaching response",
    "confidence": 0.0-1.0
}}"""
    
    result = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(result.content)
        return {
            "current_response": parsed.get("response", result.content),
            "response_reasoning": parsed.get("reasoning", ""),
            "confidence": parsed.get("confidence", 0.7),
            "iteration_count": 1
        }
    except json.JSONDecodeError:
        return {
            "current_response": result.content,
            "response_reasoning": "Direct response without explicit reasoning",
            "confidence": 0.6,
            "iteration_count": 1
        }


def run_critics(state: ReflectiveState) -> dict:
    """Run all critic evaluations in parallel."""
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    response = state.get("current_response", "")
    
    critiques = []
    weighted_score = 0.0
    total_weight = 0.0
    
    for dimension, config in CRITIC_CONFIGS.items():
        evaluator = create_critic_evaluator(dimension)
        critique = evaluator(user_query, response)
        
        critiques.append({
            "dimension": critique.dimension.value,
            "score": critique.score,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
            "confidence": critique.confidence,
            "weight": config["weight"]
        })
        
        weighted_score += critique.score * config["weight"]
        total_weight += config["weight"]
    
    aggregate_score = weighted_score / total_weight if total_weight > 0 else 0.5
    
    return {
        "critiques": critiques,
        "aggregate_quality_score": aggregate_score
    }


def perform_reflection(state: ReflectiveState) -> dict:
    """Perform self-reflection based on critiques."""
    
    engine = ReflectionEngine()
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Convert critique dicts back to CritiqueResult objects
    critique_objects = [
        CritiqueResult(
            dimension=QualityDimension(c["dimension"]),
            score=c["score"],
            issues=c["issues"],
            suggestions=c["suggestions"],
            confidence=c["confidence"]
        )
        for c in state.get("critiques", [])
    ]
    
    reflection = engine.reflect(
        original_query=user_query,
        original_response=state.get("current_response", ""),
        original_reasoning=state.get("response_reasoning", ""),
        critiques=critique_objects
    )
    
    # Compile improvement suggestions
    all_suggestions = []
    for c in state.get("critiques", []):
        all_suggestions.extend(c.get("suggestions", []))
    
    return {
        "reflection": {
            "gaps": reflection.identified_gaps,
            "perspectives": reflection.alternative_perspectives,
            "revised_approach": reflection.revised_approach,
            "confidence_adjustment": reflection.confidence_adjustment
        },
        "identified_improvements": all_suggestions,
        "confidence": min(1.0, max(0.0, state.get("confidence", 0.7) + reflection.confidence_adjustment))
    }


def refine_response(state: ReflectiveState) -> dict:
    """Refine response based on reflection and critiques."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.6)
    
    reflection = state.get("reflection", {})
    critiques = state.get("critiques", [])
    
    # Build refinement prompt
    critique_issues = []
    for c in critiques:
        if c["score"] < 0.8:
            critique_issues.append(f"- {c['dimension']}: {', '.join(c['issues'])}")
    
    prompt = f"""Refine your coaching response based on self-reflection and critique.

ORIGINAL RESPONSE:
{state.get('current_response', '')}

IDENTIFIED GAPS:
{reflection.get('gaps', [])}

ALTERNATIVE PERSPECTIVES TO CONSIDER:
{reflection.get('perspectives', [])}

CRITIQUE ISSUES TO ADDRESS:
{chr(10).join(critique_issues)}

SUGGESTED IMPROVEMENTS:
{state.get('identified_improvements', [])}

REVISED APPROACH:
{reflection.get('revised_approach', '')}

Now generate an IMPROVED response that:
1. Addresses all identified gaps
2. Incorporates multiple perspectives
3. Fixes the critique issues
4. Implements the suggested improvements

Provide ONLY the improved response, no meta-commentary."""
    
    result = llm.invoke([HumanMessage(content=prompt)])
    
    # Record iteration history
    history = state.get("iteration_history", [])
    history.append({
        "iteration": state.get("iteration_count", 1),
        "response": state.get("current_response", ""),
        "score": state.get("aggregate_quality_score", 0),
        "gaps_addressed": reflection.get("gaps", [])
    })
    
    return {
        "current_response": result.content,
        "iteration_count": state.get("iteration_count", 1) + 1,
        "iteration_history": history
    }


def check_quality(state: ReflectiveState) -> Literal["refine", "finalize"]:
    """Decide whether to refine further or finalize."""
    
    score = state.get("aggregate_quality_score", 0)
    threshold = state.get("quality_threshold", 0.8)
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)
    
    # Finalize if quality is good enough or max iterations reached
    if score >= threshold:
        return "finalize"
    
    if iteration >= max_iterations:
        return "finalize"
    
    return "refine"


def finalize_response(state: ReflectiveState) -> dict:
    """Finalize the response and generate improvement summary."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    history = state.get("iteration_history", [])
    
    # Generate improvement summary
    if len(history) > 0:
        initial_score = history[0].get("score", 0) if history else 0
        final_score = state.get("aggregate_quality_score", 0)
        
        summary_prompt = f"""Summarize the improvement process:

Initial Quality Score: {initial_score:.2f}
Final Quality Score: {final_score:.2f}
Iterations: {len(history)}

Gaps Addressed Across Iterations:
{json.dumps([h.get('gaps_addressed', []) for h in history], indent=2)}

Create a brief (2-3 sentence) summary of how the response was improved."""
        
        summary_result = llm.invoke([HumanMessage(content=summary_prompt)])
        improvement_summary = summary_result.content
    else:
        improvement_summary = "Response generated without refinement iterations."
    
    return {
        "final_response": state.get("current_response", ""),
        "improvement_summary": improvement_summary,
        "messages": [AIMessage(content=state.get("current_response", ""))]
    }


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_reflective_graph():
    """Build the self-reflective coaching graph."""
    
    graph = StateGraph(ReflectiveState)
    
    # Add nodes
    graph.add_node("generate", generate_initial_response)
    graph.add_node("critique", run_critics)
    graph.add_node("reflect", perform_reflection)
    graph.add_node("refine", refine_response)
    graph.add_node("finalize", finalize_response)
    
    # Define flow
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "reflect")
    
    # Conditional: refine or finalize
    graph.add_conditional_edges(
        "reflect",
        check_quality,
        {
            "refine": "refine",
            "finalize": "finalize"
        }
    )
    
    # After refinement, critique again
    graph.add_edge("refine", "critique")
    
    graph.add_edge("finalize", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# MAIN
# ============================================================

def main():
    """Demo the self-reflective coaching agent."""
    
    print("=" * 70)
    print("SELF-REFLECTIVE HR COACHING AGENT")
    print("=" * 70)
    print("\nThis agent:")
    print("  1. Generates initial response with explicit reasoning")
    print("  2. Runs 6 critic evaluations (accuracy, relevance, etc.)")
    print("  3. Performs deep self-reflection")
    print("  4. Iteratively refines until quality threshold met")
    print("\nQuality Dimensions:")
    for dim, config in CRITIC_CONFIGS.items():
        print(f"  • {dim.value} (weight: {config['weight']:.0%})")
    
    print("\n" + "=" * 70)
    print("Type 'quit' to exit")
    print("Type 'verbose' before query for detailed iteration tracking")
    print("=" * 70 + "\n")
    
    graph = build_reflective_graph()
    
    session_id = f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    verbose = False
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"[Verbose mode: {'ON' if verbose else 'OFF'}]\n")
            continue
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": "reflective_user",
            "current_response": "",
            "response_reasoning": "",
            "confidence": 0.0,
            "critiques": [],
            "aggregate_quality_score": 0.0,
            "quality_threshold": 0.8,
            "reflection": {},
            "identified_improvements": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "iteration_history": [],
            "final_response": "",
            "improvement_summary": ""
        }
        
        print("\n[Generating and evaluating...]")
        
        # Stream through nodes for visibility
        for event in graph.stream(state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if verbose:
                    if node_name == "generate":
                        print(f"\n[Initial Response Generated]")
                        print(f"  Confidence: {node_output.get('confidence', 0):.2f}")
                    
                    elif node_name == "critique":
                        print(f"\n[Critique Results]")
                        print(f"  Aggregate Score: {node_output.get('aggregate_quality_score', 0):.2f}")
                        for c in node_output.get("critiques", []):
                            issues = ", ".join(c.get("issues", [])[:2]) or "None"
                            print(f"  • {c['dimension']}: {c['score']:.2f} - {issues}")
                    
                    elif node_name == "reflect":
                        print(f"\n[Self-Reflection]")
                        reflection = node_output.get("reflection", {})
                        if reflection.get("gaps"):
                            print(f"  Gaps: {reflection['gaps'][:2]}")
                        print(f"  Confidence Adjustment: {reflection.get('confidence_adjustment', 0):+.2f}")
                    
                    elif node_name == "refine":
                        print(f"\n[Refined - Iteration {node_output.get('iteration_count', '?')}]")
                    
                    elif node_name == "finalize":
                        print(f"\n[Finalized]")
        
        # Get final state
        final_state = graph.get_state(config)
        state = dict(final_state.values)
        
        # Display results
        print("\n" + "-" * 50)
        print(f"QUALITY: {state.get('aggregate_quality_score', 0):.2f} | " +
              f"ITERATIONS: {len(state.get('iteration_history', []))} | " +
              f"CONFIDENCE: {state.get('confidence', 0):.2f}")
        print("-" * 50)
        
        print(f"\nCoach: {state.get('final_response', 'No response')}")
        
        if state.get("improvement_summary"):
            print(f"\n[Improvement: {state['improvement_summary']}]")
        
        print()


if __name__ == "__main__":
    main()
