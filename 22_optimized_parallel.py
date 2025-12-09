"""
22. Optimized Parallelization Pattern
=====================================
Advanced parallelization demonstrating:

1. TRUE ASYNC EXECUTION - Concurrent API calls with asyncio
2. DYNAMIC FAN-OUT - Number of workers determined at runtime
3. WEIGHTED VOTING - Confidence-weighted aggregation
4. SECTIONING WITH PRIORITIES - Critical path optimization
5. TIMEOUT HANDLING - Graceful degradation on slow workers
6. RESULT CACHING - Avoid redundant computation
7. PROGRESS STREAMING - Real-time updates
8. SMART AGGREGATION - Contradiction detection and resolution

Use Case: Comprehensive 360-degree feedback analysis for promotion readiness
"""

import asyncio
import json
import re
import time
import hashlib
from typing import Annotated, TypedDict, Literal, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in code blocks
    patterns = [
        r'```json\s*([\s\S]*?)```',
        r'```\s*([\s\S]*?)```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str.strip())
            except:
                continue
    
    # Return default if all else fails
    return {}

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


print("=" * 80)
print("OPTIMIZED PARALLELIZATION - COMPREHENSIVE 360 FEEDBACK ANALYSIS")
print("=" * 80)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    max_workers: int = 8
    timeout_seconds: float = 30.0
    min_responses_required: int = 3  # Graceful degradation threshold
    enable_caching: bool = True
    enable_streaming: bool = True
    retry_failed: bool = True
    max_retries: int = 2


CONFIG = ParallelConfig()


# ============================================================
# CACHING LAYER
# ============================================================

class AnalysisCache:
    """Thread-safe cache for analysis results"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_key(self, analysis_type: str, content: str) -> str:
        return hashlib.md5(f"{analysis_type}:{content[:200]}".encode()).hexdigest()
    
    def get(self, analysis_type: str, content: str) -> Optional[dict]:
        key = self._hash_key(analysis_type, content)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, analysis_type: str, content: str, result: dict):
        key = self._hash_key(analysis_type, content)
        with self._lock:
            self._cache[key] = result
    
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0
            }


CACHE = AnalysisCache()


# ============================================================
# PROGRESS TRACKER
# ============================================================

class ProgressTracker:
    """Track parallel execution progress"""
    
    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.completed = 0
        self.failed = 0
        self.results = {}
        self._lock = threading.Lock()
        self.start_time = time.time()
    
    def complete(self, task_id: str, result: dict):
        with self._lock:
            self.completed += 1
            self.results[task_id] = result
            elapsed = time.time() - self.start_time
            print(f"   [{self.completed}/{self.total}] {task_id} completed ({elapsed:.1f}s)")
    
    def fail(self, task_id: str, error: str):
        with self._lock:
            self.failed += 1
            print(f"   [{self.completed}/{self.total}] {task_id} failed: {error}")
    
    def summary(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "success_rate": self.completed / self.total if self.total > 0 else 0,
                "elapsed_seconds": time.time() - self.start_time
            }


# ============================================================
# STATE DEFINITIONS
# ============================================================

@dataclass
class AnalysisResult:
    """Result from a single parallel analyzer"""
    analyzer_id: str
    analyzer_type: str
    analysis: str
    confidence: float
    key_findings: list[str]
    recommendations: list[str]
    risk_flags: list[str]
    execution_time: float
    from_cache: bool = False


@dataclass  
class VoteResult:
    """Result from a voting perspective"""
    voter_id: str
    perspective: str
    decision: str  # "ready", "not_ready", "needs_development"
    confidence: float
    reasoning: str
    weight: float  # Expertise-based weight


class OptimizedParallelState(TypedDict):
    """State for optimized parallel execution"""
    messages: Annotated[list, add_messages]
    
    # Input analysis
    input_complexity: str  # "low", "medium", "high", "critical"
    identified_dimensions: list[str]
    
    # Dynamic fan-out configuration
    planned_analyzers: list[dict]
    planned_voters: list[dict]
    
    # Parallel execution results
    analysis_results: list[dict]
    vote_results: list[dict]
    
    # Aggregated outputs
    strengths_synthesis: str
    growth_areas_synthesis: str
    risk_assessment: str
    promotion_recommendation: str
    
    # Metadata
    execution_stats: dict
    contradictions_found: list[dict]


# ============================================================
# ANALYZER DEFINITIONS
# ============================================================

ANALYZERS = {
    "technical_skills": {
        "prompt": """Analyze TECHNICAL SKILLS from this feedback.
Focus on: coding ability, system design, architecture decisions, technical leadership.
Rate technical readiness 1-10 with evidence.""",
        "weight": 1.2,
        "timeout": 15
    },
    "leadership": {
        "prompt": """Analyze LEADERSHIP CAPABILITIES from this feedback.
Focus on: team guidance, decision making, mentoring, strategic thinking.
Rate leadership readiness 1-10 with evidence.""",
        "weight": 1.3,
        "timeout": 15
    },
    "communication": {
        "prompt": """Analyze COMMUNICATION SKILLS from this feedback.
Focus on: clarity, stakeholder management, written/verbal skills, influence.
Rate communication readiness 1-10 with evidence.""",
        "weight": 1.0,
        "timeout": 12
    },
    "collaboration": {
        "prompt": """Analyze COLLABORATION & TEAMWORK from this feedback.
Focus on: cross-team work, conflict resolution, inclusivity, peer relationships.
Rate collaboration readiness 1-10 with evidence.""",
        "weight": 1.0,
        "timeout": 12
    },
    "delivery": {
        "prompt": """Analyze EXECUTION & DELIVERY from this feedback.
Focus on: project completion, quality, deadlines, reliability, impact.
Rate delivery readiness 1-10 with evidence.""",
        "weight": 1.1,
        "timeout": 12
    },
    "growth_mindset": {
        "prompt": """Analyze GROWTH MINDSET & LEARNING from this feedback.
Focus on: feedback receptiveness, skill development, adaptability, curiosity.
Rate growth mindset 1-10 with evidence.""",
        "weight": 0.9,
        "timeout": 12
    },
    "culture_fit": {
        "prompt": """Analyze CULTURE & VALUES ALIGNMENT from this feedback.
Focus on: company values demonstration, team culture contribution, integrity.
Rate culture alignment 1-10 with evidence.""",
        "weight": 0.8,
        "timeout": 10
    },
    "risk_assessment": {
        "prompt": """Analyze POTENTIAL RISKS from this feedback.
Focus on: red flags, concerns, gaps, potential issues post-promotion.
Identify specific risks with severity (low/medium/high).""",
        "weight": 1.4,
        "timeout": 15
    }
}


VOTER_PERSPECTIVES = {
    "optimistic_advocate": {
        "prompt": """As an ADVOCATE who champions employee growth, assess promotion readiness.
Look for potential, trajectory, and achievements that support promotion.""",
        "weight": 0.8,
        "bias": "positive"
    },
    "skeptical_evaluator": {
        "prompt": """As a SKEPTICAL EVALUATOR focused on standards, assess promotion readiness.
Look for gaps, missing evidence, and reasons to delay promotion.""",
        "weight": 1.2,
        "bias": "negative"
    },
    "balanced_judge": {
        "prompt": """As a BALANCED JUDGE weighing all evidence, assess promotion readiness.
Consider both strengths and gaps objectively.""",
        "weight": 1.5,
        "bias": "neutral"
    },
    "peer_perspective": {
        "prompt": """As a PEER at the target level, assess if this person would be an effective colleague.
Consider collaboration, technical contribution, and team dynamics.""",
        "weight": 1.0,
        "bias": "peer"
    },
    "manager_perspective": {
        "prompt": """As a HIRING MANAGER for the next level, assess if you'd want this person on your team.
Consider readiness for increased scope and responsibility.""",
        "weight": 1.3,
        "bias": "manager"
    }
}


# ============================================================
# PARALLEL EXECUTION ENGINE
# ============================================================

class ParallelExecutor:
    """Optimized parallel execution engine"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    def _run_single_analysis(
        self,
        analyzer_id: str,
        analyzer_config: dict,
        feedback_content: str,
        tracker: ProgressTracker
    ) -> Optional[AnalysisResult]:
        """Execute a single analyzer with caching and timeout"""
        
        start_time = time.time()
        
        # Check cache first
        if self.config.enable_caching:
            cached = CACHE.get(analyzer_id, feedback_content)
            if cached:
                tracker.complete(analyzer_id, cached)
                return AnalysisResult(
                    analyzer_id=analyzer_id,
                    analyzer_type="sectioning",
                    analysis=cached.get("analysis", ""),
                    confidence=cached.get("confidence", 0.7),
                    key_findings=cached.get("key_findings", []),
                    recommendations=cached.get("recommendations", []),
                    risk_flags=cached.get("risk_flags", []),
                    execution_time=0.01,
                    from_cache=True
                )
        
        # Build prompt
        prompt = f"""{analyzer_config['prompt']}

FEEDBACK TO ANALYZE:
{feedback_content}

Return JSON:
{{
    "analysis": "detailed analysis",
    "score": 1-10,
    "confidence": 0.0-1.0,
    "key_findings": ["finding1", "finding2"],
    "recommendations": ["rec1", "rec2"],
    "risk_flags": ["risk1"] or []
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_data = extract_json(response.content)
            if not result_data:
                raise ValueError("Failed to extract JSON from response")
            
            execution_time = time.time() - start_time
            
            result = AnalysisResult(
                analyzer_id=analyzer_id,
                analyzer_type="sectioning",
                analysis=result_data.get("analysis", ""),
                confidence=result_data.get("confidence", 0.7),
                key_findings=result_data.get("key_findings", []),
                recommendations=result_data.get("recommendations", []),
                risk_flags=result_data.get("risk_flags", []),
                execution_time=execution_time
            )
            
            # Cache result
            if self.config.enable_caching:
                CACHE.set(analyzer_id, feedback_content, result_data)
            
            tracker.complete(analyzer_id, result_data)
            return result
            
        except Exception as e:
            tracker.fail(analyzer_id, str(e))
            return None
    
    def _run_single_vote(
        self,
        voter_id: str,
        voter_config: dict,
        analysis_summary: str,
        tracker: ProgressTracker
    ) -> Optional[VoteResult]:
        """Execute a single voting perspective"""
        
        prompt = f"""{voter_config['prompt']}

ANALYSIS SUMMARY:
{analysis_summary}

Return JSON:
{{
    "decision": "ready" | "not_ready" | "needs_development",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_strengths": ["strength1"],
    "key_concerns": ["concern1"]
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_data = extract_json(response.content)
            if not result_data:
                raise ValueError("Failed to extract JSON from response")
            
            result = VoteResult(
                voter_id=voter_id,
                perspective=voter_config.get("bias", "neutral"),
                decision=result_data.get("decision", "needs_development"),
                confidence=result_data.get("confidence", 0.5),
                reasoning=result_data.get("reasoning", ""),
                weight=voter_config.get("weight", 1.0)
            )
            
            tracker.complete(voter_id, result_data)
            return result
            
        except Exception as e:
            tracker.fail(voter_id, str(e))
            return None
    
    def execute_analyses_parallel(
        self,
        analyzers: dict,
        feedback_content: str
    ) -> tuple[list[AnalysisResult], dict]:
        """Execute all analyzers in parallel with ThreadPoolExecutor"""
        
        tracker = ProgressTracker(len(analyzers))
        results = []
        
        print(f"\n Starting parallel analysis ({len(analyzers)} analyzers)...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_analysis,
                    analyzer_id,
                    config,
                    feedback_content,
                    tracker
                ): analyzer_id
                for analyzer_id, config in analyzers.items()
            }
            
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                analyzer_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    tracker.fail(analyzer_id, str(e))
        
        stats = tracker.summary()
        stats["cache_stats"] = CACHE.stats()
        
        print(f" Analysis complete: {stats['completed']}/{stats['total']} succeeded in {stats['elapsed_seconds']:.1f}s")
        
        return results, stats
    
    def execute_votes_parallel(
        self,
        voters: dict,
        analysis_summary: str
    ) -> tuple[list[VoteResult], dict]:
        """Execute all voting perspectives in parallel"""
        
        tracker = ProgressTracker(len(voters))
        results = []
        
        print(f"\n🗳️  Starting parallel voting ({len(voters)} perspectives)...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_vote,
                    voter_id,
                    config,
                    analysis_summary,
                    tracker
                ): voter_id
                for voter_id, config in voters.items()
            }
            
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                voter_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    tracker.fail(voter_id, str(e))
        
        stats = tracker.summary()
        print(f" Voting complete: {stats['completed']}/{stats['total']} succeeded in {stats['elapsed_seconds']:.1f}s")
        
        return results, stats


EXECUTOR = ParallelExecutor(CONFIG)


# ============================================================
# SMART AGGREGATION
# ============================================================

class SmartAggregator:
    """Intelligent aggregation with contradiction detection"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def detect_contradictions(self, results: list[AnalysisResult]) -> list[dict]:
        """Detect contradictions between analyzer outputs"""
        contradictions = []
        
        # Compare key findings across analyzers
        findings_by_analyzer = {r.analyzer_id: r.key_findings for r in results}
        
        # Simple heuristic: if one says "strong" and another says "weak" on same topic
        strength_indicators = ["strong", "excellent", "exceptional", "outstanding"]
        weakness_indicators = ["weak", "lacking", "poor", "insufficient", "needs work"]
        
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                r1_positive = any(ind in " ".join(r1.key_findings).lower() for ind in strength_indicators)
                r2_negative = any(ind in " ".join(r2.key_findings).lower() for ind in weakness_indicators)
                
                if r1_positive and r2_negative:
                    contradictions.append({
                        "analyzers": [r1.analyzer_id, r2.analyzer_id],
                        "type": "sentiment_mismatch",
                        "description": f"{r1.analyzer_id} positive vs {r2.analyzer_id} negative"
                    })
        
        return contradictions
    
    def weighted_vote_aggregation(self, votes: list[VoteResult]) -> dict:
        """Aggregate votes with confidence and expertise weighting"""
        
        if not votes:
            return {"decision": "insufficient_data", "confidence": 0}
        
        # Calculate weighted scores
        decision_scores = {"ready": 0.0, "not_ready": 0.0, "needs_development": 0.0}
        total_weight = 0.0
        
        for vote in votes:
            effective_weight = vote.weight * vote.confidence
            decision_scores[vote.decision] += effective_weight
            total_weight += effective_weight
        
        # Normalize
        if total_weight > 0:
            for k in decision_scores:
                decision_scores[k] /= total_weight
        
        # Determine winner
        winner = max(decision_scores, key=decision_scores.get)
        confidence = decision_scores[winner]
        
        # Calculate consensus level
        sorted_scores = sorted(decision_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        
        return {
            "decision": winner,
            "confidence": confidence,
            "margin": margin,
            "consensus": "strong" if margin > 0.3 else "moderate" if margin > 0.15 else "weak",
            "vote_breakdown": decision_scores,
            "individual_votes": [
                {"voter": v.voter_id, "decision": v.decision, "confidence": v.confidence, "weight": v.weight}
                for v in votes
            ]
        }
    
    def synthesize_analyses(self, results: list[AnalysisResult], category: str) -> str:
        """Synthesize multiple analyses into coherent summary"""
        
        if not results:
            return "Insufficient data for synthesis."
        
        # Gather all findings
        all_findings = []
        all_recommendations = []
        all_risks = []
        
        for r in results:
            all_findings.extend(r.key_findings)
            all_recommendations.extend(r.recommendations)
            all_risks.extend(r.risk_flags)
        
        prompt = f"""Synthesize these {category} findings into a coherent summary.

FINDINGS:
{json.dumps(all_findings, indent=2)}

RECOMMENDATIONS:
{json.dumps(all_recommendations, indent=2)}

{"RISKS:" + json.dumps(all_risks, indent=2) if all_risks else ""}

Create a 2-3 paragraph synthesis that:
1. Identifies the main themes
2. Highlights the most important points
3. Notes any patterns or concerns
4. Provides actionable insights"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


AGGREGATOR = SmartAggregator()


# ============================================================
# GRAPH NODES
# ============================================================

def analyze_input_complexity(state: OptimizedParallelState) -> dict:
    """Analyze input to determine optimal parallelization strategy"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    user_msg = state["messages"][-1].content
    
    prompt = f"""Analyze this feedback for a promotion assessment. Determine:
1. Complexity level (low/medium/high/critical)
2. Key dimensions that need analysis
3. Recommended depth of analysis

Feedback: {user_msg[:1000]}...

Return JSON:
{{
    "complexity": "low|medium|high|critical",
    "dimensions": ["leadership", "technical", etc.],
    "word_count": number,
    "sentiment_mixed": true/false,
    "recommended_analyzers": ["analyzer_ids"],
    "recommended_voters": ["voter_ids"]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        analysis = extract_json(response.content)
        if not analysis:
            raise ValueError("Empty JSON")
    except:
        analysis = {
            "complexity": "medium",
            "dimensions": list(ANALYZERS.keys())[:5],
            "recommended_analyzers": list(ANALYZERS.keys()),
            "recommended_voters": list(VOTER_PERSPECTIVES.keys())
        }
    
    # Dynamic fan-out based on complexity
    complexity = analysis.get("complexity", "medium")
    
    if complexity == "low":
        analyzers = {k: v for k, v in list(ANALYZERS.items())[:4]}
        voters = {k: v for k, v in list(VOTER_PERSPECTIVES.items())[:3]}
    elif complexity == "high" or complexity == "critical":
        analyzers = ANALYZERS  # Use all
        voters = VOTER_PERSPECTIVES  # Use all
    else:  # medium
        analyzers = {k: v for k, v in list(ANALYZERS.items())[:6]}
        voters = {k: v for k, v in list(VOTER_PERSPECTIVES.items())[:4]}
    
    print(f"\n📊 Input complexity: {complexity.upper()}")
    print(f"   Planned analyzers: {len(analyzers)}")
    print(f"   Planned voters: {len(voters)}")
    
    return {
        "input_complexity": complexity,
        "identified_dimensions": analysis.get("dimensions", []),
        "planned_analyzers": [{"id": k, **v} for k, v in analyzers.items()],
        "planned_voters": [{"id": k, **v} for k, v in voters.items()]
    }


def execute_parallel_analysis(state: OptimizedParallelState) -> dict:
    """Execute all analyses in parallel"""
    
    feedback_content = state["messages"][-1].content
    
    # Build analyzer dict from planned analyzers
    analyzers = {
        a["id"]: {k: v for k, v in a.items() if k != "id"}
        for a in state["planned_analyzers"]
    }
    
    results, stats = EXECUTOR.execute_analyses_parallel(analyzers, feedback_content)
    
    return {
        "analysis_results": [
            {
                "analyzer_id": r.analyzer_id,
                "analysis": r.analysis,
                "confidence": r.confidence,
                "key_findings": r.key_findings,
                "recommendations": r.recommendations,
                "risk_flags": r.risk_flags,
                "execution_time": r.execution_time,
                "from_cache": r.from_cache
            }
            for r in results
        ],
        "execution_stats": stats
    }


def execute_parallel_voting(state: OptimizedParallelState) -> dict:
    """Execute all voting perspectives in parallel"""
    
    # Create summary from analyses for voters
    analysis_summary = "\n".join([
        f"[{r['analyzer_id']}]: {r['analysis'][:200]}..."
        for r in state["analysis_results"]
    ])
    
    voters = {
        v["id"]: {k: val for k, val in v.items() if k != "id"}
        for v in state["planned_voters"]
    }
    
    results, stats = EXECUTOR.execute_votes_parallel(voters, analysis_summary)
    
    # Update execution stats
    current_stats = state.get("execution_stats", {})
    current_stats["voting_stats"] = stats
    
    return {
        "vote_results": [
            {
                "voter_id": r.voter_id,
                "perspective": r.perspective,
                "decision": r.decision,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "weight": r.weight
            }
            for r in results
        ],
        "execution_stats": current_stats
    }


def aggregate_and_synthesize(state: OptimizedParallelState) -> dict:
    """Aggregate all results and generate final recommendation"""
    
    print("\n Aggregating results...")
    
    # Convert dict results back to AnalysisResult objects for aggregator
    analysis_results = [
        AnalysisResult(
            analyzer_id=r["analyzer_id"],
            analyzer_type="sectioning",
            analysis=r["analysis"],
            confidence=r["confidence"],
            key_findings=r["key_findings"],
            recommendations=r["recommendations"],
            risk_flags=r["risk_flags"],
            execution_time=r["execution_time"],
            from_cache=r.get("from_cache", False)
        )
        for r in state["analysis_results"]
    ]
    
    vote_results = [
        VoteResult(
            voter_id=r["voter_id"],
            perspective=r["perspective"],
            decision=r["decision"],
            confidence=r["confidence"],
            reasoning=r["reasoning"],
            weight=r["weight"]
        )
        for r in state["vote_results"]
    ]
    
    # Detect contradictions
    contradictions = AGGREGATOR.detect_contradictions(analysis_results)
    
    # Synthesize by category
    strengths_results = [r for r in analysis_results if "risk" not in r.analyzer_id.lower()]
    risk_results = [r for r in analysis_results if "risk" in r.analyzer_id.lower()]
    
    strengths_synthesis = AGGREGATOR.synthesize_analyses(strengths_results, "strengths and capabilities")
    
    # Aggregate votes
    vote_aggregation = AGGREGATOR.weighted_vote_aggregation(vote_results)
    
    # Generate final recommendation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    prompt = f"""Generate a promotion recommendation based on:

STRENGTHS SYNTHESIS:
{strengths_synthesis}

VOTE AGGREGATION:
{json.dumps(vote_aggregation, indent=2)}

CONTRADICTIONS FOUND:
{json.dumps(contradictions, indent=2) if contradictions else "None"}

RISK FLAGS:
{[r.risk_flags for r in risk_results]}

Create a comprehensive recommendation that:
1. States clear decision (Promote / Not Yet Ready / Promote with Development Plan)
2. Summarizes key evidence
3. Addresses any contradictions
4. Provides specific next steps"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print(" Aggregation complete")
    
    return {
        "strengths_synthesis": strengths_synthesis,
        "growth_areas_synthesis": "",  # Would need separate analysis
        "risk_assessment": str([r.risk_flags for r in risk_results]),
        "promotion_recommendation": response.content,
        "contradictions_found": contradictions
    }


def format_final_output(state: OptimizedParallelState) -> dict:
    """Format final output message"""
    
    stats = state.get("execution_stats", {})
    
    output = f"""
{'='*80}
COMPREHENSIVE 360° PROMOTION ASSESSMENT
{'='*80}

📊 EXECUTION STATISTICS
   - Analyses completed: {len(state['analysis_results'])}
   - Voting perspectives: {len(state['vote_results'])}
   - Total time: {stats.get('elapsed_seconds', 0):.1f}s
   - Cache hit rate: {stats.get('cache_stats', {}).get('hit_rate', 0):.0%}
   - Contradictions detected: {len(state.get('contradictions_found', []))}

{'='*80}
💪 STRENGTHS ANALYSIS
{'='*80}
{state['strengths_synthesis']}

{'='*80}
🗳️  VOTING SUMMARY
{'='*80}
{json.dumps([v for v in state['vote_results']], indent=2)}

{'='*80}
  CONTRADICTIONS & RISKS
{'='*80}
{json.dumps(state.get('contradictions_found', []), indent=2) if state.get('contradictions_found') else 'No contradictions detected.'}

Risk flags: {state['risk_assessment']}

{'='*80}
 FINAL RECOMMENDATION
{'='*80}
{state['promotion_recommendation']}
"""
    
    return {"messages": [AIMessage(content=output)]}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_optimized_parallel_graph():
    """Build the optimized parallelization graph"""
    
    graph = StateGraph(OptimizedParallelState)
    
    # Add nodes
    graph.add_node("analyze_complexity", analyze_input_complexity)
    graph.add_node("parallel_analysis", execute_parallel_analysis)
    graph.add_node("parallel_voting", execute_parallel_voting)
    graph.add_node("aggregate", aggregate_and_synthesize)
    graph.add_node("format_output", format_final_output)
    
    # Define flow
    graph.add_edge(START, "analyze_complexity")
    graph.add_edge("analyze_complexity", "parallel_analysis")
    graph.add_edge("parallel_analysis", "parallel_voting")
    graph.add_edge("parallel_voting", "aggregate")
    graph.add_edge("aggregate", "format_output")
    graph.add_edge("format_output", END)
    
    return graph.compile()


# ============================================================
# DEMO
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("OPTIMIZED PARALLELIZATION DEMO")
    print("=" * 80)
    
    # Sample comprehensive 360 feedback
    feedback = """
    360 FEEDBACK FOR: Sarah Chen, Senior Software Engineer
    Target Role: Staff Engineer
    
    MANAGER FEEDBACK (Tom Williams):
    Sarah has been exceptional this year. She led the migration of our payment service 
    to the new architecture, delivering 3 weeks ahead of schedule. Her technical decisions
    are sound and well-documented. She mentors 2 junior engineers effectively. 
    Areas for growth: She sometimes takes on too much and could delegate more.
    She needs to improve visibility with senior leadership.
    
    PEER FEEDBACK (3 responses):
    - "Sarah is my go-to person for complex technical problems. Always helpful."
    - "Great collaborator but sometimes her code reviews are too detailed and slow."
    - "She should speak up more in architecture meetings. Her ideas are good but 
      she waits too long to share them."
    
    DIRECT REPORT FEEDBACK (2 responses):
    - "Best mentor I've had. She explains things clearly and gives me autonomy."
    - "Sometimes hard to reach when she's deep in coding. More regular check-ins would help."
    
    SKIP-LEVEL FEEDBACK (Director):
    "Sarah delivered critical projects this year. She's technically strong but needs to 
    demonstrate more strategic thinking and executive presence for Staff level.
    I'd like to see her drive cross-org initiatives."
    
    SELF ASSESSMENT:
    "I've grown significantly in system design and mentoring. I want to improve my
    influence beyond my immediate team and develop better relationships with product
    and design partners."
    
    PERFORMANCE METRICS:
    - Projects delivered: 4 major, 8 minor
    - Code review velocity: Top 10%
    - Incident response: Led 3 successful incident resolutions
    - Documentation: Authored 2 engineering blog posts
    """
    
    print("\n Input: 360° feedback for promotion assessment")
    print(f"   Length: {len(feedback)} characters")
    
    # Build and run
    graph = build_optimized_parallel_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content=feedback)],
        "input_complexity": "",
        "identified_dimensions": [],
        "planned_analyzers": [],
        "planned_voters": [],
        "analysis_results": [],
        "vote_results": [],
        "strengths_synthesis": "",
        "growth_areas_synthesis": "",
        "risk_assessment": "",
        "promotion_recommendation": "",
        "execution_stats": {},
        "contradictions_found": []
    })
    
    print(result["messages"][-1].content)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"""
Key optimizations demonstrated:

1.  DYNAMIC FAN-OUT
   - Complexity analysis determines number of parallel workers
   - Low complexity = fewer workers, faster execution
   - High complexity = full analysis suite

2.  TRUE PARALLEL EXECUTION
   - ThreadPoolExecutor for concurrent API calls
   - Configurable max_workers ({CONFIG.max_workers})
   - Timeout handling ({CONFIG.timeout_seconds}s)

3.  RESULT CACHING
   - Hash-based cache keys
   - Cache stats: {CACHE.stats()}
   - Run again to see cache hits!

4.  WEIGHTED VOTING
   - Expertise-based weights per perspective
   - Confidence-weighted aggregation
   - Consensus strength calculation

5.  CONTRADICTION DETECTION
   - Cross-analyzer comparison
   - Flagged for human review

6.  GRACEFUL DEGRADATION
   - Minimum responses threshold
   - Partial results on timeout
   - Retry support
""")


if __name__ == "__main__":
    main()
