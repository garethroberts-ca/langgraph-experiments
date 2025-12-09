#!/usr/bin/env python3
"""
LangGraph Example 38: Organisational Psychology Engine
=======================================================

A comprehensive organisational psychology system implementing evidence-based
frameworks for understanding and improving workplace dynamics.

This implementation integrates multiple psychological models:

1. JOB DEMANDS-RESOURCES (JD-R) MODEL
   - Bakker & Demerouti's framework for burnout and engagement
   - Balance between demands (workload, pressure) and resources (autonomy, support)

2. PSYCHOLOGICAL SAFETY (Edmondson)
   - Team learning behaviours
   - Interpersonal risk-taking
   - Voice and speaking up

3. SELF-DETERMINATION THEORY (Deci & Ryan)
   - Autonomy, Competence, Relatedness
   - Intrinsic vs extrinsic motivation

4. ORGANISATIONAL JUSTICE
   - Distributive, Procedural, Interactional fairness
   - Trust and commitment outcomes

5. MASLACH BURNOUT INVENTORY DIMENSIONS
   - Exhaustion, Cynicism, Professional Efficacy

6. TEAM DEVELOPMENT (Tuckman + Lencioni)
   - Forming, Storming, Norming, Performing
   - Five Dysfunctions model

This system demonstrates how LangGraph can orchestrate complex psychological
assessments, generate insights, and recommend interventions.

Author: LangGraph Examples (Organisational Psychology Edition)
"""

import json
import math
import statistics
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
    width = 78
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


def reduce_dict(left: dict | None, right: dict | None) -> dict:
    """Merge two dicts, handling None values."""
    if not left:
        left = {}
    if not right:
        right = {}
    return {**left, **right}


def calculate_percentile(score: float, benchmark_mean: float, benchmark_sd: float) -> int:
    """Calculate percentile rank against benchmark."""
    z_score = (score - benchmark_mean) / benchmark_sd if benchmark_sd > 0 else 0
    # Approximate percentile from z-score
    percentile = int(50 * (1 + math.erf(z_score / math.sqrt(2))))
    return max(1, min(99, percentile))


def risk_level(score: float, thresholds: tuple) -> str:
    """Determine risk level based on thresholds."""
    low, medium = thresholds
    if score >= low:
        return "low"
    elif score >= medium:
        return "medium"
    else:
        return "high"


# =============================================================================
# Comprehensive State Definition
# =============================================================================

class OrganisationalPsychologyState(TypedDict):
    """Comprehensive state for organisational psychology assessment."""
    
    # Organisation metadata
    organisation_name: str
    assessment_date: str
    employee_count: int
    industry: str
    
    # Raw survey data (simulated)
    survey_responses: list[dict]
    
    # JD-R Model outputs
    job_demands: dict
    job_resources: dict
    jdr_balance_score: float
    burnout_risk: str
    engagement_prediction: str
    
    # Psychological Safety outputs
    psych_safety_score: float
    psych_safety_dimensions: dict
    team_learning_index: float
    voice_climate: str
    
    # Self-Determination Theory outputs
    autonomy_score: float
    competence_score: float
    relatedness_score: float
    intrinsic_motivation_index: float
    sdt_needs_profile: dict
    
    # Organisational Justice outputs
    distributive_justice: float
    procedural_justice: float
    interactional_justice: float
    overall_fairness_perception: float
    trust_index: float
    
    # Burnout Assessment outputs
    exhaustion_score: float
    cynicism_score: float
    efficacy_score: float
    burnout_profile: str
    burnout_risk_factors: list[str]
    
    # Team Dynamics outputs
    team_stage: str
    dysfunction_risks: list[dict]
    cohesion_score: float
    conflict_style: str
    
    # Integrated Analysis
    wellbeing_index: float
    engagement_index: float
    culture_health_score: float
    risk_factors: Annotated[list[str], reduce_list]
    protective_factors: Annotated[list[str], reduce_list]
    
    # AI-Generated Insights
    ai_diagnosis: str
    ai_insights: Annotated[list[str], reduce_list]
    
    # Interventions
    priority_interventions: list[dict]
    quick_wins: list[str]
    strategic_initiatives: list[dict]
    
    # Executive Summary
    executive_summary: str
    action_plan: dict


# =============================================================================
# Stage 1: Data Collection and Preprocessing
# =============================================================================

def collect_survey_data(state: OrganisationalPsychologyState) -> dict:
    """
    Simulate collection and preprocessing of survey responses.
    
    In production, this would integrate with Culture Amp's survey system.
    """
    log("STAGE 1: Collecting and preprocessing survey data...")
    
    # Simulate diverse survey responses across the organisation
    # Each response represents aggregated team-level data
    simulated_responses = [
        {
            "team": "Engineering",
            "n": 45,
            "workload": 4.2, "time_pressure": 3.8, "emotional_demands": 3.1,
            "autonomy": 4.1, "skill_variety": 4.3, "social_support": 3.9,
            "feedback": 3.5, "growth_opportunities": 3.8,
            "psych_safety": 3.7, "voice": 3.5, "learning_behaviour": 4.0,
            "autonomy_need": 4.0, "competence_need": 4.2, "relatedness_need": 3.8,
            "distributive_fairness": 3.4, "procedural_fairness": 3.6, "interactional_fairness": 4.0,
            "exhaustion": 3.2, "cynicism": 2.4, "efficacy": 4.1,
            "trust_leadership": 3.7, "commitment": 3.9
        },
        {
            "team": "Product",
            "n": 22,
            "workload": 4.5, "time_pressure": 4.3, "emotional_demands": 3.5,
            "autonomy": 3.6, "skill_variety": 4.0, "social_support": 4.1,
            "feedback": 3.8, "growth_opportunities": 4.0,
            "psych_safety": 4.1, "voice": 4.0, "learning_behaviour": 4.2,
            "autonomy_need": 3.5, "competence_need": 4.0, "relatedness_need": 4.2,
            "distributive_fairness": 3.6, "procedural_fairness": 3.9, "interactional_fairness": 4.2,
            "exhaustion": 3.5, "cynicism": 2.6, "efficacy": 3.9,
            "trust_leadership": 4.0, "commitment": 4.1
        },
        {
            "team": "Sales",
            "n": 38,
            "workload": 4.8, "time_pressure": 4.6, "emotional_demands": 4.2,
            "autonomy": 3.2, "skill_variety": 3.4, "social_support": 3.5,
            "feedback": 4.2, "growth_opportunities": 3.4,
            "psych_safety": 3.2, "voice": 2.9, "learning_behaviour": 3.3,
            "autonomy_need": 3.0, "competence_need": 3.6, "relatedness_need": 3.4,
            "distributive_fairness": 3.0, "procedural_fairness": 3.1, "interactional_fairness": 3.3,
            "exhaustion": 4.0, "cynicism": 3.2, "efficacy": 3.5,
            "trust_leadership": 3.2, "commitment": 3.4
        },
        {
            "team": "Customer Success",
            "n": 28,
            "workload": 4.0, "time_pressure": 3.9, "emotional_demands": 4.0,
            "autonomy": 3.8, "skill_variety": 3.6, "social_support": 4.3,
            "feedback": 3.9, "growth_opportunities": 3.5,
            "psych_safety": 4.0, "voice": 3.8, "learning_behaviour": 3.9,
            "autonomy_need": 3.7, "competence_need": 3.8, "relatedness_need": 4.3,
            "distributive_fairness": 3.5, "procedural_fairness": 3.7, "interactional_fairness": 4.1,
            "exhaustion": 3.3, "cynicism": 2.5, "efficacy": 4.0,
            "trust_leadership": 3.9, "commitment": 4.0
        },
        {
            "team": "People & Culture",
            "n": 12,
            "workload": 3.8, "time_pressure": 3.5, "emotional_demands": 3.8,
            "autonomy": 4.2, "skill_variety": 4.4, "social_support": 4.5,
            "feedback": 4.1, "growth_opportunities": 4.2,
            "psych_safety": 4.5, "voice": 4.3, "learning_behaviour": 4.4,
            "autonomy_need": 4.2, "competence_need": 4.3, "relatedness_need": 4.5,
            "distributive_fairness": 4.0, "procedural_fairness": 4.2, "interactional_fairness": 4.4,
            "exhaustion": 2.8, "cynicism": 2.0, "efficacy": 4.3,
            "trust_leadership": 4.3, "commitment": 4.4
        },
        {
            "team": "Finance",
            "n": 15,
            "workload": 4.3, "time_pressure": 4.4, "emotional_demands": 2.8,
            "autonomy": 3.4, "skill_variety": 3.2, "social_support": 3.6,
            "feedback": 3.3, "growth_opportunities": 3.0,
            "psych_safety": 3.4, "voice": 3.2, "learning_behaviour": 3.1,
            "autonomy_need": 3.3, "competence_need": 3.8, "relatedness_need": 3.4,
            "distributive_fairness": 3.8, "procedural_fairness": 3.5, "interactional_fairness": 3.6,
            "exhaustion": 3.4, "cynicism": 2.8, "efficacy": 3.8,
            "trust_leadership": 3.5, "commitment": 3.6
        }
    ]
    
    total_n = sum(r["n"] for r in simulated_responses)
    log(f"Collected responses from {len(simulated_responses)} teams ({total_n} employees)", indent=1)
    
    return {
        "survey_responses": simulated_responses,
        "employee_count": total_n
    }


# =============================================================================
# Stage 2: Job Demands-Resources Analysis
# =============================================================================

def analyse_jdr_model(state: OrganisationalPsychologyState) -> dict:
    """
    Analyse Job Demands-Resources balance using Bakker & Demerouti's model.
    
    The JD-R model proposes two parallel processes:
    1. Health impairment: High demands -> exhaustion -> health problems
    2. Motivational: High resources -> engagement -> performance
    
    The balance between demands and resources predicts burnout and engagement.
    """
    log("STAGE 2: Analysing Job Demands-Resources model...")
    
    responses = state["survey_responses"]
    
    # Calculate weighted averages (weighted by team size)
    total_n = sum(r["n"] for r in responses)
    
    # Job Demands (higher = more demanding)
    demands = {
        "workload": sum(r["workload"] * r["n"] for r in responses) / total_n,
        "time_pressure": sum(r["time_pressure"] * r["n"] for r in responses) / total_n,
        "emotional_demands": sum(r["emotional_demands"] * r["n"] for r in responses) / total_n
    }
    overall_demands = statistics.mean(demands.values())
    
    # Job Resources (higher = more resources)
    resources = {
        "autonomy": sum(r["autonomy"] * r["n"] for r in responses) / total_n,
        "skill_variety": sum(r["skill_variety"] * r["n"] for r in responses) / total_n,
        "social_support": sum(r["social_support"] * r["n"] for r in responses) / total_n,
        "feedback": sum(r["feedback"] * r["n"] for r in responses) / total_n,
        "growth_opportunities": sum(r["growth_opportunities"] * r["n"] for r in responses) / total_n
    }
    overall_resources = statistics.mean(resources.values())
    
    # JD-R Balance Score: Resources minus Demands (positive = healthy balance)
    # Normalised to -2 to +2 scale
    balance = overall_resources - overall_demands
    
    # Determine risk levels
    if balance < -0.5:
        burnout_risk = "high"
        engagement_pred = "low"
    elif balance < 0:
        burnout_risk = "moderate"
        engagement_pred = "moderate"
    elif balance < 0.5:
        burnout_risk = "low"
        engagement_pred = "moderate-high"
    else:
        burnout_risk = "very low"
        engagement_pred = "high"
    
    log(f"Overall demands: {overall_demands:.2f}/5.0", indent=1)
    log(f"Overall resources: {overall_resources:.2f}/5.0", indent=1)
    log(f"JD-R balance: {balance:+.2f} (Burnout risk: {burnout_risk})", indent=1)
    
    # Identify specific demand hotspots
    demand_hotspots = [k for k, v in demands.items() if v > 4.0]
    resource_gaps = [k for k, v in resources.items() if v < 3.5]
    
    risk_factors = []
    protective_factors = []
    
    if demand_hotspots:
        risk_factors.append(f"High demands in: {', '.join(demand_hotspots)}")
    if resource_gaps:
        risk_factors.append(f"Resource gaps in: {', '.join(resource_gaps)}")
    
    resource_strengths = [k for k, v in resources.items() if v >= 4.0]
    if resource_strengths:
        protective_factors.append(f"Strong resources: {', '.join(resource_strengths)}")
    
    return {
        "job_demands": demands,
        "job_resources": resources,
        "jdr_balance_score": balance,
        "burnout_risk": burnout_risk,
        "engagement_prediction": engagement_pred,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors
    }


# =============================================================================
# Stage 3: Psychological Safety Assessment
# =============================================================================

def assess_psychological_safety(state: OrganisationalPsychologyState) -> dict:
    """
    Assess psychological safety using Edmondson's framework.
    
    Psychological safety is the shared belief that the team is safe for
    interpersonal risk-taking. It enables:
    - Speaking up about concerns
    - Admitting mistakes
    - Asking for help
    - Challenging the status quo
    """
    log("STAGE 3: Assessing psychological safety...")
    
    responses = state["survey_responses"]
    total_n = sum(r["n"] for r in responses)
    
    # Calculate organisation-wide psychological safety
    psych_safety = sum(r["psych_safety"] * r["n"] for r in responses) / total_n
    voice = sum(r["voice"] * r["n"] for r in responses) / total_n
    learning = sum(r["learning_behaviour"] * r["n"] for r in responses) / total_n
    
    # Dimension breakdown
    dimensions = {
        "safety_to_speak_up": voice,
        "safety_to_fail": psych_safety * 0.9,  # Slightly lower typically
        "safety_to_challenge": (psych_safety + voice) / 2,
        "safety_to_ask_for_help": psych_safety * 1.05
    }
    
    # Team learning index (Edmondson's key outcome variable)
    team_learning = (psych_safety * 0.4 + voice * 0.3 + learning * 0.3)
    
    # Voice climate classification
    if voice >= 4.0:
        voice_climate = "thriving"
    elif voice >= 3.5:
        voice_climate = "healthy"
    elif voice >= 3.0:
        voice_climate = "constrained"
    else:
        voice_climate = "silenced"
    
    log(f"Psychological safety: {psych_safety:.2f}/5.0", indent=1)
    log(f"Voice climate: {voice_climate}", indent=1)
    log(f"Team learning index: {team_learning:.2f}/5.0", indent=1)
    
    # Identify teams at risk
    low_safety_teams = [r["team"] for r in responses if r["psych_safety"] < 3.5]
    if low_safety_teams:
        return {
            "psych_safety_score": psych_safety,
            "psych_safety_dimensions": dimensions,
            "team_learning_index": team_learning,
            "voice_climate": voice_climate,
            "risk_factors": [f"Low psychological safety in: {', '.join(low_safety_teams)}"]
        }
    
    return {
        "psych_safety_score": psych_safety,
        "psych_safety_dimensions": dimensions,
        "team_learning_index": team_learning,
        "voice_climate": voice_climate,
        "protective_factors": ["Healthy psychological safety across most teams"]
    }


# =============================================================================
# Stage 4: Self-Determination Theory Analysis
# =============================================================================

def analyse_sdt_needs(state: OrganisationalPsychologyState) -> dict:
    """
    Analyse basic psychological needs using Self-Determination Theory.
    
    SDT (Deci & Ryan) identifies three universal psychological needs:
    1. Autonomy: Need to feel volitional and self-directed
    2. Competence: Need to feel effective and capable
    3. Relatedness: Need to feel connected to others
    
    When these needs are satisfied, intrinsic motivation flourishes.
    """
    log("STAGE 4: Analysing Self-Determination Theory needs...")
    
    responses = state["survey_responses"]
    total_n = sum(r["n"] for r in responses)
    
    # Calculate need satisfaction scores
    autonomy = sum(r["autonomy_need"] * r["n"] for r in responses) / total_n
    competence = sum(r["competence_need"] * r["n"] for r in responses) / total_n
    relatedness = sum(r["relatedness_need"] * r["n"] for r in responses) / total_n
    
    # Intrinsic motivation index (geometric mean to penalise imbalance)
    intrinsic_motivation = (autonomy * competence * relatedness) ** (1/3)
    
    # Need profile categorisation
    needs_profile = {
        "autonomy": {
            "score": autonomy,
            "satisfaction": "high" if autonomy >= 4.0 else "moderate" if autonomy >= 3.5 else "low",
            "priority": "maintain" if autonomy >= 4.0 else "develop" if autonomy >= 3.5 else "urgent"
        },
        "competence": {
            "score": competence,
            "satisfaction": "high" if competence >= 4.0 else "moderate" if competence >= 3.5 else "low",
            "priority": "maintain" if competence >= 4.0 else "develop" if competence >= 3.5 else "urgent"
        },
        "relatedness": {
            "score": relatedness,
            "satisfaction": "high" if relatedness >= 4.0 else "moderate" if relatedness >= 3.5 else "low",
            "priority": "maintain" if relatedness >= 4.0 else "develop" if relatedness >= 3.5 else "urgent"
        }
    }
    
    log(f"Autonomy: {autonomy:.2f}/5.0 ({needs_profile['autonomy']['satisfaction']})", indent=1)
    log(f"Competence: {competence:.2f}/5.0 ({needs_profile['competence']['satisfaction']})", indent=1)
    log(f"Relatedness: {relatedness:.2f}/5.0 ({needs_profile['relatedness']['satisfaction']})", indent=1)
    log(f"Intrinsic motivation index: {intrinsic_motivation:.2f}/5.0", indent=1)
    
    # Identify thwarted needs
    thwarted = [k for k, v in needs_profile.items() if v["priority"] == "urgent"]
    
    risk_factors = []
    protective_factors = []
    
    if thwarted:
        risk_factors.append(f"Thwarted psychological needs: {', '.join(thwarted)}")
    
    satisfied = [k for k, v in needs_profile.items() if v["satisfaction"] == "high"]
    if satisfied:
        protective_factors.append(f"Well-satisfied needs: {', '.join(satisfied)}")
    
    return {
        "autonomy_score": autonomy,
        "competence_score": competence,
        "relatedness_score": relatedness,
        "intrinsic_motivation_index": intrinsic_motivation,
        "sdt_needs_profile": needs_profile,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors
    }


# =============================================================================
# Stage 5: Organisational Justice Assessment
# =============================================================================

def assess_organisational_justice(state: OrganisationalPsychologyState) -> dict:
    """
    Assess organisational justice perceptions.
    
    Three dimensions of justice (Colquitt's model):
    1. Distributive: Fairness of outcomes (pay, recognition, workload)
    2. Procedural: Fairness of decision-making processes
    3. Interactional: Fairness in interpersonal treatment
    
    Justice perceptions strongly predict trust, commitment, and OCB.
    """
    log("STAGE 5: Assessing organisational justice...")
    
    responses = state["survey_responses"]
    total_n = sum(r["n"] for r in responses)
    
    # Calculate justice dimensions
    distributive = sum(r["distributive_fairness"] * r["n"] for r in responses) / total_n
    procedural = sum(r["procedural_fairness"] * r["n"] for r in responses) / total_n
    interactional = sum(r["interactional_fairness"] * r["n"] for r in responses) / total_n
    
    # Overall fairness (weighted - interactional often matters most for day-to-day)
    overall_fairness = (distributive * 0.3 + procedural * 0.35 + interactional * 0.35)
    
    # Trust index (derived from justice and trust items)
    trust = sum(r["trust_leadership"] * r["n"] for r in responses) / total_n
    trust_index = (overall_fairness * 0.6 + trust * 0.4)
    
    log(f"Distributive justice: {distributive:.2f}/5.0", indent=1)
    log(f"Procedural justice: {procedural:.2f}/5.0", indent=1)
    log(f"Interactional justice: {interactional:.2f}/5.0", indent=1)
    log(f"Trust index: {trust_index:.2f}/5.0", indent=1)
    
    # Identify justice concerns
    risk_factors = []
    protective_factors = []
    
    if distributive < 3.5:
        risk_factors.append("Distributive justice concerns (pay/recognition equity)")
    if procedural < 3.5:
        risk_factors.append("Procedural justice concerns (decision-making transparency)")
    if interactional < 3.5:
        risk_factors.append("Interactional justice concerns (respect/dignity)")
    
    if overall_fairness >= 4.0:
        protective_factors.append("Strong overall justice perceptions")
    if trust_index >= 4.0:
        protective_factors.append("High organisational trust")
    
    return {
        "distributive_justice": distributive,
        "procedural_justice": procedural,
        "interactional_justice": interactional,
        "overall_fairness_perception": overall_fairness,
        "trust_index": trust_index,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors
    }


# =============================================================================
# Stage 6: Burnout Assessment (Maslach Model)
# =============================================================================

def assess_burnout(state: OrganisationalPsychologyState) -> dict:
    """
    Assess burnout using Maslach's three-dimensional model.
    
    Burnout dimensions:
    1. Exhaustion: Feeling emotionally drained and depleted
    2. Cynicism: Negative, detached attitude toward work
    3. Reduced Efficacy: Feelings of incompetence and lack of achievement
    
    Burnout develops progressively and requires early intervention.
    """
    log("STAGE 6: Assessing burnout dimensions...")
    
    responses = state["survey_responses"]
    total_n = sum(r["n"] for r in responses)
    
    # Calculate burnout dimensions (Note: higher = worse for exhaustion/cynicism)
    exhaustion = sum(r["exhaustion"] * r["n"] for r in responses) / total_n
    cynicism = sum(r["cynicism"] * r["n"] for r in responses) / total_n
    efficacy = sum(r["efficacy"] * r["n"] for r in responses) / total_n  # Higher = better
    
    # Burnout profile classification
    if exhaustion >= 4.0 and cynicism >= 3.5:
        profile = "burnout"
    elif exhaustion >= 3.5 or cynicism >= 3.0:
        profile = "overextended"
    elif efficacy < 3.5:
        profile = "ineffective"
    elif exhaustion < 3.0 and cynicism < 2.5 and efficacy >= 4.0:
        profile = "engaged"
    else:
        profile = "moderate"
    
    log(f"Exhaustion: {exhaustion:.2f}/5.0 (lower is better)", indent=1)
    log(f"Cynicism: {cynicism:.2f}/5.0 (lower is better)", indent=1)
    log(f"Professional efficacy: {efficacy:.2f}/5.0 (higher is better)", indent=1)
    log(f"Burnout profile: {profile}", indent=1)
    
    # Identify burnout risk factors
    burnout_risks = []
    
    # Check for high-exhaustion teams
    high_exhaustion_teams = [r["team"] for r in responses if r["exhaustion"] >= 3.8]
    if high_exhaustion_teams:
        burnout_risks.append(f"High exhaustion in: {', '.join(high_exhaustion_teams)}")
    
    # Check JD-R connection
    if state.get("jdr_balance_score", 0) < -0.3:
        burnout_risks.append("JD-R imbalance contributing to exhaustion risk")
    
    # Check for spreading cynicism
    high_cynicism_teams = [r["team"] for r in responses if r["cynicism"] >= 3.0]
    if high_cynicism_teams:
        burnout_risks.append(f"Elevated cynicism in: {', '.join(high_cynicism_teams)}")
    
    return {
        "exhaustion_score": exhaustion,
        "cynicism_score": cynicism,
        "efficacy_score": efficacy,
        "burnout_profile": profile,
        "burnout_risk_factors": burnout_risks,
        "risk_factors": burnout_risks if profile in ["burnout", "overextended"] else []
    }


# =============================================================================
# Stage 7: Team Dynamics Analysis
# =============================================================================

def analyse_team_dynamics(state: OrganisationalPsychologyState) -> dict:
    """
    Analyse team dynamics using Tuckman and Lencioni frameworks.
    
    Tuckman's stages: Forming -> Storming -> Norming -> Performing
    
    Lencioni's Five Dysfunctions (in order):
    1. Absence of Trust (invulnerability)
    2. Fear of Conflict (artificial harmony)
    3. Lack of Commitment (ambiguity)
    4. Avoidance of Accountability (low standards)
    5. Inattention to Results (status/ego)
    """
    log("STAGE 7: Analysing team dynamics...")
    
    responses = state["survey_responses"]
    total_n = sum(r["n"] for r in responses)
    
    # Derive team dynamics indicators from existing data
    trust = state.get("trust_index", 3.5)
    psych_safety = state.get("psych_safety_score", 3.5)
    voice = sum(r["voice"] * r["n"] for r in responses) / total_n
    commitment = sum(r["commitment"] * r["n"] for r in responses) / total_n
    
    # Cohesion score (composite)
    cohesion = (trust * 0.3 + psych_safety * 0.3 + 
                state.get("relatedness_score", 3.5) * 0.2 + 
                commitment * 0.2)
    
    # Determine team stage based on indicators
    if cohesion >= 4.2 and trust >= 4.0 and voice >= 4.0:
        stage = "performing"
    elif cohesion >= 3.8 and trust >= 3.5:
        stage = "norming"
    elif voice < 3.5 or psych_safety < 3.5:
        stage = "storming"
    else:
        stage = "forming"
    
    # Assess Lencioni dysfunction risks
    dysfunctions = []
    
    if trust < 3.5:
        dysfunctions.append({
            "dysfunction": "Absence of Trust",
            "severity": "high" if trust < 3.0 else "moderate",
            "indicator": f"Trust index: {trust:.2f}",
            "intervention": "Vulnerability-based trust exercises, personal histories"
        })
    
    if voice < 3.5 or psych_safety < 3.5:
        dysfunctions.append({
            "dysfunction": "Fear of Conflict",
            "severity": "high" if voice < 3.0 else "moderate",
            "indicator": f"Voice climate: {voice:.2f}",
            "intervention": "Mining for conflict, permission to push back"
        })
    
    if commitment < 3.5:
        dysfunctions.append({
            "dysfunction": "Lack of Commitment",
            "severity": "high" if commitment < 3.0 else "moderate",
            "indicator": f"Commitment: {commitment:.2f}",
            "intervention": "Cascading messaging, deadline clarity, worst-case scenario planning"
        })
    
    # Conflict style inference
    if voice >= 4.0 and psych_safety >= 4.0:
        conflict_style = "constructive"
    elif voice >= 3.5:
        conflict_style = "cautious"
    elif voice < 3.0:
        conflict_style = "avoidant"
    else:
        conflict_style = "mixed"
    
    log(f"Team stage: {stage.upper()}", indent=1)
    log(f"Cohesion score: {cohesion:.2f}/5.0", indent=1)
    log(f"Conflict style: {conflict_style}", indent=1)
    log(f"Dysfunction risks identified: {len(dysfunctions)}", indent=1)
    
    return {
        "team_stage": stage,
        "dysfunction_risks": dysfunctions,
        "cohesion_score": cohesion,
        "conflict_style": conflict_style,
        "risk_factors": [d["dysfunction"] for d in dysfunctions if d["severity"] == "high"]
    }


# =============================================================================
# Stage 8: Integrated Analysis and Diagnosis
# =============================================================================

def generate_integrated_analysis(state: OrganisationalPsychologyState) -> dict:
    """
    Generate integrated analysis combining all psychological frameworks.
    
    Uses AI to synthesise findings and identify patterns across models.
    """
    log("STAGE 8: Generating integrated analysis...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Calculate composite indices
    wellbeing_index = (
        (5 - state.get("exhaustion_score", 3)) * 0.25 +  # Invert exhaustion
        (5 - state.get("cynicism_score", 3)) * 0.15 +     # Invert cynicism
        state.get("efficacy_score", 3.5) * 0.15 +
        state.get("psych_safety_score", 3.5) * 0.15 +
        state.get("intrinsic_motivation_index", 3.5) * 0.15 +
        state.get("trust_index", 3.5) * 0.15
    )
    
    engagement_index = (
        state.get("intrinsic_motivation_index", 3.5) * 0.25 +
        state.get("efficacy_score", 3.5) * 0.20 +
        state.get("cohesion_score", 3.5) * 0.20 +
        (5 + state.get("jdr_balance_score", 0)) / 2 * 0.35  # Normalise JD-R to 0-5
    )
    
    culture_health = (
        state.get("psych_safety_score", 3.5) * 0.25 +
        state.get("trust_index", 3.5) * 0.25 +
        state.get("overall_fairness_perception", 3.5) * 0.25 +
        state.get("team_learning_index", 3.5) * 0.25
    )
    
    log(f"Wellbeing index: {wellbeing_index:.2f}/5.0", indent=1)
    log(f"Engagement index: {engagement_index:.2f}/5.0", indent=1)
    log(f"Culture health score: {culture_health:.2f}/5.0", indent=1)
    
    # Generate AI diagnosis
    diagnosis_prompt = f"""As an organisational psychologist, synthesise these assessment findings into a diagnostic summary:

ORGANISATION: {state['organisation_name']}
EMPLOYEES: {state['employee_count']}

JOB DEMANDS-RESOURCES:
- Balance score: {state.get('jdr_balance_score', 0):+.2f}
- Burnout risk: {state.get('burnout_risk', 'unknown')}
- Key demands: {state.get('job_demands', {})}
- Key resources: {state.get('job_resources', {})}

PSYCHOLOGICAL SAFETY:
- Score: {state.get('psych_safety_score', 0):.2f}/5.0
- Voice climate: {state.get('voice_climate', 'unknown')}
- Team learning: {state.get('team_learning_index', 0):.2f}/5.0

SELF-DETERMINATION THEORY:
- Autonomy: {state.get('autonomy_score', 0):.2f}/5.0
- Competence: {state.get('competence_score', 0):.2f}/5.0
- Relatedness: {state.get('relatedness_score', 0):.2f}/5.0
- Intrinsic motivation: {state.get('intrinsic_motivation_index', 0):.2f}/5.0

ORGANISATIONAL JUSTICE:
- Distributive: {state.get('distributive_justice', 0):.2f}/5.0
- Procedural: {state.get('procedural_justice', 0):.2f}/5.0
- Interactional: {state.get('interactional_justice', 0):.2f}/5.0
- Trust index: {state.get('trust_index', 0):.2f}/5.0

BURNOUT ASSESSMENT:
- Profile: {state.get('burnout_profile', 'unknown')}
- Exhaustion: {state.get('exhaustion_score', 0):.2f}/5.0
- Cynicism: {state.get('cynicism_score', 0):.2f}/5.0

TEAM DYNAMICS:
- Stage: {state.get('team_stage', 'unknown')}
- Cohesion: {state.get('cohesion_score', 0):.2f}/5.0
- Dysfunction risks: {len(state.get('dysfunction_risks', []))}

COMPOSITE SCORES:
- Wellbeing: {wellbeing_index:.2f}/5.0
- Engagement: {engagement_index:.2f}/5.0
- Culture health: {culture_health:.2f}/5.0

RISK FACTORS: {state.get('risk_factors', [])}
PROTECTIVE FACTORS: {state.get('protective_factors', [])}

Provide:
1. A diagnostic summary (3-4 sentences) identifying the primary psychological dynamics
2. The most critical systemic pattern connecting multiple findings
3. The single most important lever for improvement

Be specific and grounded in the data. Use organisational psychology terminology."""

    response = llm.invoke(diagnosis_prompt)
    diagnosis = response.content.strip()
    
    # Generate specific insights
    insights = []
    
    if state.get("jdr_balance_score", 0) < 0 and state.get("exhaustion_score", 0) > 3.5:
        insights.append(
            "[JD-R + Burnout] The demand-resource imbalance is manifesting in elevated exhaustion. "
            "This follows the health impairment pathway predicted by JD-R theory."
        )
    
    if state.get("psych_safety_score", 0) < 3.5 and state.get("voice_climate") in ["constrained", "silenced"]:
        insights.append(
            "[Psychological Safety] Constrained voice climate suggests interpersonal risk is perceived as too high. "
            "This will impair learning, innovation, and problem identification."
        )
    
    if state.get("distributive_justice", 0) < state.get("procedural_justice", 0):
        insights.append(
            "[Justice] Procedural justice exceeds distributive justice, suggesting processes are fairer than outcomes. "
            "Review compensation, recognition, and workload distribution equity."
        )
    
    autonomy = state.get("autonomy_score", 3.5)
    if autonomy < 3.5 and state.get("intrinsic_motivation_index", 3.5) < 3.5:
        insights.append(
            "[SDT] Low autonomy is suppressing intrinsic motivation. "
            "Micromanagement or rigid processes may be undermining self-determination."
        )
    
    if state.get("burnout_profile") == "overextended" and state.get("efficacy_score", 4) >= 4.0:
        insights.append(
            "[Burnout Pattern] High efficacy with elevated exhaustion indicates productive but unsustainable performance. "
            "Without intervention, this profile typically progresses to full burnout."
        )
    
    return {
        "wellbeing_index": wellbeing_index,
        "engagement_index": engagement_index,
        "culture_health_score": culture_health,
        "ai_diagnosis": diagnosis,
        "ai_insights": insights
    }


# =============================================================================
# Stage 9: Intervention Recommendations
# =============================================================================

def generate_interventions(state: OrganisationalPsychologyState) -> dict:
    """
    Generate evidence-based intervention recommendations.
    
    Interventions are categorised by:
    - Timeline (quick wins vs strategic)
    - Target level (individual, team, organisation)
    - Primary framework addressed
    """
    log("STAGE 9: Generating intervention recommendations...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    
    priority_interventions = []
    quick_wins = []
    strategic_initiatives = []
    
    # JD-R based interventions
    if state.get("jdr_balance_score", 0) < 0:
        priority_interventions.append({
            "intervention": "Job Crafting Programme",
            "framework": "JD-R",
            "target": "Individual + Team",
            "description": "Enable employees to proactively redesign their jobs to increase resources and optimise demands",
            "evidence_base": "Bakker et al. (2012) - Job crafting increases engagement and reduces burnout",
            "timeline": "3-6 months",
            "success_metrics": ["JD-R balance improvement", "Engagement survey scores", "Voluntary turnover"]
        })
        
        quick_wins.append("Implement 'no meeting' blocks to reduce time pressure demands")
        quick_wins.append("Increase feedback frequency through lightweight check-in tools")
    
    # Psychological safety interventions
    if state.get("psych_safety_score", 5) < 3.8:
        priority_interventions.append({
            "intervention": "Psychological Safety Workshop Series",
            "framework": "Edmondson",
            "target": "Team",
            "description": "Leader-led workshops to establish team norms around interpersonal risk-taking",
            "evidence_base": "Edmondson (1999) - Team psychological safety predicts learning behaviour",
            "timeline": "2-4 months",
            "success_metrics": ["Psychological safety score", "Speaking up behaviour", "Error reporting rates"]
        })
        
        quick_wins.append("Leaders model vulnerability by sharing own mistakes in team meetings")
        quick_wins.append("Implement 'what I learned from failure' sharing in retrospectives")
    
    # SDT interventions
    if state.get("autonomy_score", 5) < 3.5:
        priority_interventions.append({
            "intervention": "Autonomy-Supportive Leadership Development",
            "framework": "SDT",
            "target": "Managers",
            "description": "Train managers in autonomy-supportive behaviours: providing choice, acknowledging feelings, offering rationale",
            "evidence_base": "Deci & Ryan (2000) - Autonomy support increases intrinsic motivation",
            "timeline": "4-6 months",
            "success_metrics": ["Autonomy scores", "Intrinsic motivation index", "Manager effectiveness ratings"]
        })
        
        quick_wins.append("Increase decision-making authority at team level")
    
    # Justice interventions
    if state.get("distributive_justice", 5) < 3.5:
        strategic_initiatives.append({
            "initiative": "Compensation and Recognition Equity Review",
            "framework": "Organisational Justice",
            "description": "Comprehensive audit of pay equity, promotion patterns, and recognition distribution",
            "timeline": "6-12 months",
            "stakeholders": ["People & Culture", "Finance", "Leadership Team"],
            "expected_outcomes": ["Improved distributive justice", "Reduced pay-related turnover", "Enhanced employer brand"]
        })
    
    if state.get("procedural_justice", 5) < 3.5:
        priority_interventions.append({
            "intervention": "Decision Transparency Initiative",
            "framework": "Organisational Justice",
            "target": "Organisation",
            "description": "Increase transparency in how decisions are made, who is involved, and how input is considered",
            "evidence_base": "Colquitt et al. (2001) - Procedural justice predicts trust and commitment",
            "timeline": "2-3 months",
            "success_metrics": ["Procedural justice scores", "Trust index", "Decision acceptance rates"]
        })
    
    # Burnout interventions
    if state.get("burnout_profile") in ["burnout", "overextended"]:
        priority_interventions.append({
            "intervention": "Workload Sustainability Review",
            "framework": "Maslach + JD-R",
            "target": "Organisation + Team",
            "description": "Systematic review of workload distribution, staffing levels, and sustainable capacity",
            "evidence_base": "Maslach & Leiter (2016) - Workload is primary predictor of exhaustion",
            "timeline": "1-3 months",
            "success_metrics": ["Exhaustion scores", "Overtime hours", "Sick leave patterns"]
        })
        
        quick_wins.append("Protected recovery time - enforce maximum meeting hours per day")
        quick_wins.append("Manager training on recognising early burnout warning signs")
    
    # Team dynamics interventions
    for dysfunction in state.get("dysfunction_risks", []):
        if dysfunction.get("severity") == "high":
            priority_interventions.append({
                "intervention": f"Address {dysfunction['dysfunction']}",
                "framework": "Lencioni",
                "target": "Team",
                "description": dysfunction.get("intervention", "Targeted team development"),
                "evidence_base": "Lencioni (2002) - Dysfunctions build on each other; address foundational issues first",
                "timeline": "2-4 months",
                "success_metrics": ["Team cohesion score", "Conflict constructiveness", "Commitment levels"]
            })
    
    log(f"Priority interventions: {len(priority_interventions)}", indent=1)
    log(f"Quick wins: {len(quick_wins)}", indent=1)
    log(f"Strategic initiatives: {len(strategic_initiatives)}", indent=1)
    
    return {
        "priority_interventions": priority_interventions,
        "quick_wins": quick_wins,
        "strategic_initiatives": strategic_initiatives
    }


# =============================================================================
# Stage 10: Executive Summary and Action Plan
# =============================================================================

def generate_executive_summary(state: OrganisationalPsychologyState) -> dict:
    """
    Generate executive summary and comprehensive action plan.
    """
    log("STAGE 10: Generating executive summary...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    summary_prompt = f"""As a senior organisational psychologist, write an executive summary for the leadership team.

ORGANISATION: {state['organisation_name']}
ASSESSMENT DATE: {state['assessment_date']}
EMPLOYEES ASSESSED: {state['employee_count']}

KEY METRICS:
- Wellbeing Index: {state.get('wellbeing_index', 0):.2f}/5.0
- Engagement Index: {state.get('engagement_index', 0):.2f}/5.0  
- Culture Health: {state.get('culture_health_score', 0):.2f}/5.0
- Psychological Safety: {state.get('psych_safety_score', 0):.2f}/5.0
- Trust Index: {state.get('trust_index', 0):.2f}/5.0

DIAGNOSTIC FINDINGS:
{state.get('ai_diagnosis', 'No diagnosis available')}

KEY INSIGHTS:
{chr(10).join(['- ' + i for i in state.get('ai_insights', [])])}

RISK FACTORS:
{chr(10).join(['- ' + r for r in state.get('risk_factors', [])])}

PROTECTIVE FACTORS:
{chr(10).join(['- ' + p for p in state.get('protective_factors', [])])}

RECOMMENDED PRIORITY INTERVENTIONS: {len(state.get('priority_interventions', []))}
QUICK WINS IDENTIFIED: {len(state.get('quick_wins', []))}

Write an executive summary (4-5 paragraphs) that:
1. Opens with the overall organisational health assessment
2. Highlights the most significant finding and its business implications
3. Identifies the primary risk requiring attention
4. Recommends the top 2-3 actions for the next quarter
5. Closes with a forward-looking statement about potential

Use British English. Be direct and actionable. Ground recommendations in the psychological evidence."""

    response = llm.invoke(summary_prompt)
    executive_summary = response.content.strip()
    
    # Create structured action plan
    action_plan = {
        "immediate_actions": state.get("quick_wins", [])[:5],
        "30_day_priorities": [
            i["intervention"] for i in state.get("priority_interventions", [])[:3]
        ],
        "90_day_initiatives": [
            i["intervention"] for i in state.get("priority_interventions", [])[3:5]
        ] + [
            i["initiative"] for i in state.get("strategic_initiatives", [])[:2]
        ],
        "success_metrics": [
            {"metric": "Wellbeing Index", "current": state.get("wellbeing_index", 0), "target": min(4.5, state.get("wellbeing_index", 0) + 0.3)},
            {"metric": "Engagement Index", "current": state.get("engagement_index", 0), "target": min(4.5, state.get("engagement_index", 0) + 0.3)},
            {"metric": "Psychological Safety", "current": state.get("psych_safety_score", 0), "target": min(4.5, state.get("psych_safety_score", 0) + 0.3)},
            {"metric": "Exhaustion (inverse)", "current": 5 - state.get("exhaustion_score", 3), "target": min(4.0, 5 - state.get("exhaustion_score", 3) + 0.5)}
        ],
        "review_cadence": "Monthly leadership review, quarterly full reassessment",
        "accountable_parties": ["Chief People Officer", "Leadership Team", "People & Culture"]
    }
    
    return {
        "executive_summary": executive_summary,
        "action_plan": action_plan
    }


# =============================================================================
# Main Demonstration
# =============================================================================

def run_full_assessment():
    """
    Run the complete organisational psychology assessment.
    """
    print_banner("Organisational Psychology Engine")
    print("Integrating: JD-R | Psychological Safety | SDT | Justice | Burnout | Team Dynamics\n")
    
    # Build the assessment graph
    builder = StateGraph(OrganisationalPsychologyState)
    
    # Add all assessment stages
    builder.add_node("collect_data", collect_survey_data)
    builder.add_node("jdr_analysis", analyse_jdr_model)
    builder.add_node("psych_safety", assess_psychological_safety)
    builder.add_node("sdt_analysis", analyse_sdt_needs)
    builder.add_node("justice_assessment", assess_organisational_justice)
    builder.add_node("burnout_assessment", assess_burnout)
    builder.add_node("team_dynamics", analyse_team_dynamics)
    builder.add_node("integrated_analysis", generate_integrated_analysis)
    builder.add_node("interventions", generate_interventions)
    builder.add_node("executive_summary", generate_executive_summary)
    
    # Define the assessment flow
    builder.add_edge(START, "collect_data")
    builder.add_edge("collect_data", "jdr_analysis")
    builder.add_edge("jdr_analysis", "psych_safety")
    builder.add_edge("psych_safety", "sdt_analysis")
    builder.add_edge("sdt_analysis", "justice_assessment")
    builder.add_edge("justice_assessment", "burnout_assessment")
    builder.add_edge("burnout_assessment", "team_dynamics")
    builder.add_edge("team_dynamics", "integrated_analysis")
    builder.add_edge("integrated_analysis", "interventions")
    builder.add_edge("interventions", "executive_summary")
    builder.add_edge("executive_summary", END)
    
    graph = builder.compile()
    
    # Run assessment
    initial_state = {
        "organisation_name": "Innovate Technologies",
        "assessment_date": datetime.now().strftime("%Y-%m-%d"),
        "employee_count": 0,
        "industry": "Technology",
        "survey_responses": [],
        "job_demands": {},
        "job_resources": {},
        "jdr_balance_score": 0.0,
        "burnout_risk": "",
        "engagement_prediction": "",
        "psych_safety_score": 0.0,
        "psych_safety_dimensions": {},
        "team_learning_index": 0.0,
        "voice_climate": "",
        "autonomy_score": 0.0,
        "competence_score": 0.0,
        "relatedness_score": 0.0,
        "intrinsic_motivation_index": 0.0,
        "sdt_needs_profile": {},
        "distributive_justice": 0.0,
        "procedural_justice": 0.0,
        "interactional_justice": 0.0,
        "overall_fairness_perception": 0.0,
        "trust_index": 0.0,
        "exhaustion_score": 0.0,
        "cynicism_score": 0.0,
        "efficacy_score": 0.0,
        "burnout_profile": "",
        "burnout_risk_factors": [],
        "team_stage": "",
        "dysfunction_risks": [],
        "cohesion_score": 0.0,
        "conflict_style": "",
        "wellbeing_index": 0.0,
        "engagement_index": 0.0,
        "culture_health_score": 0.0,
        "risk_factors": [],
        "protective_factors": [],
        "ai_diagnosis": "",
        "ai_insights": [],
        "priority_interventions": [],
        "quick_wins": [],
        "strategic_initiatives": [],
        "executive_summary": "",
        "action_plan": {}
    }
    
    result = graph.invoke(initial_state)
    
    # Display comprehensive results
    print_banner("ASSESSMENT RESULTS")
    
    print("COMPOSITE INDICES")
    print("-" * 50)
    print(f"  Wellbeing Index:      {result['wellbeing_index']:.2f}/5.0  ", end="")
    print(f"{'[GOOD]' if result['wellbeing_index'] >= 4.0 else '[ATTENTION]' if result['wellbeing_index'] >= 3.5 else '[CONCERN]'}")
    print(f"  Engagement Index:     {result['engagement_index']:.2f}/5.0  ", end="")
    print(f"{'[GOOD]' if result['engagement_index'] >= 4.0 else '[ATTENTION]' if result['engagement_index'] >= 3.5 else '[CONCERN]'}")
    print(f"  Culture Health:       {result['culture_health_score']:.2f}/5.0  ", end="")
    print(f"{'[GOOD]' if result['culture_health_score'] >= 4.0 else '[ATTENTION]' if result['culture_health_score'] >= 3.5 else '[CONCERN]'}")
    
    print("\n\nFRAMEWORK SCORES")
    print("-" * 50)
    print(f"  JD-R Balance:         {result['jdr_balance_score']:+.2f}  (Burnout risk: {result['burnout_risk']})")
    print(f"  Psychological Safety: {result['psych_safety_score']:.2f}/5.0  (Voice: {result['voice_climate']})")
    print(f"  SDT Motivation:       {result['intrinsic_motivation_index']:.2f}/5.0")
    print(f"  Organisational Trust: {result['trust_index']:.2f}/5.0")
    print(f"  Burnout Profile:      {result['burnout_profile'].upper()}")
    print(f"  Team Stage:           {result['team_stage'].upper()}")
    
    print("\n\nRISK FACTORS")
    print("-" * 50)
    for risk in result['risk_factors']:
        print(f"  [!] {risk}")
    
    print("\n\nPROTECTIVE FACTORS")
    print("-" * 50)
    for protective in result['protective_factors']:
        print(f"  [+] {protective}")
    
    print("\n\nAI DIAGNOSIS")
    print("-" * 50)
    print(f"  {result['ai_diagnosis']}")
    
    print("\n\nKEY INSIGHTS")
    print("-" * 50)
    for insight in result['ai_insights']:
        print(f"\n  {insight}")
    
    print("\n\nPRIORITY INTERVENTIONS")
    print("-" * 50)
    for i, intervention in enumerate(result['priority_interventions'][:5], 1):
        print(f"\n  {i}. {intervention['intervention']}")
        print(f"     Framework: {intervention['framework']}")
        print(f"     Timeline: {intervention['timeline']}")
        print(f"     Evidence: {intervention['evidence_base'][:60]}...")
    
    print("\n\nQUICK WINS")
    print("-" * 50)
    for win in result['quick_wins']:
        print(f"  - {win}")
    
    print("\n\nEXECUTIVE SUMMARY")
    print("=" * 78)
    print(f"\n{result['executive_summary']}")
    
    print("\n\nACTION PLAN")
    print("-" * 50)
    plan = result['action_plan']
    print("\n  IMMEDIATE (This Week):")
    for action in plan.get('immediate_actions', [])[:3]:
        print(f"    - {action}")
    print("\n  30-DAY PRIORITIES:")
    for action in plan.get('30_day_priorities', []):
        print(f"    - {action}")
    print("\n  90-DAY INITIATIVES:")
    for action in plan.get('90_day_initiatives', []):
        print(f"    - {action}")
    
    print("\n\n  SUCCESS METRICS:")
    for metric in plan.get('success_metrics', []):
        print(f"    - {metric['metric']}: {metric['current']:.2f} -> {metric['target']:.2f}")
    
    print(f"\n  REVIEW CADENCE: {plan.get('review_cadence', 'TBD')}")
    
    print("\n" + "=" * 78)
    print(" Assessment Complete ".center(78))
    print("=" * 78 + "\n")
    
    return result


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run the organisational psychology engine."""
    print("\n" + "=" * 78)
    print(" ORGANISATIONAL PSYCHOLOGY ENGINE ".center(78))
    print(" Evidence-Based Workplace Assessment ".center(78))
    print("=" * 78)
    print("""
    Integrating Multiple Psychological Frameworks:
    
    - Job Demands-Resources Model (Bakker & Demerouti)
    - Psychological Safety (Edmondson)
    - Self-Determination Theory (Deci & Ryan)
    - Organisational Justice (Colquitt)
    - Maslach Burnout Inventory
    - Team Dynamics (Tuckman + Lencioni)
    """)
    
    run_full_assessment()


if __name__ == "__main__":
    main()
