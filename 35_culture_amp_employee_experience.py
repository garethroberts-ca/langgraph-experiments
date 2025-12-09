#!/usr/bin/env python3
"""
LangGraph Advanced Example 35: Culture Amp Employee Experience Platform
========================================================================

This example demonstrates LangGraph patterns tailored for employee experience
workflows, inspired by Culture Amp's platform capabilities:

- Engagement survey analysis and sentiment detection
- AI-powered comment summarisation
- Retention risk identification
- DEI (Diversity, Equity & Inclusion) insights
- Action plan generation and prioritisation
- Manager coaching recommendations
- Pulse survey orchestration

These patterns showcase how LangGraph can power intelligent HR analytics
and employee feedback systems at scale.

Author: LangGraph Examples
"""

import operator
from datetime import datetime
from typing import Annotated, Any

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
# Demo 1: Engagement Survey Analysis Pipeline
# =============================================================================

def demo_engagement_survey_analysis():
    """
    Analyse engagement survey responses with AI-powered insights.
    
    Demonstrates:
    - Multi-stage survey processing
    - Sentiment analysis on open-ended responses
    - Benchmark comparison
    - Insight generation
    """
    print_banner("Demo 1: Engagement Survey Analysis Pipeline")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class SurveyAnalysisState(TypedDict):
        survey_responses: list[dict]
        quantitative_scores: dict
        sentiment_analysis: dict
        benchmark_comparison: dict
        key_insights: list[str]
        executive_summary: str
    
    def calculate_quantitative_scores(state: SurveyAnalysisState) -> dict:
        """Calculate aggregate scores from Likert-scale questions."""
        log("Calculating quantitative engagement scores...")
        
        responses = state["survey_responses"]
        
        # Aggregate by category
        categories = {
            "engagement": [],
            "leadership": [],
            "growth": [],
            "wellbeing": [],
            "belonging": []
        }
        
        for response in responses:
            for category, score in response.get("scores", {}).items():
                if category in categories:
                    categories[category].append(score)
        
        # Calculate averages
        scores = {}
        for category, values in categories.items():
            if values:
                avg = sum(values) / len(values)
                scores[category] = {
                    "average": round(avg, 2),
                    "response_count": len(values),
                    "favourable_pct": round(sum(1 for v in values if v >= 4) / len(values) * 100, 1)
                }
        
        log(f"Processed {len(responses)} responses across {len(scores)} categories", indent=1)
        return {"quantitative_scores": scores}
    
    def analyse_open_ended_responses(state: SurveyAnalysisState) -> dict:
        """Analyse sentiment and themes in open-ended comments."""
        log("Analysing open-ended responses...")
        
        comments = [
            r.get("comment", "") 
            for r in state["survey_responses"] 
            if r.get("comment")
        ]
        
        if not comments:
            return {"sentiment_analysis": {"themes": [], "overall_sentiment": "neutral"}}
        
        # Use LLM for theme extraction
        combined_comments = "\n".join(f"- {c}" for c in comments[:10])  # Limit for demo
        
        response = llm.invoke(
            f"""Analyse these employee survey comments and provide:
1. Top 3 positive themes
2. Top 3 areas of concern
3. Overall sentiment (positive/neutral/negative)

Comments:
{combined_comments}

Respond in a structured format."""
        )
        
        log(f"Analysed {len(comments)} comments", indent=1)
        return {
            "sentiment_analysis": {
                "comment_count": len(comments),
                "analysis": response.content,
                "themes_extracted": True
            }
        }
    
    def compare_to_benchmarks(state: SurveyAnalysisState) -> dict:
        """Compare scores against industry benchmarks."""
        log("Comparing against industry benchmarks...")
        
        # Simulated benchmark data (would come from Culture Amp's benchmark database)
        benchmarks = {
            "engagement": {"industry_avg": 3.8, "top_quartile": 4.2},
            "leadership": {"industry_avg": 3.6, "top_quartile": 4.0},
            "growth": {"industry_avg": 3.5, "top_quartile": 3.9},
            "wellbeing": {"industry_avg": 3.7, "top_quartile": 4.1},
            "belonging": {"industry_avg": 3.9, "top_quartile": 4.3}
        }
        
        comparison = {}
        for category, scores in state["quantitative_scores"].items():
            if category in benchmarks:
                benchmark = benchmarks[category]
                avg = scores["average"]
                comparison[category] = {
                    "score": avg,
                    "vs_industry": round(avg - benchmark["industry_avg"], 2),
                    "vs_top_quartile": round(avg - benchmark["top_quartile"], 2),
                    "position": "above" if avg > benchmark["industry_avg"] else "below"
                }
        
        log(f"Compared {len(comparison)} categories against benchmarks", indent=1)
        return {"benchmark_comparison": comparison}
    
    def generate_insights(state: SurveyAnalysisState) -> dict:
        """Generate actionable insights from the analysis."""
        log("Generating key insights...")
        
        insights = []
        
        # Analyse benchmark comparison
        for category, data in state["benchmark_comparison"].items():
            if data["vs_industry"] > 0.3:
                insights.append(f"Strength: {category.title()} scores significantly above industry average (+{data['vs_industry']})")
            elif data["vs_industry"] < -0.3:
                insights.append(f"Focus area: {category.title()} scores below industry average ({data['vs_industry']})")
        
        # Add sentiment insights
        sentiment = state.get("sentiment_analysis", {})
        if sentiment.get("themes_extracted"):
            insights.append("Qualitative analysis reveals distinct themes requiring attention")
        
        log(f"Generated {len(insights)} insights", indent=1)
        return {"key_insights": insights}
    
    def create_executive_summary(state: SurveyAnalysisState) -> dict:
        """Create an executive summary of the survey results."""
        log("Creating executive summary...")
        
        scores = state["quantitative_scores"]
        insights = state["key_insights"]
        
        # Calculate overall engagement
        if "engagement" in scores:
            eng_score = scores["engagement"]["average"]
            eng_favourable = scores["engagement"]["favourable_pct"]
        else:
            eng_score = 0
            eng_favourable = 0
        
        summary = f"""ENGAGEMENT SURVEY EXECUTIVE SUMMARY
=====================================
Overall Engagement Score: {eng_score}/5.0 ({eng_favourable}% favourable)
Response Count: {sum(s['response_count'] for s in scores.values())}

KEY INSIGHTS:
{chr(10).join(f'- {i}' for i in insights)}

RECOMMENDED FOCUS AREAS:
"""
        
        # Add focus areas based on lowest scores
        sorted_categories = sorted(
            scores.items(),
            key=lambda x: x[1]["average"]
        )
        
        for category, data in sorted_categories[:2]:
            summary += f"- {category.title()}: Currently at {data['average']}/5.0\n"
        
        return {"executive_summary": summary}
    
    # Build graph
    builder = StateGraph(SurveyAnalysisState)
    builder.add_node("quantitative", calculate_quantitative_scores)
    builder.add_node("qualitative", analyse_open_ended_responses)
    builder.add_node("benchmark", compare_to_benchmarks)
    builder.add_node("insights", generate_insights)
    builder.add_node("summary", create_executive_summary)
    
    # Parallel quantitative and qualitative analysis
    builder.add_edge(START, "quantitative")
    builder.add_edge(START, "qualitative")
    builder.add_edge("quantitative", "benchmark")
    builder.add_edge("benchmark", "insights")
    builder.add_edge("qualitative", "insights")
    builder.add_edge("insights", "summary")
    builder.add_edge("summary", END)
    
    graph = builder.compile()
    
    # Sample survey data
    sample_responses = [
        {"scores": {"engagement": 4, "leadership": 3, "growth": 4, "wellbeing": 5, "belonging": 4},
         "comment": "Great team culture but would like more career development opportunities."},
        {"scores": {"engagement": 5, "leadership": 4, "growth": 3, "wellbeing": 4, "belonging": 5},
         "comment": "Leadership has improved significantly this year."},
        {"scores": {"engagement": 3, "leadership": 2, "growth": 2, "wellbeing": 3, "belonging": 3},
         "comment": "Feeling disconnected from company direction. Need more transparency."},
        {"scores": {"engagement": 4, "leadership": 4, "growth": 4, "wellbeing": 4, "belonging": 4},
         "comment": "Solid workplace overall. Flexible working arrangements are appreciated."},
        {"scores": {"engagement": 2, "leadership": 3, "growth": 2, "wellbeing": 2, "belonging": 3},
         "comment": "Workload is unsustainable. Burnout is a real concern in our team."},
    ]
    
    result = graph.invoke({
        "survey_responses": sample_responses,
        "quantitative_scores": {},
        "sentiment_analysis": {},
        "benchmark_comparison": {},
        "key_insights": [],
        "executive_summary": ""
    })
    
    log("\n" + result["executive_summary"])


# =============================================================================
# Demo 2: Retention Risk Analysis
# =============================================================================

def demo_retention_risk_analysis():
    """
    Identify employees at risk of leaving using multiple signals.
    
    Demonstrates:
    - Multi-factor risk assessment
    - Weighted scoring algorithms
    - Intervention recommendations
    """
    print_banner("Demo 2: Retention Risk Analysis")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class RetentionState(TypedDict):
        employee_data: list[dict]
        risk_scores: list[dict]
        high_risk_employees: list[dict]
        intervention_plans: list[dict]
    
    def calculate_risk_scores(state: RetentionState) -> dict:
        """Calculate retention risk scores for each employee."""
        log("Calculating retention risk scores...")
        
        risk_scores = []
        
        for employee in state["employee_data"]:
            # Risk factors and weights
            factors = {
                "engagement_score": (employee.get("engagement_score", 3), -0.3),  # Lower = higher risk
                "tenure_months": (employee.get("tenure_months", 24), -0.1),  # Very short or very long tenure
                "recent_promotion": (1 if employee.get("recent_promotion") else 0, -0.2),  # No promotion = risk
                "manager_rating": (employee.get("manager_rating", 3), -0.2),  # Poor manager = risk
                "salary_vs_market": (employee.get("salary_vs_market", 1.0), -0.15),  # Below market = risk
                "workload_score": (employee.get("workload_score", 3), 0.15),  # High workload = risk
            }
            
            # Calculate weighted risk score (0-100)
            base_risk = 50
            for factor, (value, weight) in factors.items():
                if factor == "engagement_score":
                    # Invert: low engagement = high risk
                    base_risk += (5 - value) * 10 * abs(weight)
                elif factor == "tenure_months":
                    # U-shaped: very new or 18-24 months = higher risk
                    if value < 6 or (18 <= value <= 30):
                        base_risk += 15
                elif factor == "recent_promotion":
                    if value == 0:
                        base_risk += 10
                elif factor == "manager_rating":
                    base_risk += (5 - value) * 8 * abs(weight)
                elif factor == "salary_vs_market":
                    if value < 0.95:
                        base_risk += (1 - value) * 50
                elif factor == "workload_score":
                    if value > 4:
                        base_risk += 15
            
            risk_scores.append({
                "employee_id": employee["employee_id"],
                "name": employee["name"],
                "department": employee.get("department", "Unknown"),
                "risk_score": min(100, max(0, round(base_risk))),
                "risk_level": "High" if base_risk >= 70 else ("Medium" if base_risk >= 50 else "Low")
            })
        
        log(f"Calculated risk scores for {len(risk_scores)} employees", indent=1)
        return {"risk_scores": risk_scores}
    
    def identify_high_risk(state: RetentionState) -> dict:
        """Identify high-risk employees for intervention."""
        log("Identifying high-risk employees...")
        
        high_risk = [
            emp for emp in state["risk_scores"]
            if emp["risk_level"] == "High"
        ]
        
        # Sort by risk score descending
        high_risk.sort(key=lambda x: x["risk_score"], reverse=True)
        
        log(f"Identified {len(high_risk)} high-risk employees", indent=1)
        return {"high_risk_employees": high_risk}
    
    def generate_intervention_plans(state: RetentionState) -> dict:
        """Generate personalised intervention plans for high-risk employees."""
        log("Generating intervention plans...")
        
        interventions = []
        
        for employee in state["high_risk_employees"][:3]:  # Top 3 for demo
            # Find original employee data
            emp_data = next(
                (e for e in state["employee_data"] if e["employee_id"] == employee["employee_id"]),
                {}
            )
            
            # Determine intervention based on risk factors
            actions = []
            
            if emp_data.get("engagement_score", 3) < 3:
                actions.append("Schedule 1:1 with skip-level manager to discuss concerns")
            
            if not emp_data.get("recent_promotion"):
                actions.append("Review career development plan and discuss growth opportunities")
            
            if emp_data.get("salary_vs_market", 1.0) < 0.95:
                actions.append("Initiate compensation review with People team")
            
            if emp_data.get("workload_score", 3) > 4:
                actions.append("Assess workload distribution and consider resource reallocation")
            
            if emp_data.get("manager_rating", 3) < 3:
                actions.append("Consider team transfer or manager coaching intervention")
            
            if not actions:
                actions.append("Schedule stay interview to understand motivations")
            
            interventions.append({
                "employee_id": employee["employee_id"],
                "name": employee["name"],
                "risk_score": employee["risk_score"],
                "recommended_actions": actions,
                "urgency": "Immediate" if employee["risk_score"] >= 80 else "This quarter"
            })
        
        log(f"Generated {len(interventions)} intervention plans", indent=1)
        return {"intervention_plans": interventions}
    
    # Build graph
    builder = StateGraph(RetentionState)
    builder.add_node("calculate_risk", calculate_risk_scores)
    builder.add_node("identify_high_risk", identify_high_risk)
    builder.add_node("generate_interventions", generate_intervention_plans)
    builder.add_edge(START, "calculate_risk")
    builder.add_edge("calculate_risk", "identify_high_risk")
    builder.add_edge("identify_high_risk", "generate_interventions")
    builder.add_edge("generate_interventions", END)
    
    graph = builder.compile()
    
    # Sample employee data
    employees = [
        {"employee_id": "E001", "name": "Sarah Chen", "department": "Engineering",
         "engagement_score": 2, "tenure_months": 22, "recent_promotion": False,
         "manager_rating": 2, "salary_vs_market": 0.88, "workload_score": 5},
        {"employee_id": "E002", "name": "James Wilson", "department": "Product",
         "engagement_score": 4, "tenure_months": 36, "recent_promotion": True,
         "manager_rating": 4, "salary_vs_market": 1.05, "workload_score": 3},
        {"employee_id": "E003", "name": "Priya Sharma", "department": "Engineering",
         "engagement_score": 3, "tenure_months": 18, "recent_promotion": False,
         "manager_rating": 3, "salary_vs_market": 0.92, "workload_score": 4},
        {"employee_id": "E004", "name": "Michael O'Brien", "department": "Sales",
         "engagement_score": 2, "tenure_months": 8, "recent_promotion": False,
         "manager_rating": 2, "salary_vs_market": 0.90, "workload_score": 5},
    ]
    
    result = graph.invoke({
        "employee_data": employees,
        "risk_scores": [],
        "high_risk_employees": [],
        "intervention_plans": []
    })
    
    log("\nRETENTION RISK SUMMARY")
    log("=" * 40)
    for emp in result["risk_scores"]:
        log(f"{emp['name']}: {emp['risk_score']}% ({emp['risk_level']})", indent=1)
    
    log("\nINTERVENTION PLANS")
    log("=" * 40)
    for plan in result["intervention_plans"]:
        log(f"\n{plan['name']} (Risk: {plan['risk_score']}%) - {plan['urgency']}", indent=1)
        for action in plan["recommended_actions"]:
            log(f"- {action}", indent=2)


# =============================================================================
# Demo 3: AI Comment Summarisation
# =============================================================================

def demo_comment_summarisation():
    """
    Summarise large volumes of employee comments using AI.
    
    Demonstrates:
    - Batch processing of comments
    - Theme extraction
    - Sentiment clustering
    - Executive-ready summaries
    """
    print_banner("Demo 3: AI Comment Summarisation")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class CommentSummaryState(TypedDict):
        raw_comments: list[dict]
        categorised_comments: dict
        theme_summaries: dict
        overall_summary: str
    
    def categorise_comments(state: CommentSummaryState) -> dict:
        """Categorise comments by topic area."""
        log("Categorising comments by topic...")
        
        categories = {
            "leadership": [],
            "growth_development": [],
            "wellbeing": [],
            "collaboration": [],
            "processes_tools": [],
            "recognition": [],
            "other": []
        }
        
        # Keywords for categorisation (simplified for demo)
        keyword_map = {
            "leadership": ["manager", "leadership", "direction", "vision", "executive", "lead"],
            "growth_development": ["career", "growth", "learning", "development", "promotion", "skills", "training"],
            "wellbeing": ["workload", "stress", "burnout", "balance", "health", "flexible", "remote"],
            "collaboration": ["team", "collaborate", "communication", "silos", "cross-functional"],
            "processes_tools": ["process", "tool", "system", "efficient", "bureaucracy", "approval"],
            "recognition": ["recognition", "appreciate", "valued", "feedback", "reward"]
        }
        
        for comment_data in state["raw_comments"]:
            comment = comment_data.get("comment", "").lower()
            categorised = False
            
            for category, keywords in keyword_map.items():
                if any(kw in comment for kw in keywords):
                    categories[category].append(comment_data)
                    categorised = True
                    break
            
            if not categorised:
                categories["other"].append(comment_data)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        log(f"Categorised {len(state['raw_comments'])} comments into {len(categories)} themes", indent=1)
        return {"categorised_comments": categories}
    
    def summarise_themes(state: CommentSummaryState) -> dict:
        """Generate AI summaries for each theme."""
        log("Generating theme summaries...")
        
        summaries = {}
        
        for theme, comments in state["categorised_comments"].items():
            if not comments:
                continue
            
            comment_text = "\n".join(f"- {c.get('comment', '')}" for c in comments[:15])
            
            response = llm.invoke(
                f"""Summarise these employee feedback comments about {theme.replace('_', ' ')} 
in 2-3 sentences. Highlight the key sentiment and any specific concerns or praise.

Comments:
{comment_text}

Summary:"""
            )
            
            # Determine sentiment
            positive_words = ["great", "love", "excellent", "appreciate", "happy", "good"]
            negative_words = ["concern", "issue", "problem", "frustrated", "difficult", "poor"]
            
            text_lower = comment_text.lower()
            positive_count = sum(1 for w in positive_words if w in text_lower)
            negative_count = sum(1 for w in negative_words if w in text_lower)
            
            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"
            else:
                sentiment = "Mixed"
            
            summaries[theme] = {
                "comment_count": len(comments),
                "sentiment": sentiment,
                "summary": response.content
            }
        
        log(f"Generated summaries for {len(summaries)} themes", indent=1)
        return {"theme_summaries": summaries}
    
    def create_overall_summary(state: CommentSummaryState) -> dict:
        """Create an executive summary of all comments."""
        log("Creating overall summary...")
        
        total_comments = len(state["raw_comments"])
        themes = state["theme_summaries"]
        
        # Build summary
        summary_parts = [
            f"EMPLOYEE FEEDBACK SUMMARY",
            f"=" * 40,
            f"Total Comments Analysed: {total_comments}",
            f"Themes Identified: {len(themes)}",
            "",
            "THEME BREAKDOWN:"
        ]
        
        for theme, data in themes.items():
            theme_title = theme.replace("_", " ").title()
            summary_parts.append(f"\n{theme_title} ({data['comment_count']} comments, {data['sentiment']})")
            summary_parts.append(data["summary"])
        
        return {"overall_summary": "\n".join(summary_parts)}
    
    # Build graph
    builder = StateGraph(CommentSummaryState)
    builder.add_node("categorise", categorise_comments)
    builder.add_node("summarise_themes", summarise_themes)
    builder.add_node("overall_summary", create_overall_summary)
    builder.add_edge(START, "categorise")
    builder.add_edge("categorise", "summarise_themes")
    builder.add_edge("summarise_themes", "overall_summary")
    builder.add_edge("overall_summary", END)
    
    graph = builder.compile()
    
    # Sample comments
    comments = [
        {"comment": "My manager provides excellent support and regular feedback.", "department": "Engineering"},
        {"comment": "Career growth opportunities feel limited. Would like clearer pathways.", "department": "Product"},
        {"comment": "Workload has been unsustainable for months. Team is burnt out.", "department": "Engineering"},
        {"comment": "Cross-team collaboration has improved significantly this quarter.", "department": "Design"},
        {"comment": "Leadership communication about company direction could be better.", "department": "Sales"},
        {"comment": "Love the flexible working arrangements. Great work-life balance.", "department": "Marketing"},
        {"comment": "Too many approval processes slow down our ability to deliver.", "department": "Engineering"},
        {"comment": "Feel valued and recognised for my contributions.", "department": "Customer Success"},
        {"comment": "Manager rarely provides constructive feedback or career guidance.", "department": "Operations"},
        {"comment": "Training and development budget has been very helpful.", "department": "Engineering"},
    ]
    
    result = graph.invoke({
        "raw_comments": comments,
        "categorised_comments": {},
        "theme_summaries": {},
        "overall_summary": ""
    })
    
    log("\n" + result["overall_summary"])


# =============================================================================
# Demo 4: DEI Insights Analysis
# =============================================================================

def demo_dei_insights():
    """
    Analyse diversity, equity and inclusion survey data.
    
    Demonstrates:
    - Demographic segmentation
    - Equity gap identification
    - Inclusive action recommendations
    """
    print_banner("Demo 4: DEI Insights Analysis")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class DEIState(TypedDict):
        survey_data: list[dict]
        demographic_breakdowns: dict
        equity_gaps: list[dict]
        inclusion_score: float
        recommendations: list[str]
    
    def calculate_demographic_breakdowns(state: DEIState) -> dict:
        """Calculate scores broken down by demographic groups."""
        log("Calculating demographic breakdowns...")
        
        breakdowns = {}
        
        # Group by demographics
        demographics = ["gender", "ethnicity", "tenure_band", "level"]
        
        for demo in demographics:
            groups = {}
            for response in state["survey_data"]:
                group = response.get(demo, "Not specified")
                if group not in groups:
                    groups[group] = {"belonging": [], "equity": [], "inclusion": []}
                
                groups[group]["belonging"].append(response.get("belonging_score", 3))
                groups[group]["equity"].append(response.get("equity_score", 3))
                groups[group]["inclusion"].append(response.get("inclusion_score", 3))
            
            # Calculate averages
            breakdowns[demo] = {}
            for group, scores in groups.items():
                breakdowns[demo][group] = {
                    "belonging": round(sum(scores["belonging"]) / len(scores["belonging"]), 2),
                    "equity": round(sum(scores["equity"]) / len(scores["equity"]), 2),
                    "inclusion": round(sum(scores["inclusion"]) / len(scores["inclusion"]), 2),
                    "response_count": len(scores["belonging"])
                }
        
        log(f"Calculated breakdowns across {len(demographics)} demographic dimensions", indent=1)
        return {"demographic_breakdowns": breakdowns}
    
    def identify_equity_gaps(state: DEIState) -> dict:
        """Identify significant gaps between demographic groups."""
        log("Identifying equity gaps...")
        
        gaps = []
        threshold = 0.5  # Significant gap threshold
        
        for dimension, groups in state["demographic_breakdowns"].items():
            if len(groups) < 2:
                continue
            
            # Calculate overall average for comparison
            all_scores = []
            for group_data in groups.values():
                all_scores.append(group_data["equity"])
            overall_avg = sum(all_scores) / len(all_scores)
            
            # Find groups below threshold
            for group, data in groups.items():
                diff = data["equity"] - overall_avg
                if diff < -threshold:
                    gaps.append({
                        "dimension": dimension,
                        "group": group,
                        "score": data["equity"],
                        "gap": round(diff, 2),
                        "sample_size": data["response_count"]
                    })
        
        # Sort by gap size
        gaps.sort(key=lambda x: x["gap"])
        
        log(f"Identified {len(gaps)} significant equity gaps", indent=1)
        return {"equity_gaps": gaps}
    
    def calculate_inclusion_score(state: DEIState) -> dict:
        """Calculate overall inclusion score."""
        log("Calculating overall inclusion score...")
        
        all_inclusion = []
        for response in state["survey_data"]:
            all_inclusion.append(response.get("inclusion_score", 3))
        
        avg_score = sum(all_inclusion) / len(all_inclusion) if all_inclusion else 0
        
        log(f"Overall inclusion score: {avg_score:.2f}/5.0", indent=1)
        return {"inclusion_score": round(avg_score, 2)}
    
    def generate_dei_recommendations(state: DEIState) -> dict:
        """Generate recommendations based on DEI analysis."""
        log("Generating DEI recommendations...")
        
        recommendations = []
        
        # Address equity gaps
        for gap in state["equity_gaps"][:3]:
            recommendations.append(
                f"Address equity gap for {gap['group']} ({gap['dimension']}): "
                f"Score {gap['score']}/5.0, {abs(gap['gap'])} below average. "
                f"Consider targeted focus groups and support programmes."
            )
        
        # Overall inclusion recommendations
        if state["inclusion_score"] < 3.5:
            recommendations.append(
                "Prioritise inclusion training for all people managers"
            )
            recommendations.append(
                "Review hiring and promotion processes for potential bias"
            )
        
        if state["inclusion_score"] >= 4.0:
            recommendations.append(
                "Share inclusion best practices as a case study across teams"
            )
        
        # Check belonging scores by tenure
        tenure_data = state["demographic_breakdowns"].get("tenure_band", {})
        new_hires = tenure_data.get("0-1 years", {})
        if new_hires and new_hires.get("belonging", 5) < 3.5:
            recommendations.append(
                "Enhance onboarding programme to improve belonging for new hires"
            )
        
        log(f"Generated {len(recommendations)} recommendations", indent=1)
        return {"recommendations": recommendations}
    
    # Build graph
    builder = StateGraph(DEIState)
    builder.add_node("breakdowns", calculate_demographic_breakdowns)
    builder.add_node("gaps", identify_equity_gaps)
    builder.add_node("inclusion", calculate_inclusion_score)
    builder.add_node("recommendations", generate_dei_recommendations)
    
    builder.add_edge(START, "breakdowns")
    builder.add_edge(START, "inclusion")
    builder.add_edge("breakdowns", "gaps")
    builder.add_edge("gaps", "recommendations")
    builder.add_edge("inclusion", "recommendations")
    builder.add_edge("recommendations", END)
    
    graph = builder.compile()
    
    # Sample DEI survey data
    survey_data = [
        {"gender": "Female", "ethnicity": "Asian", "tenure_band": "2-5 years", "level": "IC",
         "belonging_score": 4, "equity_score": 3, "inclusion_score": 4},
        {"gender": "Male", "ethnicity": "White", "tenure_band": "5+ years", "level": "Manager",
         "belonging_score": 5, "equity_score": 5, "inclusion_score": 5},
        {"gender": "Female", "ethnicity": "Black", "tenure_band": "0-1 years", "level": "IC",
         "belonging_score": 3, "equity_score": 2, "inclusion_score": 3},
        {"gender": "Non-binary", "ethnicity": "Hispanic", "tenure_band": "2-5 years", "level": "IC",
         "belonging_score": 3, "equity_score": 3, "inclusion_score": 3},
        {"gender": "Male", "ethnicity": "Asian", "tenure_band": "0-1 years", "level": "IC",
         "belonging_score": 3, "equity_score": 4, "inclusion_score": 4},
        {"gender": "Female", "ethnicity": "White", "tenure_band": "5+ years", "level": "Director",
         "belonging_score": 5, "equity_score": 4, "inclusion_score": 5},
    ]
    
    result = graph.invoke({
        "survey_data": survey_data,
        "demographic_breakdowns": {},
        "equity_gaps": [],
        "inclusion_score": 0.0,
        "recommendations": []
    })
    
    log("\nDEI ANALYSIS RESULTS")
    log("=" * 40)
    log(f"Overall Inclusion Score: {result['inclusion_score']}/5.0")
    
    log("\nEquity Gaps Identified:")
    for gap in result["equity_gaps"]:
        log(f"- {gap['group']} ({gap['dimension']}): {gap['gap']} below average", indent=1)
    
    log("\nRecommendations:")
    for rec in result["recommendations"]:
        log(f"- {rec}", indent=1)


# =============================================================================
# Demo 5: Manager Effectiveness Coaching
# =============================================================================

def demo_manager_coaching():
    """
    Generate personalised coaching recommendations for managers.
    
    Demonstrates:
    - 360-degree feedback analysis
    - Strength and development area identification
    - Personalised coaching plans
    """
    print_banner("Demo 5: Manager Effectiveness Coaching")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    class ManagerCoachingState(TypedDict):
        manager_data: dict
        team_feedback: list[dict]
        strengths: list[str]
        development_areas: list[str]
        coaching_plan: str
    
    def analyse_team_feedback(state: ManagerCoachingState) -> dict:
        """Analyse feedback from direct reports."""
        log("Analysing team feedback...")
        
        feedback = state["team_feedback"]
        
        # Aggregate scores
        dimensions = [
            "communication", "support", "recognition", 
            "development", "clarity", "trust"
        ]
        
        scores = {d: [] for d in dimensions}
        for fb in feedback:
            for dim in dimensions:
                if dim in fb:
                    scores[dim].append(fb[dim])
        
        # Calculate averages and identify strengths/development areas
        strengths = []
        development_areas = []
        
        for dim, values in scores.items():
            if not values:
                continue
            avg = sum(values) / len(values)
            
            if avg >= 4.0:
                strengths.append(f"{dim.title()}: {avg:.1f}/5.0")
            elif avg < 3.5:
                development_areas.append(f"{dim.title()}: {avg:.1f}/5.0")
        
        log(f"Identified {len(strengths)} strengths, {len(development_areas)} development areas", indent=1)
        return {
            "strengths": strengths,
            "development_areas": development_areas
        }
    
    def generate_coaching_plan(state: ManagerCoachingState) -> dict:
        """Generate a personalised coaching plan."""
        log("Generating coaching plan...")
        
        manager = state["manager_data"]
        strengths = state["strengths"]
        dev_areas = state["development_areas"]
        
        # Prepare context for LLM
        context = f"""
Manager: {manager.get('name', 'Unknown')}
Team Size: {manager.get('team_size', 'Unknown')}
Tenure as Manager: {manager.get('manager_tenure', 'Unknown')}

Strengths (rated 4.0+/5.0 by team):
{chr(10).join(f'- {s}' for s in strengths) if strengths else '- None identified'}

Development Areas (rated below 3.5/5.0 by team):
{chr(10).join(f'- {d}' for d in dev_areas) if dev_areas else '- None identified'}
"""
        
        response = llm.invoke(
            f"""Based on this manager's 360 feedback, create a brief coaching plan with:
1. One key strength to leverage
2. One priority development area with specific actions
3. A suggested coaching focus for the next quarter

{context}

Coaching Plan:"""
        )
        
        plan = f"""MANAGER COACHING PLAN
=====================
Manager: {manager.get('name', 'Unknown')}
Generated: {datetime.now().strftime('%Y-%m-%d')}

FEEDBACK SUMMARY
----------------
Strengths: {', '.join(strengths) if strengths else 'None identified'}
Development Areas: {', '.join(dev_areas) if dev_areas else 'None identified'}

COACHING RECOMMENDATIONS
------------------------
{response.content}
"""
        
        return {"coaching_plan": plan}
    
    # Build graph
    builder = StateGraph(ManagerCoachingState)
    builder.add_node("analyse", analyse_team_feedback)
    builder.add_node("coaching", generate_coaching_plan)
    builder.add_edge(START, "analyse")
    builder.add_edge("analyse", "coaching")
    builder.add_edge("coaching", END)
    
    graph = builder.compile()
    
    # Sample manager data
    manager_data = {
        "name": "Alex Thompson",
        "team_size": 6,
        "manager_tenure": "18 months",
        "department": "Engineering"
    }
    
    team_feedback = [
        {"communication": 3, "support": 4, "recognition": 2, "development": 3, "clarity": 3, "trust": 4},
        {"communication": 3, "support": 5, "recognition": 3, "development": 2, "clarity": 4, "trust": 5},
        {"communication": 4, "support": 4, "recognition": 2, "development": 3, "clarity": 3, "trust": 4},
        {"communication": 3, "support": 5, "recognition": 3, "development": 4, "clarity": 3, "trust": 5},
    ]
    
    result = graph.invoke({
        "manager_data": manager_data,
        "team_feedback": team_feedback,
        "strengths": [],
        "development_areas": [],
        "coaching_plan": ""
    })
    
    log("\n" + result["coaching_plan"])


# =============================================================================
# Demo 6: Action Plan Generator
# =============================================================================

def demo_action_plan_generator():
    """
    Generate prioritised action plans from survey insights.
    
    Demonstrates:
    - Impact vs effort prioritisation
    - SMART goal generation
    - Owner assignment suggestions
    """
    print_banner("Demo 6: Action Plan Generator")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class ActionPlanState(TypedDict):
        survey_insights: list[dict]
        prioritised_actions: list[dict]
        action_plan: str
    
    def prioritise_actions(state: ActionPlanState) -> dict:
        """Prioritise actions based on impact and effort."""
        log("Prioritising actions...")
        
        actions = []
        
        for insight in state["survey_insights"]:
            # Calculate priority score
            impact = insight.get("impact_score", 3)  # 1-5
            effort = insight.get("effort_score", 3)  # 1-5 (lower = easier)
            affected_count = insight.get("affected_employees", 0)
            
            # Priority formula: higher impact, lower effort, more affected = higher priority
            priority_score = (impact * 2) + (6 - effort) + (min(affected_count / 10, 5))
            
            actions.append({
                "issue": insight["issue"],
                "category": insight.get("category", "General"),
                "impact_score": impact,
                "effort_score": effort,
                "affected_employees": affected_count,
                "priority_score": round(priority_score, 1),
                "priority_level": "High" if priority_score >= 12 else ("Medium" if priority_score >= 8 else "Low")
            })
        
        # Sort by priority
        actions.sort(key=lambda x: x["priority_score"], reverse=True)
        
        log(f"Prioritised {len(actions)} potential actions", indent=1)
        return {"prioritised_actions": actions}
    
    def generate_action_plan(state: ActionPlanState) -> dict:
        """Generate detailed action plan for top priorities."""
        log("Generating action plan...")
        
        top_actions = [a for a in state["prioritised_actions"] if a["priority_level"] in ["High", "Medium"]][:5]
        
        plan_sections = [
            "EMPLOYEE EXPERIENCE ACTION PLAN",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
            f"Total Issues Identified: {len(state['survey_insights'])}",
            f"Priority Actions: {len(top_actions)}",
            "",
            "PRIORITY ACTIONS",
            "-" * 30
        ]
        
        for i, action in enumerate(top_actions, 1):
            # Generate SMART goal using LLM
            response = llm.invoke(
                f"""Create a SMART goal (Specific, Measurable, Achievable, Relevant, Time-bound) 
to address this employee feedback issue in one sentence:

Issue: {action['issue']}
Category: {action['category']}
Employees Affected: {action['affected_employees']}

SMART Goal:"""
            )
            
            # Suggest owner based on category
            owner_map = {
                "Leadership": "Executive Team",
                "Growth": "People & Development Team",
                "Wellbeing": "People Experience Team",
                "Processes": "Operations Team",
                "Recognition": "People Team & Managers"
            }
            suggested_owner = owner_map.get(action["category"], "People Team")
            
            plan_sections.extend([
                f"\n{i}. {action['issue']}",
                f"   Priority: {action['priority_level']} (Score: {action['priority_score']})",
                f"   Impact: {action['impact_score']}/5 | Effort: {action['effort_score']}/5",
                f"   Employees Affected: {action['affected_employees']}",
                f"   Suggested Owner: {suggested_owner}",
                f"   SMART Goal: {response.content.strip()}"
            ])
        
        plan_sections.extend([
            "",
            "NEXT STEPS",
            "-" * 30,
            "1. Review action plan with leadership team",
            "2. Assign owners and agree timelines",
            "3. Communicate plan to employees",
            "4. Schedule quarterly progress reviews"
        ])
        
        return {"action_plan": "\n".join(plan_sections)}
    
    # Build graph
    builder = StateGraph(ActionPlanState)
    builder.add_node("prioritise", prioritise_actions)
    builder.add_node("generate", generate_action_plan)
    builder.add_edge(START, "prioritise")
    builder.add_edge("prioritise", "generate")
    builder.add_edge("generate", END)
    
    graph = builder.compile()
    
    # Sample insights
    insights = [
        {"issue": "Limited career growth visibility", "category": "Growth", 
         "impact_score": 5, "effort_score": 3, "affected_employees": 45},
        {"issue": "Inconsistent manager feedback", "category": "Leadership",
         "impact_score": 4, "effort_score": 2, "affected_employees": 32},
        {"issue": "Unsustainable workload in Engineering", "category": "Wellbeing",
         "impact_score": 5, "effort_score": 4, "affected_employees": 18},
        {"issue": "Lack of recognition programmes", "category": "Recognition",
         "impact_score": 3, "effort_score": 2, "affected_employees": 60},
        {"issue": "Slow approval processes", "category": "Processes",
         "impact_score": 3, "effort_score": 4, "affected_employees": 25},
    ]
    
    result = graph.invoke({
        "survey_insights": insights,
        "prioritised_actions": [],
        "action_plan": ""
    })
    
    log("\n" + result["action_plan"])


# =============================================================================
# Demo 7: Pulse Survey Orchestration
# =============================================================================

def demo_pulse_survey_orchestration():
    """
    Orchestrate rapid pulse surveys with real-time analysis.
    
    Demonstrates:
    - Survey configuration
    - Response collection simulation
    - Real-time trend analysis
    - Alert generation
    """
    print_banner("Demo 7: Pulse Survey Orchestration")
    
    class PulseState(TypedDict):
        survey_config: dict
        responses: list[dict]
        current_results: dict
        trend_analysis: dict
        alerts: list[str]
    
    def configure_survey(state: PulseState) -> dict:
        """Configure the pulse survey parameters."""
        log("Configuring pulse survey...")
        
        config = state["survey_config"]
        log(f"Survey: {config.get('name', 'Unnamed')}", indent=1)
        log(f"Questions: {len(config.get('questions', []))}", indent=1)
        log(f"Target audience: {config.get('audience', 'All employees')}", indent=1)
        
        return {}
    
    def collect_responses(state: PulseState) -> dict:
        """Simulate response collection."""
        log("Collecting responses...")
        
        # In reality, this would integrate with survey distribution
        # Here we simulate incoming responses
        import random
        random.seed(42)
        
        simulated_responses = []
        for i in range(25):
            response = {
                "respondent_id": f"R{i+1:03d}",
                "timestamp": datetime.now().isoformat(),
                "answers": {}
            }
            
            for q in state["survey_config"].get("questions", []):
                if q["type"] == "rating":
                    response["answers"][q["id"]] = random.randint(2, 5)
                elif q["type"] == "text":
                    response["answers"][q["id"]] = f"Sample response {i+1}"
            
            simulated_responses.append(response)
        
        log(f"Collected {len(simulated_responses)} responses", indent=1)
        return {"responses": simulated_responses}
    
    def calculate_results(state: PulseState) -> dict:
        """Calculate real-time survey results."""
        log("Calculating results...")
        
        results = {
            "response_count": len(state["responses"]),
            "response_rate": len(state["responses"]) / 50 * 100,  # Assume 50 target
            "question_results": {}
        }
        
        for q in state["survey_config"].get("questions", []):
            if q["type"] == "rating":
                scores = [
                    r["answers"].get(q["id"], 0) 
                    for r in state["responses"]
                    if q["id"] in r.get("answers", {})
                ]
                
                if scores:
                    results["question_results"][q["id"]] = {
                        "question": q["text"],
                        "average": round(sum(scores) / len(scores), 2),
                        "favourable_pct": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
                        "response_count": len(scores)
                    }
        
        log(f"Response rate: {results['response_rate']:.1f}%", indent=1)
        return {"current_results": results}
    
    def analyse_trends(state: PulseState) -> dict:
        """Compare against previous pulse results."""
        log("Analysing trends...")
        
        # Simulated historical data
        previous_pulse = {
            "q1": {"average": 3.8, "favourable_pct": 65},
            "q2": {"average": 4.0, "favourable_pct": 72},
            "q3": {"average": 3.5, "favourable_pct": 55}
        }
        
        trends = {}
        for q_id, current in state["current_results"].get("question_results", {}).items():
            prev = previous_pulse.get(q_id, {})
            if prev:
                change = current["average"] - prev.get("average", current["average"])
                trends[q_id] = {
                    "current": current["average"],
                    "previous": prev.get("average"),
                    "change": round(change, 2),
                    "direction": "up" if change > 0 else ("down" if change < 0 else "stable")
                }
        
        log(f"Compared {len(trends)} questions against previous pulse", indent=1)
        return {"trend_analysis": trends}
    
    def generate_alerts(state: PulseState) -> dict:
        """Generate alerts for significant changes or low scores."""
        log("Checking for alerts...")
        
        alerts = []
        
        # Check for low scores
        for q_id, result in state["current_results"].get("question_results", {}).items():
            if result["average"] < 3.0:
                alerts.append(f"ALERT: '{result['question']}' score critically low at {result['average']}/5.0")
            elif result["favourable_pct"] < 50:
                alerts.append(f"WARNING: '{result['question']}' has only {result['favourable_pct']}% favourable responses")
        
        # Check for significant negative trends
        for q_id, trend in state["trend_analysis"].items():
            if trend["change"] < -0.5:
                alerts.append(f"TREND ALERT: Question {q_id} dropped by {abs(trend['change'])} points since last pulse")
        
        # Check response rate
        if state["current_results"]["response_rate"] < 50:
            alerts.append(f"PARTICIPATION: Response rate at {state['current_results']['response_rate']:.1f}% - consider extending survey")
        
        log(f"Generated {len(alerts)} alerts", indent=1)
        return {"alerts": alerts}
    
    # Build graph
    builder = StateGraph(PulseState)
    builder.add_node("configure", configure_survey)
    builder.add_node("collect", collect_responses)
    builder.add_node("calculate", calculate_results)
    builder.add_node("trends", analyse_trends)
    builder.add_node("alerts", generate_alerts)
    
    builder.add_edge(START, "configure")
    builder.add_edge("configure", "collect")
    builder.add_edge("collect", "calculate")
    builder.add_edge("calculate", "trends")
    builder.add_edge("trends", "alerts")
    builder.add_edge("alerts", END)
    
    graph = builder.compile()
    
    # Sample pulse survey config
    config = {
        "name": "Weekly Wellbeing Pulse",
        "audience": "Engineering Department",
        "questions": [
            {"id": "q1", "text": "I feel supported by my team", "type": "rating"},
            {"id": "q2", "text": "My workload is manageable", "type": "rating"},
            {"id": "q3", "text": "I have the resources I need", "type": "rating"},
        ]
    }
    
    result = graph.invoke({
        "survey_config": config,
        "responses": [],
        "current_results": {},
        "trend_analysis": {},
        "alerts": []
    })
    
    log("\nPULSE SURVEY RESULTS")
    log("=" * 40)
    log(f"Survey: {config['name']}")
    log(f"Responses: {result['current_results']['response_count']} ({result['current_results']['response_rate']:.1f}% rate)")
    
    log("\nQuestion Results:")
    for q_id, data in result["current_results"]["question_results"].items():
        trend = result["trend_analysis"].get(q_id, {})
        trend_str = f" ({trend.get('direction', 'n/a')} {abs(trend.get('change', 0))})" if trend else ""
        log(f"- {data['question']}: {data['average']}/5.0{trend_str}", indent=1)
    
    if result["alerts"]:
        log("\nAlerts:")
        for alert in result["alerts"]:
            log(f"- {alert}", indent=1)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all Culture Amp employee experience demonstrations."""
    print("\n" + "=" * 70)
    print(" LangGraph: Culture Amp Employee Experience Platform ".center(70))
    print("=" * 70)
    
    demo_engagement_survey_analysis()
    demo_retention_risk_analysis()
    demo_comment_summarisation()
    demo_dei_insights()
    demo_manager_coaching()
    demo_action_plan_generator()
    demo_pulse_survey_orchestration()
    
    print("\n" + "=" * 70)
    print(" All Employee Experience Demonstrations Complete ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
