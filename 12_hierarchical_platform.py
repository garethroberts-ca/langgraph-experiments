"""
Script 12: Hierarchical Multi-Graph System with LangGraph Platform Patterns
Demonstrates: Graph-as-tool, remote graph invocation, deployment patterns,
              API design, multi-tenant support, graph composition
"""

from typing import Annotated, TypedDict, Literal, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import json
import hashlib
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


# ============================================================
# MULTI-TENANT CONFIGURATION
# ============================================================

@dataclass
class TenantConfig:
    """Configuration for a tenant organization."""
    tenant_id: str
    name: str
    tier: str  # "basic", "professional", "enterprise"
    features: list[str]
    llm_config: dict
    custom_prompts: dict
    rate_limits: dict
    

TENANT_CONFIGS = {
    "acme_corp": TenantConfig(
        tenant_id="acme_corp",
        name="Acme Corporation",
        tier="enterprise",
        features=["advanced_coaching", "custom_frameworks", "api_access", "sso"],
        llm_config={"model": "gpt-4o", "temperature": 0.7},
        custom_prompts={
            "coaching_style": "Direct and action-oriented, aligned with Acme values",
            "frameworks": ["GROW model", "Acme Leadership Compass"]
        },
        rate_limits={"requests_per_minute": 100, "concurrent_sessions": 50}
    ),
    "startup_inc": TenantConfig(
        tenant_id="startup_inc",
        name="Startup Inc",
        tier="basic",
        features=["basic_coaching"],
        llm_config={"model": "gpt-4o-mini", "temperature": 0.8},
        custom_prompts={},
        rate_limits={"requests_per_minute": 20, "concurrent_sessions": 5}
    )
}


def get_tenant_config(tenant_id: str) -> TenantConfig:
    """Retrieve tenant configuration."""
    return TENANT_CONFIGS.get(tenant_id, TENANT_CONFIGS["startup_inc"])


# ============================================================
# GRAPH REGISTRY - For graph-as-tool pattern
# ============================================================

class GraphRegistry:
    """Registry of available graphs that can be invoked as tools."""
    
    _instance = None
    _graphs: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._graphs = {}
        return cls._instance
    
    @classmethod
    def register(cls, name: str, graph: Any, metadata: dict = None):
        """Register a graph with the registry."""
        cls._graphs[name] = {
            "graph": graph,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        }
    
    @classmethod
    def get(cls, name: str) -> Optional[dict]:
        """Get a graph by name."""
        return cls._graphs.get(name)
    
    @classmethod
    def list_graphs(cls) -> list[str]:
        """List all registered graphs."""
        return list(cls._graphs.keys())
    
    @classmethod
    def invoke(cls, name: str, state: dict, config: dict = None) -> dict:
        """Invoke a registered graph."""
        graph_entry = cls.get(name)
        if not graph_entry:
            raise ValueError(f"Graph '{name}' not found in registry")
        
        return graph_entry["graph"].invoke(state, config or {})


registry = GraphRegistry()


# ============================================================
# LEVEL 3: LEAF GRAPHS (Specialized micro-services)
# ============================================================

class LeafGraphState(TypedDict):
    input: str
    context: dict
    output: str
    confidence: float


def build_skill_assessment_graph():
    """Build specialized skill assessment graph."""
    
    def assess_skills(state: LeafGraphState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        prompt = f"""Assess skills based on this input:
Input: {state['input']}
Context: {json.dumps(state.get('context', {}))}

Provide:
1. Current skill level (1-10)
2. Key strengths
3. Development areas
4. Recommended resources

Format as JSON."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        try:
            assessment = json.loads(response.content)
            confidence = 0.85
        except:
            assessment = {"raw": response.content}
            confidence = 0.6
        
        return {
            "output": json.dumps(assessment),
            "confidence": confidence
        }
    
    graph = StateGraph(LeafGraphState)
    graph.add_node("assess", assess_skills)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()


def build_goal_generator_graph():
    """Build specialized goal generation graph."""
    
    def generate_goals(state: LeafGraphState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        prompt = f"""Generate SMART goals based on:
Input: {state['input']}
Context: {json.dumps(state.get('context', {}))}

Create 3-5 goals in JSON format:
[{{"goal": "...", "specific": "...", "measurable": "...", "timeline": "..."}}]"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "output": response.content,
            "confidence": 0.8
        }
    
    graph = StateGraph(LeafGraphState)
    graph.add_node("generate", generate_goals)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


def build_feedback_analyzer_graph():
    """Build specialized feedback analysis graph."""
    
    def analyze_feedback(state: LeafGraphState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        prompt = f"""Analyze this feedback:
{state['input']}

Provide:
1. Sentiment analysis
2. Key themes
3. Actionable insights
4. Suggested responses

Format as JSON."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "output": response.content,
            "confidence": 0.9
        }
    
    graph = StateGraph(LeafGraphState)
    graph.add_node("analyze", analyze_feedback)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)
    
    return graph.compile()


# Register leaf graphs
registry.register("skill_assessment", build_skill_assessment_graph(), {
    "description": "Assess employee skills and competencies",
    "tier_required": "basic"
})
registry.register("goal_generator", build_goal_generator_graph(), {
    "description": "Generate SMART goals",
    "tier_required": "basic"
})
registry.register("feedback_analyzer", build_feedback_analyzer_graph(), {
    "description": "Analyze feedback and extract insights",
    "tier_required": "professional"
})


# ============================================================
# LEVEL 2: DOMAIN GRAPHS (Compose leaf graphs)
# ============================================================

class DomainGraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    user_id: str
    domain: str
    subgraph_results: dict
    final_output: str


def build_career_development_domain():
    """Build career development domain graph that composes leaf graphs."""
    
    def orchestrate_career_analysis(state: DomainGraphState) -> dict:
        """Orchestrate multiple leaf graphs for career analysis."""
        
        user_message = state["messages"][-1].content if state["messages"] else ""
        context = {"tenant_id": state.get("tenant_id"), "domain": "career"}
        
        results = {}
        
        # Invoke skill assessment
        skill_result = registry.invoke(
            "skill_assessment",
            {"input": user_message, "context": context, "output": "", "confidence": 0.0}
        )
        results["skills"] = skill_result
        
        # Invoke goal generator
        goal_result = registry.invoke(
            "goal_generator",
            {"input": user_message, "context": {**context, "skills": skill_result.get("output", "")}, "output": "", "confidence": 0.0}
        )
        results["goals"] = goal_result
        
        return {"subgraph_results": results}
    
    def synthesize_career_guidance(state: DomainGraphState) -> dict:
        """Synthesize results from leaf graphs."""
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        results = state.get("subgraph_results", {})
        
        prompt = f"""Synthesize career development guidance from these analyses:

Skill Assessment:
{results.get('skills', {}).get('output', 'N/A')}

Generated Goals:
{results.get('goals', {}).get('output', 'N/A')}

Create a cohesive career development plan that connects skills to goals."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "final_output": response.content,
            "messages": [AIMessage(content=response.content)]
        }
    
    graph = StateGraph(DomainGraphState)
    graph.add_node("orchestrate", orchestrate_career_analysis)
    graph.add_node("synthesize", synthesize_career_guidance)
    
    graph.add_edge(START, "orchestrate")
    graph.add_edge("orchestrate", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def build_performance_domain():
    """Build performance management domain graph."""
    
    def analyze_performance(state: DomainGraphState) -> dict:
        user_message = state["messages"][-1].content if state["messages"] else ""
        context = {"tenant_id": state.get("tenant_id"), "domain": "performance"}
        
        results = {}
        
        # Invoke feedback analyzer if available for tenant tier
        tenant = get_tenant_config(state.get("tenant_id", "startup_inc"))
        
        if "advanced_coaching" in tenant.features:
            feedback_result = registry.invoke(
                "feedback_analyzer",
                {"input": user_message, "context": context, "output": "", "confidence": 0.0}
            )
            results["feedback_analysis"] = feedback_result
        
        return {"subgraph_results": results}
    
    def generate_performance_guidance(state: DomainGraphState) -> dict:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        results = state.get("subgraph_results", {})
        
        prompt = f"""Generate performance coaching guidance:

Analysis Results:
{json.dumps(results, indent=2)}

Original Query:
{state['messages'][-1].content if state['messages'] else 'N/A'}

Provide actionable performance improvement guidance."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "final_output": response.content,
            "messages": [AIMessage(content=response.content)]
        }
    
    graph = StateGraph(DomainGraphState)
    graph.add_node("analyze", analyze_performance)
    graph.add_node("generate", generate_performance_guidance)
    
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


# Register domain graphs
registry.register("career_domain", build_career_development_domain(), {
    "description": "Career development coaching",
    "tier_required": "basic",
    "composes": ["skill_assessment", "goal_generator"]
})
registry.register("performance_domain", build_performance_domain(), {
    "description": "Performance management coaching",
    "tier_required": "professional",
    "composes": ["feedback_analyzer"]
})


# ============================================================
# LEVEL 1: ORCHESTRATOR GRAPH (Entry point)
# ============================================================

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    user_id: str
    session_id: str
    
    # Routing
    detected_intent: str
    selected_domain: str
    
    # Results
    domain_result: dict
    final_response: str
    
    # Metadata
    request_metadata: dict


# Tools for the orchestrator to invoke domain graphs
@tool
def invoke_career_domain(query: str, tenant_id: str, user_id: str) -> str:
    """Invoke the career development domain graph."""
    result = registry.invoke(
        "career_domain",
        {
            "messages": [HumanMessage(content=query)],
            "tenant_id": tenant_id,
            "user_id": user_id,
            "domain": "career",
            "subgraph_results": {},
            "final_output": ""
        }
    )
    return result.get("final_output", "No result")


@tool
def invoke_performance_domain(query: str, tenant_id: str, user_id: str) -> str:
    """Invoke the performance management domain graph."""
    result = registry.invoke(
        "performance_domain",
        {
            "messages": [HumanMessage(content=query)],
            "tenant_id": tenant_id,
            "user_id": user_id,
            "domain": "performance",
            "subgraph_results": {},
            "final_output": ""
        }
    )
    return result.get("final_output", "No result")


@tool
def list_available_domains(tenant_id: str) -> str:
    """List domains available for this tenant."""
    tenant = get_tenant_config(tenant_id)
    
    available = []
    for name in registry.list_graphs():
        graph_info = registry.get(name)
        if graph_info:
            tier_required = graph_info.get("metadata", {}).get("tier_required", "basic")
            tier_order = {"basic": 0, "professional": 1, "enterprise": 2}
            
            if tier_order.get(tenant.tier, 0) >= tier_order.get(tier_required, 0):
                available.append({
                    "name": name,
                    "description": graph_info.get("metadata", {}).get("description", "")
                })
    
    return json.dumps(available, indent=2)


ORCHESTRATOR_TOOLS = [invoke_career_domain, invoke_performance_domain, list_available_domains]


def build_orchestrator_graph():
    """Build the top-level orchestrator graph."""
    
    def authenticate_and_configure(state: OrchestratorState) -> dict:
        """Authenticate tenant and load configuration."""
        tenant_id = state.get("tenant_id", "startup_inc")
        tenant = get_tenant_config(tenant_id)
        
        return {
            "request_metadata": {
                "tenant": tenant.name,
                "tier": tenant.tier,
                "features": tenant.features,
                "authenticated_at": datetime.now().isoformat()
            }
        }
    
    def route_to_domain(state: OrchestratorState) -> dict:
        """Classify intent and route to appropriate domain."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        prompt = f"""Classify this HR coaching request:
"{user_message}"

Categories:
- career: Career development, skills, goals, promotions, transitions
- performance: Performance reviews, feedback, improvement plans
- general: Other HR questions

Respond with just the category name."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()
        
        if intent not in ["career", "performance", "general"]:
            intent = "general"
        
        return {
            "detected_intent": intent,
            "selected_domain": f"{intent}_domain" if intent != "general" else "general"
        }
    
    def invoke_domain_graph(state: OrchestratorState) -> dict:
        """Invoke the selected domain graph."""
        domain = state.get("selected_domain", "general")
        tenant_id = state.get("tenant_id", "startup_inc")
        user_id = state.get("user_id", "anonymous")
        query = state["messages"][-1].content if state["messages"] else ""
        
        if domain == "career_domain":
            result = registry.invoke(
                "career_domain",
                {
                    "messages": [HumanMessage(content=query)],
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "domain": "career",
                    "subgraph_results": {},
                    "final_output": ""
                }
            )
        elif domain == "performance_domain":
            result = registry.invoke(
                "performance_domain",
                {
                    "messages": [HumanMessage(content=query)],
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "domain": "performance",
                    "subgraph_results": {},
                    "final_output": ""
                }
            )
        else:
            # General fallback
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            response = llm.invoke([
                SystemMessage(content="You are a helpful HR coaching assistant."),
                HumanMessage(content=query)
            ])
            result = {"final_output": response.content}
        
        return {
            "domain_result": result,
            "final_response": result.get("final_output", ""),
            "messages": [AIMessage(content=result.get("final_output", ""))]
        }
    
    graph = StateGraph(OrchestratorState)
    
    graph.add_node("authenticate", authenticate_and_configure)
    graph.add_node("route", route_to_domain)
    graph.add_node("invoke_domain", invoke_domain_graph)
    
    graph.add_edge(START, "authenticate")
    graph.add_edge("authenticate", "route")
    graph.add_edge("route", "invoke_domain")
    graph.add_edge("invoke_domain", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# API ENDPOINT SIMULATION
# ============================================================

class CoachingAPIResponse:
    """Simulated API response structure."""
    
    def __init__(self, 
                 success: bool, 
                 data: Any, 
                 metadata: dict,
                 request_id: str):
        self.success = success
        self.data = data
        self.metadata = metadata
        self.request_id = request_id
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "request_id": self.request_id,
            "timestamp": self.timestamp
        }


def api_invoke(
    tenant_id: str,
    user_id: str,
    message: str,
    session_id: str = None
) -> CoachingAPIResponse:
    """Simulated API endpoint for invoking the coaching system."""
    
    request_id = hashlib.md5(
        f"{tenant_id}{user_id}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    session_id = session_id or f"session_{request_id}"
    
    orchestrator = build_orchestrator_graph()
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        result = orchestrator.invoke({
            "messages": [HumanMessage(content=message)],
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "detected_intent": "",
            "selected_domain": "",
            "domain_result": {},
            "final_response": "",
            "request_metadata": {}
        }, config)
        
        return CoachingAPIResponse(
            success=True,
            data={
                "response": result.get("final_response", ""),
                "intent": result.get("detected_intent", ""),
                "domain": result.get("selected_domain", "")
            },
            metadata=result.get("request_metadata", {}),
            request_id=request_id
        )
        
    except Exception as e:
        return CoachingAPIResponse(
            success=False,
            data={"error": str(e)},
            metadata={"tenant_id": tenant_id},
            request_id=request_id
        )


# ============================================================
# MAIN
# ============================================================

def main():
    """Demo the hierarchical multi-graph system."""
    
    print("=" * 70)
    print("HIERARCHICAL MULTI-GRAPH COACHING PLATFORM")
    print("=" * 70)
    print("\nArchitecture:")
    print("  Level 1: Orchestrator (routing, auth, API)")
    print("  Level 2: Domain Graphs (career, performance)")
    print("  Level 3: Leaf Graphs (skills, goals, feedback)")
    print("\nRegistered Graphs:")
    for name in registry.list_graphs():
        info = registry.get(name)
        desc = info.get("metadata", {}).get("description", "")
        print(f"  • {name}: {desc}")
    
    print("\nTenants:")
    for tid, config in TENANT_CONFIGS.items():
        print(f"  • {config.name} ({config.tier}): {', '.join(config.features)}")
    
    print("\n" + "=" * 70)
    
    # Select tenant
    print("\nAvailable tenants: acme_corp, startup_inc")
    tenant_id = input("Select tenant [acme_corp]: ").strip() or "acme_corp"
    user_id = f"user_{tenant_id}"
    session_id = None
    
    print(f"\nConnected as: {get_tenant_config(tenant_id).name}")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        # Call API
        response = api_invoke(
            tenant_id=tenant_id,
            user_id=user_id,
            message=user_input,
            session_id=session_id
        )
        
        # Store session for continuity
        session_id = f"session_{response.request_id}"
        
        # Display response
        print(f"\n[Request: {response.request_id}]")
        print(f"[Intent: {response.data.get('intent', 'N/A')} → Domain: {response.data.get('domain', 'N/A')}]")
        print(f"\nCoach: {response.data.get('response', response.data.get('error', 'No response'))}\n")


if __name__ == "__main__":
    main()
