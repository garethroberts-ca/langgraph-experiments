"""
27. Prebuilt ReAct Agent with LangGraph
========================================
Demonstrates LangGraph's prebuilt components for rapid agent development:

PREBUILT COMPONENTS:
├── create_react_agent: One-line ReAct agent creation
├── ToolNode: Prebuilt tool execution node
├── tools_condition: Routing helper for tool calls
├── InjectedState: Access graph state in tools
└── InjectedStore: Access memory store in tools

CUSTOMIZATION OPTIONS:
├── Custom prompts (system message)
├── State modifiers (pre/post processing)
├── Custom checkpointing
├── Tool configuration
└── Message handling

USE CASES:
- Rapid prototyping of agents
- Standard ReAct patterns
- Tool-calling agents
- Conversational assistants
"""

from typing import Annotated, TypedDict, Literal, Any
from datetime import datetime

from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition, InjectedState
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, BaseTool


# ============================================================
# CONFIGURATION
# ============================================================

def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# DEFINE TOOLS
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "san francisco": "Foggy, 58°F",
        "new york": "Sunny, 72°F",
        "london": "Rainy, 52°F",
        "tokyo": "Clear, 68°F",
        "sydney": "Warm, 78°F"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_hr_policies(query: str) -> str:
    """Search HR policies and guidelines."""
    policies = {
        "vacation": "Employees receive 20 days PTO annually. Unused days roll over up to 5 days.",
        "remote": "Remote work allowed 3 days/week. Must be available during core hours 10am-4pm.",
        "sick leave": "Unlimited sick leave with manager notification. Doctor's note required after 3 consecutive days.",
        "parental": "16 weeks paid parental leave for all parents.",
        "expense": "Submit expenses within 30 days. Meals up to $75/day when traveling."
    }
    
    for key, value in policies.items():
        if key in query.lower():
            return value
    
    return "No specific policy found. Please contact HR at hr@company.com"


@tool
def schedule_meeting(
    title: str,
    attendees: list[str],
    duration_minutes: int = 30
) -> str:
    """Schedule a meeting with specified attendees."""
    log(f"Scheduling: {title} with {attendees} for {duration_minutes}min", "")
    meeting_id = f"MTG-{datetime.now().strftime('%Y%m%d%H%M')}"
    return f"Meeting '{title}' scheduled. ID: {meeting_id}. Invites sent to {', '.join(attendees)}."


@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to a user."""
    log(f"Notification to {recipient}: {message[:50]}...", "")
    return f"Notification sent to {recipient}."


@tool
def calculate_pto_balance(employee_id: str) -> str:
    """Calculate remaining PTO balance for an employee."""
    # Simulated data
    balances = {
        "E001": {"used": 8, "remaining": 12, "pending": 2},
        "E002": {"used": 15, "remaining": 5, "pending": 0},
        "E003": {"used": 3, "remaining": 17, "pending": 5}
    }
    
    if employee_id in balances:
        b = balances[employee_id]
        return f"PTO Balance for {employee_id}: {b['remaining']} days remaining ({b['used']} used, {b['pending']} pending approval)"
    
    return f"Employee {employee_id} not found in system."


# Collect all tools
hr_tools = [
    get_weather,
    search_hr_policies,
    schedule_meeting,
    send_notification,
    calculate_pto_balance
]


# ============================================================
# DEMO 1: BASIC create_react_agent
# ============================================================

def demo_basic_react_agent():
    """Demo: Simplest way to create a ReAct agent"""
    print("\n" + "="*70)
    print(" DEMO 1: BASIC create_react_agent")
    print("="*70)
    
    # One line to create a full ReAct agent!
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, search_hr_policies]
    )
    
    # Run it
    print("\n Query: 'What's the weather in San Francisco and what's the vacation policy?'")
    
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco and what's the vacation policy?")]
    })
    
    # Show the conversation
    print("\n Agent Response:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   {msg.content}")


# ============================================================
# DEMO 2: REACT AGENT WITH CUSTOM PROMPT
# ============================================================

def demo_custom_prompt():
    """Demo: ReAct agent with customized system prompt"""
    print("\n" + "="*70)
    print(" DEMO 2: CUSTOM SYSTEM PROMPT")
    print("="*70)
    
    # Custom prompt for HR assistant persona
    hr_prompt = """You are an HR Assistant for TechCorp.

Your responsibilities:
1. Answer questions about company policies
2. Help employees check their PTO balance
3. Assist with scheduling meetings
4. Provide general HR guidance

Always be helpful, professional, and empathetic.
If you don't know something, direct employees to hr@techcorp.com.

Current date: {current_date}"""

    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=hr_tools,
        prompt=hr_prompt.format(current_date=datetime.now().strftime("%Y-%m-%d"))
    )
    
    print("\n Query: 'Can you check my PTO balance? My ID is E001'")
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Can you check my PTO balance? My ID is E001")]
    })
    
    print("\n HR Assistant Response:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   {msg.content}")


# ============================================================
# DEMO 3: REACT AGENT WITH CHECKPOINTING
# ============================================================

def demo_with_checkpointing():
    """Demo: ReAct agent with conversation memory"""
    print("\n" + "="*70)
    print(" DEMO 3: AGENT WITH CHECKPOINTING (Memory)")
    print("="*70)
    
    checkpointer = MemorySaver()
    
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=hr_tools,
        prompt="You are a helpful HR assistant. Remember context from our conversation.",
        checkpointer=checkpointer
    )
    
    config = {"configurable": {"thread_id": "hr-session-1"}}
    
    # Turn 1
    print("\n Turn 1: 'Hi, my employee ID is E002'")
    result1 = agent.invoke({
        "messages": [HumanMessage(content="Hi, my employee ID is E002")]
    }, config)
    
    for msg in result1["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   Assistant: {msg.content}")
    
    # Turn 2 - agent should remember the employee ID
    print("\n Turn 2: 'What's my PTO balance?'")
    result2 = agent.invoke({
        "messages": [HumanMessage(content="What's my PTO balance?")]
    }, config)
    
    for msg in result2["messages"][-2:]:  # Last messages
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   Assistant: {msg.content}")
    
    # Turn 3
    print("\n Turn 3: 'And what's the remote work policy?'")
    result3 = agent.invoke({
        "messages": [HumanMessage(content="And what's the remote work policy?")]
    }, config)
    
    for msg in result3["messages"][-2:]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   Assistant: {msg.content}")


# ============================================================
# DEMO 4: TOOLNODE - STANDALONE TOOL EXECUTION
# ============================================================

def demo_tool_node():
    """Demo: Using ToolNode for tool execution"""
    print("\n" + "="*70)
    print(" DEMO 4: STANDALONE TOOLNODE")
    print("="*70)
    
    # Create a ToolNode
    tool_node = ToolNode(tools=[get_weather, search_hr_policies])
    
    # Simulate an AI message with tool calls
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "get_weather",
                "args": {"city": "Tokyo"}
            },
            {
                "id": "call_2", 
                "name": "search_hr_policies",
                "args": {"query": "vacation policy"}
            }
        ]
    )
    
    # Execute tools via ToolNode
    print("\n Executing tool calls via ToolNode:")
    print(f"   - get_weather(city='Tokyo')")
    print(f"   - search_hr_policies(query='vacation policy')")
    
    result = tool_node.invoke({"messages": [ai_message]})
    
    print("\n📤 Tool Results:")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"   [{msg.name}]: {msg.content}")


# ============================================================
# DEMO 5: CUSTOM REACT AGENT FROM SCRATCH
# ============================================================

def demo_custom_react():
    """Demo: Building a custom ReAct agent for more control"""
    print("\n" + "="*70)
    print("🛠️ DEMO 5: CUSTOM REACT AGENT (Manual Build)")
    print("="*70)
    
    # For when you need more control than create_react_agent provides
    
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(hr_tools)
    
    def agent_node(state: MessagesState) -> dict:
        """The agent reasoning node"""
        system = SystemMessage(content="""You are an HR assistant.
Think step by step before using tools.
Always explain what you're doing.""")
        
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        
        return {"messages": [response]}
    
    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        """Decide whether to call tools or finish"""
        last_message = state["messages"][-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "__end__"
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools=hr_tools))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END}
    )
    workflow.add_edge("tools", "agent")  # After tools, go back to agent
    
    graph = workflow.compile()
    
    # Run it
    print("\n Query: 'Schedule a meeting about performance reviews with alice@company.com'")
    
    result = graph.invoke({
        "messages": [HumanMessage(
            content="Schedule a meeting about performance reviews with alice@company.com"
        )]
    })
    
    print("\n Custom Agent Execution:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"    Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
            if msg.content:
                print(f"    {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"   📤 [{msg.name}]: {msg.content[:80]}...")


# ============================================================
# DEMO 6: TOOLS_CONDITION HELPER
# ============================================================

def demo_tools_condition():
    """Demo: Using the tools_condition helper function"""
    print("\n" + "="*70)
    print(" DEMO 6: tools_condition HELPER")
    print("="*70)
    
    # tools_condition is a prebuilt function that checks if the last message
    # has tool calls and returns "tools" or "__end__"
    
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])
    
    def call_model(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}
    
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools=[get_weather]))
    
    workflow.add_edge(START, "agent")
    
    # Use the prebuilt tools_condition instead of writing our own
    workflow.add_conditional_edges(
        "agent",
        tools_condition,  # Returns "tools" or "__end__" automatically
    )
    workflow.add_edge("tools", "agent")
    
    graph = workflow.compile()
    
    print("\n Using tools_condition for automatic routing")
    print("   (Returns 'tools' if tool_calls exist, else '__end__')")
    
    # Test with a query that needs tools
    result = graph.invoke({
        "messages": [HumanMessage(content="What's the weather in London?")]
    })
    
    print("\n Result:")
    final_msg = result["messages"][-1]
    if isinstance(final_msg, AIMessage) and final_msg.content:
        print(f"   {final_msg.content}")


# ============================================================
# DEMO 7: STATE MODIFIER
# ============================================================

def demo_state_modifier():
    """Demo: Using state modifier for pre/post processing"""
    print("\n" + "="*70)
    print(" DEMO 7: STATE MODIFIER")
    print("="*70)
    
    def add_context(messages: list) -> list:
        """Pre-process messages to add context"""
        context = SystemMessage(content=f"""
Current time: {datetime.now().strftime('%H:%M')}
User timezone: PST
Company: TechCorp
        """)
        return [context] + messages
    
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        state_modifier=add_context  # Applied before each LLM call
    )
    
    print("\n State modifier adds context before each LLM call")
    print("   (Adds current time, timezone, company info)")
    
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in New York?")]
    })
    
    print("\n Response:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"   {msg.content}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           LangGraph Prebuilt ReAct Agent Demo                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Prebuilt Components:                                            ║
║  • create_react_agent - One-line agent creation                  ║
║  • ToolNode - Prebuilt tool executor                             ║
║  • tools_condition - Routing helper                              ║
║  • state_modifier - Pre/post processing                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_basic_react_agent()
        demo_custom_prompt()
        demo_with_checkpointing()
        demo_tool_node()
        demo_custom_react()
        demo_tools_condition()
        demo_state_modifier()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:

1. create_react_agent() - Fastest way to build a ReAct agent:
   agent = create_react_agent(model="openai:gpt-4o-mini", tools=[...])

2. ToolNode - Prebuilt node for executing tools:
   tool_node = ToolNode(tools=[...])

3. tools_condition - Helper for conditional routing:
   workflow.add_conditional_edges("agent", tools_condition)

4. Customization options:
   - prompt: Custom system message
   - checkpointer: Memory persistence
   - state_modifier: Pre-process messages

5. When to use custom vs prebuilt:
   - create_react_agent: Quick prototypes, standard patterns
   - Custom build: Complex routing, multiple agent types, special state
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
