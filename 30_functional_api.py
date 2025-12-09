"""
30. LangGraph Functional API
=============================
Demonstrates LangGraph's Functional API as an alternative to StateGraph:

FUNCTIONAL API COMPONENTS:
├── @entrypoint: Define the main workflow function
├── @task: Define individual task functions
├── entrypoint.final(): Return final value with state to save
├── .result(): Await task completion
└── Automatic checkpointing and state management

ADVANTAGES:
├── More Pythonic - feels like regular function calls
├── Simpler mental model - no explicit graph construction
├── Automatic state persistence
├── Easy parallel task execution
└── Cleaner error handling

USE CASES:
- Simple workflows without complex branching
- Prototyping and rapid development
- When you prefer functional programming style
- Linear pipelines with occasional parallelism
"""

import asyncio
from typing import Any
from datetime import datetime

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig


# ============================================================
# CONFIGURATION
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# DEMO 1: BASIC FUNCTIONAL WORKFLOW
# ============================================================

# Define tasks with @task decorator
@task
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of the input text"""
    log("Analyzing sentiment...", "")
    
    response = llm.invoke([
        SystemMessage(content="Analyze the sentiment. Respond with: POSITIVE, NEGATIVE, or NEUTRAL"),
        HumanMessage(content=text)
    ])
    
    sentiment = response.content.strip().upper()
    return {"sentiment": sentiment, "original_text": text[:50]}


@task
def generate_response(sentiment: str, topic: str) -> str:
    """Generate a response based on sentiment"""
    log(f"Generating response for {sentiment} sentiment...", "")
    
    prompts = {
        "POSITIVE": "Write an enthusiastic, appreciative response",
        "NEGATIVE": "Write an empathetic, helpful response addressing concerns",
        "NEUTRAL": "Write a professional, informative response"
    }
    
    prompt = prompts.get(sentiment, prompts["NEUTRAL"])
    
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Topic: {topic}")
    ])
    
    return response.content


@task
def format_output(response: str, metadata: dict) -> dict:
    """Format the final output"""
    log("Formatting output...", "")
    
    return {
        "response": response,
        "metadata": metadata,
        "timestamp": datetime.now().isoformat()
    }


# Define the workflow with @entrypoint
@entrypoint(checkpointer=MemorySaver())
def basic_workflow(user_input: str, *, previous: dict = None) -> dict:
    """
    A basic functional workflow demonstrating @task and @entrypoint.
    
    Args:
        user_input: The user's message
        previous: Previous state (for checkpoint continuity)
    """
    log("Starting basic workflow...", "")
    
    # Call tasks and await results with .result()
    sentiment_result = analyze_sentiment(user_input).result()
    
    response = generate_response(
        sentiment_result["sentiment"],
        user_input
    ).result()
    
    output = format_output(response, sentiment_result).result()
    
    # Return final value - this gets checkpointed
    return entrypoint.final(value=output)


def demo_basic_workflow():
    """Demo: Basic functional workflow"""
    print("\n" + "="*70)
    print(" DEMO 1: BASIC FUNCTIONAL WORKFLOW")
    print("="*70)
    
    inputs = [
        "I absolutely love your product! It changed my life!",
        "Your service is terrible. I've been waiting for 3 weeks!",
        "Can you tell me about your return policy?"
    ]
    
    for user_input in inputs:
        print(f"\n Input: '{user_input[:50]}...'")
        
        result = basic_workflow.invoke(
            user_input,
            {"configurable": {"thread_id": f"basic-{hash(user_input)}"}}
        )
        
        print(f"   Sentiment: {result['metadata']['sentiment']}")
        print(f"   Response: {result['response'][:100]}...")


# ============================================================
# DEMO 2: PARALLEL TASK EXECUTION
# ============================================================

@task
def research_topic(topic: str) -> str:
    """Research a topic"""
    log(f"Researching: {topic}", "")
    
    response = llm.invoke([
        SystemMessage(content="Provide 2-3 key facts about this topic."),
        HumanMessage(content=topic)
    ])
    return response.content


@task
def get_related_topics(topic: str) -> list[str]:
    """Get related topics"""
    log(f"Finding related topics for: {topic}", "")
    
    response = llm.invoke([
        SystemMessage(content="List 3 related topics, one per line."),
        HumanMessage(content=topic)
    ])
    return response.content.strip().split("\n")[:3]


@task
def generate_summary(research: str, related: list[str]) -> str:
    """Generate a summary combining research and related topics"""
    log("Generating summary...", "")
    
    response = llm.invoke([
        SystemMessage(content="Create a brief summary incorporating the research and mentioning related topics."),
        HumanMessage(content=f"Research:\n{research}\n\nRelated topics: {', '.join(related)}")
    ])
    return response.content


@entrypoint(checkpointer=MemorySaver())
def parallel_workflow(topic: str) -> dict:
    """Workflow with parallel task execution"""
    log("Starting parallel workflow...", "")
    
    # Launch tasks in parallel (don't call .result() yet)
    research_task = research_topic(topic)
    related_task = get_related_topics(topic)
    
    # Now await results (tasks ran in parallel)
    research = research_task.result()
    related = related_task.result()
    
    log("Parallel tasks completed", "")
    
    # Sequential task using parallel results
    summary = generate_summary(research, related).result()
    
    return entrypoint.final(value={
        "topic": topic,
        "research": research,
        "related_topics": related,
        "summary": summary
    })


def demo_parallel_tasks():
    """Demo: Parallel task execution"""
    print("\n" + "="*70)
    print("⚡ DEMO 2: PARALLEL TASK EXECUTION")
    print("="*70)
    
    print("\n Running parallel research workflow...")
    
    result = parallel_workflow.invoke(
        "artificial intelligence ethics",
        {"configurable": {"thread_id": "parallel-demo"}}
    )
    
    print(f"\n Results:")
    print(f"   Topic: {result['topic']}")
    print(f"   Related: {result['related_topics']}")
    print(f"   Summary: {result['summary'][:200]}...")


# ============================================================
# DEMO 3: CONVERSATIONAL WORKFLOW WITH STATE
# ============================================================

@task
def process_message(message: str, history: list[dict]) -> str:
    """Process a user message with conversation history"""
    log(f"Processing message with {len(history)} previous turns...", "")
    
    # Build messages from history
    messages = [SystemMessage(content="You are a helpful assistant. Be conversational.")]
    
    for h in history[-5:]:  # Last 5 turns
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))
    
    messages.append(HumanMessage(content=message))
    
    response = llm.invoke(messages)
    return response.content


@entrypoint(checkpointer=MemorySaver())
def conversational_workflow(
    message: str,
    *,
    previous: dict = None,
    config: RunnableConfig
) -> dict:
    """
    Conversational workflow that maintains history across calls.
    
    Uses 'previous' to access state from previous invocations.
    """
    # Get history from previous state or start fresh
    history = (previous or {}).get("history", [])
    
    log(f"Conversation turn {len(history) // 2 + 1}", "")
    
    # Process the message
    response = process_message(message, history).result()
    
    # Update history
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    
    # Return with state to save
    return entrypoint.final(
        value={"response": response, "turn": len(new_history) // 2},
        save={"history": new_history}  # This gets persisted
    )


def demo_conversational():
    """Demo: Conversational workflow with persistent state"""
    print("\n" + "="*70)
    print(" DEMO 3: CONVERSATIONAL WORKFLOW")
    print("="*70)
    
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    messages = [
        "Hi! My name is Alex.",
        "What's my name?",
        "Tell me a joke.",
        "That was funny! What's my name again?"
    ]
    
    for msg in messages:
        print(f"\n User: {msg}")
        
        result = conversational_workflow.invoke(msg, config)
        
        print(f" Assistant: {result['response']}")
        print(f"   (Turn {result['turn']})")


# ============================================================
# DEMO 4: WORKFLOW WITH STORE (Cross-Thread Memory)
# ============================================================

@task
def save_user_info(store: BaseStore, user_id: str, info: dict) -> str:
    """Save user information to store"""
    log(f"Saving info for user {user_id}", "")
    
    namespace = ("users", user_id)
    for key, value in info.items():
        store.put(namespace, key, {"value": value, "updated": datetime.now().isoformat()})
    
    return f"Saved {len(info)} items"


@task
def get_user_info(store: BaseStore, user_id: str) -> dict:
    """Get user information from store"""
    log(f"Loading info for user {user_id}", "")
    
    namespace = ("users", user_id)
    items = store.search(namespace)
    
    return {item.key: item.value.get("value") for item in items}


@task
def personalized_greeting(user_info: dict, message: str) -> str:
    """Generate a personalized greeting"""
    log("Generating personalized response...", "")
    
    context = ""
    if user_info:
        context = f"User info: {user_info}\n"
    
    response = llm.invoke([
        SystemMessage(content=f"You are a helpful assistant. {context}Personalize your response based on what you know about the user."),
        HumanMessage(content=message)
    ])
    
    return response.content


# Create store for this workflow
memory_store = InMemoryStore()


@entrypoint(checkpointer=MemorySaver(), store=memory_store)
def memory_workflow(
    action: str,
    user_id: str,
    data: dict = None,
    *,
    store: BaseStore
) -> dict:
    """
    Workflow demonstrating cross-thread memory with Store.
    
    Actions: 'save', 'greet'
    """
    if action == "save" and data:
        result = save_user_info(store, user_id, data).result()
        return entrypoint.final(value={"action": "save", "result": result})
    
    elif action == "greet":
        user_info = get_user_info(store, user_id).result()
        message = data.get("message", "Hello!") if data else "Hello!"
        greeting = personalized_greeting(user_info, message).result()
        return entrypoint.final(value={
            "action": "greet",
            "user_info": user_info,
            "response": greeting
        })
    
    return entrypoint.final(value={"error": "Unknown action"})


def demo_with_store():
    """Demo: Workflow with cross-thread memory store"""
    print("\n" + "="*70)
    print(" DEMO 4: WORKFLOW WITH MEMORY STORE")
    print("="*70)
    
    user_id = "user-functional-api"
    
    # Thread 1: Save user info
    print("\n📱 Thread 1: Saving user preferences")
    result1 = memory_workflow.invoke(
        {"action": "save", "user_id": user_id, "data": {"name": "Jordan", "favorite_color": "blue", "hobby": "hiking"}},
        {"configurable": {"thread_id": "thread-1"}}
    )
    print(f"   Result: {result1['result']}")
    
    # Thread 2: Different thread, same user - memory persists!
    print("\n📱 Thread 2: New thread, accessing same user's memory")
    result2 = memory_workflow.invoke(
        {"action": "greet", "user_id": user_id, "data": {"message": "What activities would you recommend for me?"}},
        {"configurable": {"thread_id": "thread-2"}}  # Different thread!
    )
    print(f"   User Info Retrieved: {result2['user_info']}")
    print(f"   Personalized Response: {result2['response'][:150]}...")


# ============================================================
# DEMO 5: ERROR HANDLING
# ============================================================

@task
def risky_task(should_fail: bool) -> str:
    """A task that might fail"""
    log(f"Running risky task (should_fail={should_fail})", "")
    
    if should_fail:
        raise ValueError("Simulated failure!")
    
    return "Success!"


@task
def fallback_task() -> str:
    """Fallback when main task fails"""
    log("Running fallback...", "")
    return "Fallback result"


@entrypoint(checkpointer=MemorySaver())
def error_handling_workflow(should_fail: bool = False) -> dict:
    """Workflow demonstrating error handling"""
    log("Starting error handling workflow...", "")
    
    try:
        result = risky_task(should_fail).result()
        return entrypoint.final(value={"status": "success", "result": result})
    
    except ValueError as e:
        log(f"Caught error: {e}", "")
        fallback = fallback_task().result()
        return entrypoint.final(value={"status": "fallback", "result": fallback, "error": str(e)})


def demo_error_handling():
    """Demo: Error handling in functional workflows"""
    print("\n" + "="*70)
    print("🛡️ DEMO 5: ERROR HANDLING")
    print("="*70)
    
    # Test success case
    print("\n Test 1: Success case")
    result1 = error_handling_workflow.invoke(
        False,
        {"configurable": {"thread_id": "error-1"}}
    )
    print(f"   Status: {result1['status']}")
    print(f"   Result: {result1['result']}")
    
    # Test failure case with fallback
    print("\n Test 2: Failure with fallback")
    result2 = error_handling_workflow.invoke(
        True,
        {"configurable": {"thread_id": "error-2"}}
    )
    print(f"   Status: {result2['status']}")
    print(f"   Error: {result2.get('error')}")
    print(f"   Result: {result2['result']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║             LangGraph Functional API Demo                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Functional API Components:                                      ║
║  • @entrypoint - Define main workflow                            ║
║  • @task - Define individual tasks                               ║
║  • .result() - Await task completion                             ║
║  • entrypoint.final() - Return with state                        ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_basic_workflow()
        demo_parallel_tasks()
        demo_conversational()
        demo_with_store()
        demo_error_handling()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:

1. @entrypoint defines the main workflow:
   @entrypoint(checkpointer=MemorySaver())
   def my_workflow(input, *, previous=None):
       ...

2. @task defines individual tasks:
   @task
   def my_task(arg):
       return result

3. Execute tasks and await results:
   result = my_task(arg).result()

4. Parallel execution (launch then await):
   task1 = my_task_a(x)  # Starts immediately
   task2 = my_task_b(y)  # Runs in parallel
   r1, r2 = task1.result(), task2.result()

5. Return with state persistence:
   return entrypoint.final(
       value={"response": ...},  # Return value
       save={"history": ...}     # State to persist
   )

6. Access previous state via 'previous' parameter:
   def my_workflow(input, *, previous=None):
       history = (previous or {}).get("history", [])

WHEN TO USE:
- Functional API: Simple/linear workflows, rapid prototyping
- StateGraph: Complex branching, explicit control flow
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
