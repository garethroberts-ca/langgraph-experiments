"""
26. Cross-Thread Long-Term Memory in LangGraph
================================================
Demonstrates LangGraph's Store interface for persistent memory across threads:

FEATURES:
├── Store Interface: put/get/search/delete operations
├── Namespacing: Organize memories by user/org/context
├── Cross-Thread: Access memories from any conversation
├── Semantic Search: Find memories by meaning (with embeddings)
├── TTL Support: Auto-expire memories
└── Structured Data: Store JSON documents

USE CASES:
- User preferences across sessions
- Organizational knowledge base
- Learning from past interactions
- Personalized assistants
"""

import uuid
from typing import Annotated, TypedDict, Optional, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig


# ============================================================
# CONFIGURATION
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def log(msg: str, icon: str = ""):
    print(f"  {icon} {msg}")


# ============================================================
# STATE DEFINITION
# ============================================================

class AssistantState(TypedDict):
    """State for the memory-aware assistant"""
    messages: list
    user_id: str
    current_memories: list[dict]
    response: str


# ============================================================
# MEMORY STORE SETUP
# ============================================================

def create_memory_store(use_embeddings: bool = False) -> InMemoryStore:
    """
    Create an InMemoryStore, optionally with semantic search.
    
    For production, use PostgresStore or other persistent stores.
    """
    if use_embeddings:
        # With embeddings for semantic search
        return InMemoryStore(
            index={
                "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
                "dims": 1536
            }
        )
    else:
        # Simple store without embeddings
        return InMemoryStore()


# ============================================================
# MEMORY OPERATIONS
# ============================================================

class MemoryManager:
    """Helper class for common memory operations"""
    
    def __init__(self, store: BaseStore):
        self.store = store
    
    def save_user_preference(self, user_id: str, key: str, value: Any):
        """Save a user preference"""
        namespace = ("users", user_id, "preferences")
        self.store.put(
            namespace,
            key,
            {"value": value, "updated_at": datetime.now().isoformat()}
        )
        log(f"Saved preference: {key}={value}", "")
    
    def get_user_preference(self, user_id: str, key: str) -> Optional[Any]:
        """Get a specific user preference"""
        namespace = ("users", user_id, "preferences")
        item = self.store.get(namespace, key)
        if item:
            return item.value.get("value")
        return None
    
    def save_memory(self, user_id: str, content: str, memory_type: str = "general"):
        """Save a new memory about the user"""
        namespace = ("users", user_id, "memories")
        memory_id = str(uuid.uuid4())
        self.store.put(
            namespace,
            memory_id,
            {
                "content": content,
                "type": memory_type,
                "created_at": datetime.now().isoformat()
            }
        )
        log(f"Saved memory: {content[:50]}...", "")
        return memory_id
    
    def search_memories(self, user_id: str, query: str = None, limit: int = 5) -> list[dict]:
        """Search user memories"""
        namespace = ("users", user_id, "memories")
        
        if query:
            # Semantic search if embeddings enabled
            results = self.store.search(namespace, query=query, limit=limit)
        else:
            # List all
            results = self.store.search(namespace, limit=limit)
        
        return [
            {
                "key": item.key,
                "content": item.value.get("content", ""),
                "type": item.value.get("type", "general"),
                "created_at": item.value.get("created_at", "")
            }
            for item in results
        ]
    
    def get_all_preferences(self, user_id: str) -> dict:
        """Get all preferences for a user"""
        namespace = ("users", user_id, "preferences")
        results = self.store.search(namespace)
        return {item.key: item.value.get("value") for item in results}
    
    def delete_memory(self, user_id: str, memory_id: str):
        """Delete a specific memory"""
        namespace = ("users", user_id, "memories")
        self.store.delete(namespace, memory_id)
        log(f"Deleted memory: {memory_id}", "")


# ============================================================
# GRAPH NODES
# ============================================================

def load_user_context(state: AssistantState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Load user memories and preferences at the start"""
    user_id = state.get("user_id", "anonymous")
    
    log(f"Loading context for user: {user_id}", "")
    
    memory_manager = MemoryManager(store)
    
    # Get preferences
    preferences = memory_manager.get_all_preferences(user_id)
    
    # Get recent memories
    memories = memory_manager.search_memories(user_id, limit=10)
    
    # Build context
    context_parts = []
    
    if preferences:
        prefs_str = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
        context_parts.append(f"User preferences: {prefs_str}")
    
    if memories:
        mem_str = "\n".join([f"- {m['content']}" for m in memories[:5]])
        context_parts.append(f"Things I remember about this user:\n{mem_str}")
    
    log(f"Loaded {len(preferences)} preferences, {len(memories)} memories", "")
    
    return {
        "current_memories": memories
    }


def process_and_respond(state: AssistantState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Process the message and respond, learning from the conversation"""
    user_id = state.get("user_id", "anonymous")
    messages = state.get("messages", [])
    memories = state.get("current_memories", [])
    
    memory_manager = MemoryManager(store)
    
    # Build system prompt with memory context
    memory_context = ""
    if memories:
        memory_context = "\n\nThings I know about this user:\n" + \
                        "\n".join([f"- {m['content']}" for m in memories[:5]])
    
    # Check for preference updates in the message
    last_message = messages[-1].content if messages else ""
    
    # Simple pattern matching for preferences (in production, use LLM to extract)
    preference_keywords = {
        "call me": "name",
        "my name is": "name",
        "prefer": "preference",
        "like": "likes",
        "don't like": "dislikes",
        "hate": "dislikes"
    }
    
    # Extract and save any new information
    for keyword, pref_type in preference_keywords.items():
        if keyword.lower() in last_message.lower():
            # Save as a memory
            memory_manager.save_memory(
                user_id,
                f"User mentioned: {last_message[:100]}",
                memory_type="user_statement"
            )
            break
    
    # Generate response with context
    system_prompt = f"""You are a helpful assistant with memory capabilities.
You remember things about users across conversations.
{memory_context}

If the user tells you something about themselves (name, preferences, etc.),
acknowledge that you'll remember it.

Be conversational and reference what you know about them when relevant."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])
    
    return {
        "messages": messages + [response],
        "response": response.content
    }


def save_conversation_summary(state: AssistantState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Save a summary of the conversation for future reference"""
    user_id = state.get("user_id", "anonymous")
    messages = state.get("messages", [])
    
    if len(messages) < 2:
        return {}
    
    memory_manager = MemoryManager(store)
    
    # Generate a brief summary of what was discussed
    summary_prompt = """Summarize the key information about the user from this conversation in one sentence.
If there's nothing notable about the user, respond with 'Nothing notable'."""
    
    summary_response = llm.invoke([
        SystemMessage(content=summary_prompt),
        *messages[-4:]  # Last few messages
    ])
    
    summary = summary_response.content
    
    if "nothing notable" not in summary.lower():
        memory_manager.save_memory(user_id, summary, memory_type="conversation_summary")
    
    return {}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_memory_assistant():
    """Build the memory-aware assistant graph"""
    
    workflow = StateGraph(AssistantState)
    
    # Add nodes
    workflow.add_node("load_context", load_user_context)
    workflow.add_node("respond", process_and_respond)
    workflow.add_node("save_summary", save_conversation_summary)
    
    # Define flow
    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "respond")
    workflow.add_edge("respond", "save_summary")
    workflow.add_edge("save_summary", END)
    
    # Create stores
    memory_store = create_memory_store(use_embeddings=False)
    checkpointer = MemorySaver()
    
    # Compile with both checkpointer and store
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=memory_store
    )
    
    return graph, memory_store


# ============================================================
# DEMO: BASIC CROSS-THREAD MEMORY
# ============================================================

def demo_cross_thread_memory():
    """Demo: Memory persists across different threads (conversations)"""
    print("\n" + "="*70)
    print(" DEMO 1: CROSS-THREAD MEMORY")
    print("="*70)
    
    graph, store = build_memory_assistant()
    memory_manager = MemoryManager(store)
    
    user_id = "user-123"
    
    # Thread 1: User introduces themselves
    print("\n📱 THREAD 1: Introduction")
    print("-" * 40)
    
    config_1 = {
        "configurable": {
            "thread_id": "thread-1",
            "user_id": user_id
        }
    }
    
    result_1 = graph.invoke({
        "messages": [HumanMessage(content="Hi! My name is Sarah and I'm a software engineer.")],
        "user_id": user_id
    }, config_1)
    
    print(f"User: Hi! My name is Sarah and I'm a software engineer.")
    print(f"Assistant: {result_1['response']}")
    
    # Manually save some preferences
    memory_manager.save_user_preference(user_id, "name", "Sarah")
    memory_manager.save_user_preference(user_id, "occupation", "software engineer")
    
    # Thread 2: Different conversation, same user
    print("\n📱 THREAD 2: New conversation (different thread)")
    print("-" * 40)
    
    config_2 = {
        "configurable": {
            "thread_id": "thread-2",  # Different thread!
            "user_id": user_id
        }
    }
    
    result_2 = graph.invoke({
        "messages": [HumanMessage(content="What do you remember about me?")],
        "user_id": user_id
    }, config_2)
    
    print(f"User: What do you remember about me?")
    print(f"Assistant: {result_2['response']}")
    
    # Thread 3: Yet another conversation
    print("\n📱 THREAD 3: Another new conversation")
    print("-" * 40)
    
    config_3 = {
        "configurable": {
            "thread_id": "thread-3",
            "user_id": user_id
        }
    }
    
    result_3 = graph.invoke({
        "messages": [HumanMessage(content="I just got promoted to senior engineer!")],
        "user_id": user_id
    }, config_3)
    
    print(f"User: I just got promoted to senior engineer!")
    print(f"Assistant: {result_3['response']}")
    
    # Show all stored memories
    print("\n📚 ALL STORED MEMORIES FOR USER")
    print("-" * 40)
    all_memories = memory_manager.search_memories(user_id, limit=20)
    for mem in all_memories:
        print(f"  [{mem['type']}] {mem['content'][:60]}...")


# ============================================================
# DEMO: NAMESPACE ORGANIZATION
# ============================================================

def demo_namespaces():
    """Demo: Organizing memories with namespaces"""
    print("\n" + "="*70)
    print("📁 DEMO 2: NAMESPACE ORGANIZATION")
    print("="*70)
    
    store = create_memory_store()
    
    # Different namespace patterns
    
    # 1. User-specific memories
    user_namespace = ("users", "user-456", "memories")
    store.put(user_namespace, "mem-1", {
        "content": "User prefers morning meetings",
        "type": "preference"
    })
    
    # 2. Organization-wide knowledge
    org_namespace = ("org", "acme-corp", "policies")
    store.put(org_namespace, "vacation", {
        "content": "Employees get 20 days PTO per year",
        "type": "policy"
    })
    store.put(org_namespace, "remote", {
        "content": "Remote work allowed 3 days per week",
        "type": "policy"
    })
    
    # 3. Project-specific context
    project_namespace = ("org", "acme-corp", "projects", "project-x")
    store.put(project_namespace, "deadline", {
        "content": "Project X deadline is Q2 2025",
        "type": "milestone"
    })
    
    # 4. Shared resources
    shared_namespace = ("shared", "templates")
    store.put(shared_namespace, "email-intro", {
        "content": "Dear [Name], Thank you for reaching out...",
        "type": "template"
    })
    
    # Query different namespaces
    print("\n User Memories:")
    for item in store.search(user_namespace):
        print(f"  - {item.key}: {item.value['content']}")
    
    print("\n Organization Policies:")
    for item in store.search(org_namespace):
        print(f"  - {item.key}: {item.value['content']}")
    
    print("\n Project Context:")
    for item in store.search(project_namespace):
        print(f"  - {item.key}: {item.value['content']}")
    
    print("\n Shared Templates:")
    for item in store.search(shared_namespace):
        print(f"  - {item.key}: {item.value['content'][:40]}...")


# ============================================================
# DEMO: MULTI-USER ISOLATION
# ============================================================

def demo_multi_user():
    """Demo: Each user's memories are isolated"""
    print("\n" + "="*70)
    print("👥 DEMO 3: MULTI-USER ISOLATION")
    print("="*70)
    
    store = create_memory_store()
    memory_manager = MemoryManager(store)
    
    # User 1: Alice
    memory_manager.save_user_preference("alice", "name", "Alice")
    memory_manager.save_user_preference("alice", "role", "Manager")
    memory_manager.save_memory("alice", "Prefers async communication")
    
    # User 2: Bob
    memory_manager.save_user_preference("bob", "name", "Bob")
    memory_manager.save_user_preference("bob", "role", "Developer")
    memory_manager.save_memory("bob", "Expert in Python and Go")
    
    # User 3: Carol
    memory_manager.save_user_preference("carol", "name", "Carol")
    memory_manager.save_user_preference("carol", "role", "Designer")
    memory_manager.save_memory("carol", "Focuses on accessibility")
    
    # Show isolation
    print("\n Alice's data:")
    print(f"  Preferences: {memory_manager.get_all_preferences('alice')}")
    print(f"  Memories: {[m['content'] for m in memory_manager.search_memories('alice')]}")
    
    print("\n Bob's data:")
    print(f"  Preferences: {memory_manager.get_all_preferences('bob')}")
    print(f"  Memories: {[m['content'] for m in memory_manager.search_memories('bob')]}")
    
    print("\n Carol's data:")
    print(f"  Preferences: {memory_manager.get_all_preferences('carol')}")
    print(f"  Memories: {[m['content'] for m in memory_manager.search_memories('carol')]}")
    
    # Verify isolation - Bob can't see Alice's data
    print("\n🔒 Isolation check:")
    alice_in_bob = memory_manager.get_user_preference("bob", "name")
    print(f"  Bob's namespace returns Alice's name? {alice_in_bob == 'Alice'}")
    print(f"  Bob's namespace returns Bob's name? {alice_in_bob == 'Bob'}")


# ============================================================
# DEMO: SEMANTIC SEARCH (requires embeddings)
# ============================================================

def demo_semantic_search():
    """Demo: Search memories by semantic meaning"""
    print("\n" + "="*70)
    print(" DEMO 4: SEMANTIC SEARCH")
    print("="*70)
    
    try:
        # Create store with embeddings
        store = create_memory_store(use_embeddings=True)
        
        user_id = "semantic-user"
        namespace = ("users", user_id, "memories")
        
        # Add diverse memories
        memories = [
            "User loves hiking in the mountains on weekends",
            "Prefers Python for backend development",
            "Has two cats named Whiskers and Shadow",
            "Recently started learning machine learning",
            "Enjoys reading science fiction novels",
            "Works remotely from Seattle",
            "Allergic to peanuts",
            "Morning person, starts work at 7am"
        ]
        
        for i, mem in enumerate(memories):
            store.put(namespace, f"mem-{i}", {"content": mem})
        
        print("\n Stored memories:")
        for mem in memories:
            print(f"  - {mem}")
        
        # Semantic searches
        queries = [
            "outdoor activities",
            "programming languages",
            "pets",
            "health considerations",
            "work schedule"
        ]
        
        print("\n Semantic Search Results:")
        for query in queries:
            print(f"\n  Query: '{query}'")
            results = store.search(namespace, query=query, limit=2)
            for r in results:
                print(f"    → {r.value['content']}")
                
    except Exception as e:
        print(f"\n Semantic search requires embeddings setup: {e}")
        print("   To enable, ensure OPENAI_API_KEY is set for embeddings.")


# ============================================================
# DEMO: MEMORY LIFECYCLE
# ============================================================

def demo_memory_lifecycle():
    """Demo: Create, update, delete memories"""
    print("\n" + "="*70)
    print("♻️ DEMO 5: MEMORY LIFECYCLE")
    print("="*70)
    
    store = create_memory_store()
    memory_manager = MemoryManager(store)
    
    user_id = "lifecycle-user"
    
    # CREATE
    print("\n1️⃣ CREATE memories:")
    mem_id_1 = memory_manager.save_memory(user_id, "User's favorite color is blue")
    mem_id_2 = memory_manager.save_memory(user_id, "User has 3 years experience")
    
    # READ
    print("\n2️⃣ READ memories:")
    memories = memory_manager.search_memories(user_id)
    for m in memories:
        print(f"  [{m['key'][:8]}...] {m['content']}")
    
    # UPDATE (put with same key overwrites)
    print("\n3️⃣ UPDATE memory:")
    namespace = ("users", user_id, "memories")
    store.put(namespace, mem_id_1, {
        "content": "User's favorite color is now green (changed)",
        "type": "general",
        "created_at": datetime.now().isoformat()
    })
    log("Updated memory: color changed from blue to green", "✏️")
    
    # Verify update
    memories = memory_manager.search_memories(user_id)
    for m in memories:
        print(f"  [{m['key'][:8]}...] {m['content']}")
    
    # DELETE
    print("\n4️⃣ DELETE memory:")
    memory_manager.delete_memory(user_id, mem_id_2)
    
    # Verify deletion
    memories = memory_manager.search_memories(user_id)
    print(f"  Remaining memories: {len(memories)}")
    for m in memories:
        print(f"  [{m['key'][:8]}...] {m['content']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        LangGraph Cross-Thread Long-Term Memory Demo           ║
╠══════════════════════════════════════════════════════════════════╣
║  Features demonstrated:                                          ║
║  • Store interface (put/get/search/delete)                       ║
║  • Cross-thread persistence                                      ║
║  • Namespace organization                                        ║
║  • Multi-user isolation                                          ║
║  • Semantic search (with embeddings)                             ║
║  • Memory lifecycle management                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_cross_thread_memory()
        demo_namespaces()
        demo_multi_user()
        demo_semantic_search()
        demo_memory_lifecycle()
        
        print("\n" + "="*70)
        print(" ALL DEMOS COMPLETE")
        print("="*70)
        print("""
KEY TAKEAWAYS:
1. InMemoryStore for development, PostgresStore for production
2. Namespaces organize memories: ("users", user_id, "memories")
3. Store is passed to compile() and accessible in nodes via `store` param
4. Checkpointer = short-term (thread), Store = long-term (cross-thread)
5. Add embeddings for semantic search capabilities

PRODUCTION STORES:
- langgraph-checkpoint-postgres: PostgresStore
- langgraph-checkpoint-mongodb: MongoDBStore
- langgraph-checkpoint-redis: RedisStore
        """)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
