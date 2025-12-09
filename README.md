# LangGraph HR Coaching Examples

A comprehensive collection of 38 LangGraph demonstrations showcasing patterns for building intelligent HR coaching and organisational psychology systems. Each example builds on previous concepts, progressing from basic patterns to production-ready architectures.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Examples](#running-examples)
- [Troubleshooting](#troubleshooting)
- [Overview](#overview)
- [File Reference](#file-reference)

---

## Quick Start

Get up and running in 60 seconds:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/langgraph-hr-coaching.git
cd langgraph-hr-coaching

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"  # On Windows: set OPENAI_API_KEY=sk-your-key-here

# 5. Run your first example
python 01_basic_coach.py
```

---

## Prerequisites

### Required

- **Python 3.10+** - Check with `python3 --version`
- **OpenAI API Key** - Get one at [platform.openai.com](https://platform.openai.com/api-keys)

### Recommended

- **Git** - For cloning the repository
- **Virtual Environment** - Keeps dependencies isolated

---

## Installation

### Step 1: Clone or Download

```bash
# Option A: Clone with git
git clone https://github.com/YOUR_USERNAME/langgraph-hr-coaching.git
cd langgraph-hr-coaching

# Option B: Download ZIP from GitHub and extract
```

### Step 2: Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

**Full installation (recommended):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Minimal installation (basic examples only):**
```bash
pip install langgraph langchain-openai langchain-core
```

### Step 4: Configure OpenAI API Key

**macOS / Linux:**
```bash
# Set for current session
export OPENAI_API_KEY="sk-your-key-here"

# Or add to shell profile for persistence
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.zshrc  # or ~/.bashrc
source ~/.zshrc
```

**Windows (PowerShell):**
```powershell
# Set for current session
$env:OPENAI_API_KEY = "sk-your-key-here"

# Or set permanently via System Properties > Environment Variables
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

### Step 5: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check packages installed
pip show langgraph langchain-openai

# Verify API key is set
echo $OPENAI_API_KEY  # Should show your key (or part of it)
```

---

## Running Examples

### Interactive CLI Examples

Most examples run as interactive command-line programmes:

```bash
# Basic coaching bot
python 01_basic_coach.py

# Multi-agent system with specialist routing
python 02_multi_agent_coach.py

# Memory-enabled longitudinal coaching
python 03_memory_coach.py

# Safety guardrails and risk classification
python 04_safety_coach.py
```

Type your messages and press Enter. Type `quit` to exit.

### Streamlit Web UI

Launch the web interface:

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` with:
- Basic Coach mode
- Reflective Coach mode (self-improving responses)

### Running Specific Demos

Some files contain multiple demos. Run them to see all patterns:

```bash
# All 7 Anthropic agentic patterns
python 21_anthropic_patterns.py

# LangGraph time travel debugging
python 25_time_travel_debug.py

# All streaming modes demonstrated
python 32_streaming_modes.py

# Comprehensive organisational psychology assessment
python 38_organisational_psychology_engine.py
```

### Example Categories

| Examples | Category | What You'll Learn |
|----------|----------|-------------------|
| 01-06 | Fundamentals | State, routing, memory, tools, subgraphs |
| 07-10 | Production | Human-in-loop, parallel, async, observability |
| 11-18 | Multi-Agent | Collaboration, reflection, orchestration |
| 19-24 | Enterprise | Full platforms, all patterns combined |
| 25-34 | Deep Dives | Time travel, Send API, streaming, errors |
| 35-38 | Domain | Culture Amp, EVOLVE model, psychology |

---

## Troubleshooting

### "No module named 'langgraph'"

Ensure you've activated your virtual environment:

```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Then reinstall
pip install -r requirements.txt
```

### "Invalid API Key" or "AuthenticationError"

Check your OpenAI API key:

```bash
# Verify it's set
echo $OPENAI_API_KEY

# Should start with "sk-" and be ~50 characters
```

If not set, see [Configure OpenAI API Key](#step-4-configure-openai-api-key) above.

### "Rate limit exceeded"

OpenAI has usage limits. Solutions:
- Wait a few minutes and retry
- Check your usage at [platform.openai.com/usage](https://platform.openai.com/usage)
- Use a different API key with higher limits

### "Python version 3.9 not supported"

This project requires Python 3.10+. Check your version:

```bash
python3 --version
```

Install a newer Python version from [python.org](https://www.python.org/downloads/).

### Streamlit not opening browser

Manually navigate to `http://localhost:8501` or try:

```bash
streamlit run app.py --server.headless true
```

### AsyncIO errors on Windows

Add this at the start of async examples:

```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

## Overview

This repository demonstrates LangGraph capabilities through HR coaching use cases, covering:

- **Core LangGraph Patterns** (Files 01-10): Fundamentals of graph-based agents
- **Advanced Multi-Agent Systems** (Files 11-18): Collaboration, orchestration, and risk handling
- **Enterprise Patterns** (Files 19-24): Production-ready architectures and optimisation
- **LangGraph Features Deep Dives** (Files 25-34): Time travel, streaming, Send API, and more
- **Domain Applications** (Files 35-38): Culture Amp integrations and psychological frameworks

---

## File Reference

### Core LangGraph Patterns (01-10)

#### 01 - Basic Coach
**File:** `01_basic_coach.py`

The foundational example demonstrating the simplest LangGraph pattern.

**Key Concepts:**
- `StateGraph` creation and compilation
- State schema with `TypedDict` and `Annotated` message lists
- Single node graphs with `START` and `END` edges
- System prompt integration with OpenAI

**Features:**
- Interactive coaching loop
- Configurable coaching focus (career, performance, wellbeing)
- Message history accumulation

---

#### 02 - Multi-Agent Coach
**File:** `02_multi_agent_coach.py`

Introduces routing and specialist agents.

**Key Concepts:**
- Router pattern for intent classification
- Conditional edges with `add_conditional_edges`
- Multiple specialist nodes (goal_setting, feedback, reflection, general)
- Dynamic routing based on LLM classification

**Features:**
- Automatic intent detection from user messages
- Goal context injection into specialist prompts
- Extensible specialist agent definitions

---

#### 03 - Memory Coach
**File:** `03_memory_coach.py`

Demonstrates checkpointing and memory persistence.

**Key Concepts:**
- `MemorySaver` checkpointer for state persistence
- Thread-based session management
- Structured coaching memory (goals, action items, insights, preferences)
- Memory extraction and update nodes

**Features:**
- Cross-session memory persistence
- Automatic insight extraction from conversations
- User preference learning

---

#### 04 - Safety Coach
**File:** `04_safety_coach.py`

Implements safety guardrails and risk classification.

**Key Concepts:**
- Multi-level risk classification (LOW, MEDIUM, HIGH, CRITICAL)
- Rule-based and LLM-based risk detection
- Safety response routing
- Human escalation workflows

**Features:**
- Keyword-based risk flagging (mental health, harassment, legal)
- Appropriate safety responses for high-risk situations
- EAP and HR escalation pathways

---

#### 05 - Tool-Enabled Coach
**File:** `05_tool_enabled_coach.py`

Full agentic loop with tool calling.

**Key Concepts:**
- `@tool` decorator for function tools
- `llm.bind_tools()` for tool-enabled LLMs
- `ToolNode` for automatic tool execution
- Conditional routing between coach and tools

**Tools Implemented:**
- `get_user_goals`: Retrieve current goals
- `get_recent_feedback`: Access peer/manager feedback
- `search_policies`: RAG over company policies
- `create_action_item`: Generate action items
- `suggest_learning_resource`: Recommend training
- `schedule_followup`: Book coaching sessions

---

#### 06 - Subgraph Coach
**File:** `06_subgraph_coach.py`

Modular agent composition with nested graphs.

**Key Concepts:**
- Building and compiling subgraphs
- Adding compiled graphs as nodes
- Domain-specific subgraphs (career, performance, wellbeing)
- Tool sets per subgraph

**Subgraphs:**
- **Career Development**: Career paths, promotion readiness, skill requirements
- **Performance**: Performance data, 360 feedback, development goals
- **Wellbeing**: EAP resources, stress assessment, work-life balance

---

#### 07 - Human-in-the-Loop Coach
**File:** `07_human_in_loop_coach.py`

Approval workflows and interrupt handling.

**Key Concepts:**
- `interrupt()` for pausing execution
- Pending action tracking
- Approval-required tools detection
- Multi-level approval workflows

**Approval-Required Actions:**
- Performance Improvement Plans (manager approval)
- Promotion nominations (calibration committee)
- Compensation discussions (manager + HR)
- Sensitive skip-level meetings (HR review)

---

#### 08 - Parallel Agents Coach
**File:** `08_parallel_agents_coach.py`

Fan-out/fan-in pattern with parallel execution.

**Key Concepts:**
- `ThreadPoolExecutor` for parallel LLM calls
- `operator.add` reducer for result aggregation
- Specialist analyser pattern
- Confidence-weighted synthesis

**Parallel Specialists:**
- Skills Analyst
- Career Strategist
- Relationship Advisor
- Risk Assessor

---

#### 09 - Streaming Async Coach
**File:** `09_streaming_async_coach.py`

Asynchronous execution with streaming support.

**Key Concepts:**
- `async/await` patterns in nodes
- `AsyncSqliteSaver` for async checkpointing
- `ainvoke` for async LLM calls
- Session metadata tracking

**Features:**
- Async tools for non-blocking I/O
- Topic extraction and session summarisation
- SQLite persistence

---

#### 10 - Production Coach
**File:** `10_production_coach.py`

Production-grade patterns with observability.

**Key Concepts:**
- Supervisor agent pattern
- Metrics collection and logging
- Retry with exponential backoff
- Graceful degradation

**Features:**
- `MetricsCollector` for observability
- `ObservabilityCallback` for LLM tracing
- `@with_retry` decorator
- Quality scoring and human review flags

---

### Advanced Multi-Agent Systems (11-18)

#### 11 - Collaborative Agents
**File:** `11_collaborative_agents.py`

Agent-to-agent communication and shared workspaces.

**Key Concepts:**
- Agent messaging system with message types (REQUEST, RESPONSE, NEGOTIATE)
- Shared blackboard pattern
- Round-based conversation orchestration
- Consensus building

**Agent Roles:**
- HR Strategist (organisational design, talent strategy)
- Career Advisor (career pathing, skill development)
- Org Psychologist (behaviour, motivation, dynamics)
- Data Analyst (metrics, benchmarks, evidence)

---

#### 12 - Hierarchical Platform
**File:** `12_hierarchical_platform.py`

Graph-as-tool and multi-tenant patterns.

**Key Concepts:**
- `GraphRegistry` for dynamic graph invocation
- Multi-tenant configuration
- Hierarchical graph composition (leaf → mid → top)
- Tenant-specific customisation

**Features:**
- Skill assessment, goal generation, feedback analysis graphs
- Enterprise vs basic tier configurations
- Rate limiting per tenant

---

#### 13 - Reflective Agent
**File:** `13_reflective_agent.py`

Self-evaluation and iterative refinement.

**Key Concepts:**
- Multi-dimensional critique (accuracy, relevance, actionability, empathy, safety)
- Weighted quality scoring
- Reflection and self-improvement loops
- Confidence calibration

**Quality Dimensions:**
- Accuracy (25%)
- Relevance (20%)
- Actionability (20%)
- Empathy (15%)
- Completeness (10%)
- Safety (10%)

Also includes `13_reflective_fast.py` - a lightweight version with 3x fewer LLM calls.

---

#### 14 - Goal Lifecycle Coach
**File:** `14_goal_lifecycle_coach.py`

OKR-style goal management.

**Key Concepts:**
- Goal decomposition into key results
- Progress tracking with evidence
- Status lifecycle (DRAFT → ACTIVE → COMPLETED)
- Conflict detection between goals

**Features:**
- SMART goal framework
- Key Result progress calculation
- Action item generation
- Timeline and priority management

---

#### 15 - Feedback Synthesis
**File:** `15_feedback_synthesis.py`

Multi-source feedback aggregation.

**Key Concepts:**
- Feedback source categorisation (manager, peer, direct report, self, skip-level)
- Theme extraction and pattern detection
- Contradiction identification
- Evidence-anchored recommendations

**Analysis Stages:**
- Load and structure feedback
- Extract recurring themes
- Identify strengths and growth areas
- Detect contradictions
- Generate coaching questions

---

#### 16 - Memory Coach
**File:** `16_memory_coach.py`

Sophisticated three-layer memory system.

**Key Concepts:**
- **Episodic Memory**: Session summaries and key moments
- **Semantic Memory**: Stable inferences about the person
- **Procedural Memory**: How to work effectively with this person
- Memory retrieval and relevance scoring

**Features:**
- Memory update lifecycle
- Confidence-scored inferences
- Custom coaching instructions per user
- Commitment tracking

---

#### 17 - Risk Triage
**File:** `17_risk_triage.py`

Multi-level risk classification with escalation.

**Key Concepts:**
- Risk categories (mental health, harassment, discrimination, legal, safety)
- Confidence-calibrated decisions
- Escalation contact routing
- Audit logging for compliance

**Risk Levels:**
- CRITICAL: Immediate human intervention
- HIGH: Specialist routing, may need escalation
- MODERATE: Handle with care, document
- LOW: Standard coaching with awareness
- MINIMAL: Normal interaction

---

#### 18 - Full Orchestrator
**File:** `18_full_orchestrator.py`

Production-style supervisor with specialist routing.

**Key Concepts:**
- Intent analysis for routing
- Specialist agent delegation
- Shared context and memory
- Quality assurance checks

**Specialists:**
- Goal Coach
- Feedback Coach
- Career Coach
- Wellbeing Coach
- Conflict Coach
- General Coach
- Safety Handler

---

### Enterprise Patterns (19-24)

#### 19 - Enterprise Platform
**File:** `19_enterprise_platform.py`

Near-production reference implementation.

**Key Concepts:**
- SQLite database with full schema
- RAG over policies and frameworks
- Comprehensive audit logging
- Pending approval workflow

**Database Tables:**
- users, sessions, memories, goals
- actions, audit_log, knowledge_base, pending_approvals

---

#### 20 - Adaptive Development Coach
**File:** `20_adaptive_dev_coach.py`

Plan-and-execute with adaptive user modelling.

**Key Concepts:**
- Dynamic user model that evolves
- Multi-phase development journeys (90-day plans)
- Commitment tracking with follow-ups
- Strategy adaptation based on outcomes

**User Model Attributes:**
- Communication/learning/decision styles
- Motivation, confidence, stress levels
- Observed strengths and challenges
- Successful vs ineffective strategies

---

#### 21 - Anthropic Patterns
**File:** `21_anthropic_patterns.py`

All 7 patterns from Anthropic's "Building Effective Agents" blog.

**Patterns Implemented:**
1. **Augmented LLM**: Retrieval + tools + memory
2. **Prompt Chaining**: Sequential steps building on previous outputs
3. **Routing**: Classify and direct to specialised handlers
4. **Parallelisation**: Sectioning and voting patterns
5. **Orchestrator-Workers**: Dynamic delegation
6. **Evaluator-Optimizer**: Generate → evaluate → improve loop
7. **Autonomous Agent**: Think → act → observe cycle

---

#### 22 - Optimised Parallel
**File:** `22_optimized_parallel.py`

Advanced parallelisation for 360 feedback analysis.

**Key Concepts:**
- True async execution with `asyncio`
- Dynamic fan-out based on runtime data
- Weighted voting for aggregation
- Result caching with hash keys
- Graceful degradation on timeouts

**Features:**
- `AnalysisCache` with thread-safe access
- `ProgressTracker` for real-time monitoring
- Contradiction detection and resolution
- Configurable retry policies

---

#### 23 - Parallel YAML Processor
**File:** `23_parallel_yaml_processor.py`

Ultimate HR platform combining all patterns.

**Key Concepts:**
- All Anthropic agentic patterns combined
- Culture Amp Perform features integration
- Bulk YAML data processing
- Comprehensive knowledge base

---

#### 24 - Ultimate HR Platform
**File:** `24_ultimate_hr_platform.py`

The most advanced implementation combining everything.

**Features:**
- All Anthropic patterns
- Culture Amp Perform integration (self-reflections, reviews, OKRs, 1-on-1s, feedback, shoutouts)
- LangGraph advanced features (Send API, subgraphs, checkpointing, human-in-the-loop)
- Full optimisation (caching, weighted aggregation, graceful degradation)

---

### LangGraph Features Deep Dives (25-34)

#### 25 - Time Travel Debug
**File:** `25_time_travel_debug.py`

Checkpoint history and state replay.

**Key Concepts:**
- `get_state_history()` for checkpoint access
- Replay from any checkpoint
- Branch execution to explore alternatives
- State editing and forking

**Use Cases:**
- Debug agent decisions
- Explore "what if" scenarios
- Recover from errors without restart
- A/B test different paths

---

#### 26 - Cross-Thread Memory
**File:** `26_cross_thread_memory.py`

LangGraph Store interface for persistent memory.

**Key Concepts:**
- `InMemoryStore` with namespacing
- put/get/search/delete operations
- Cross-thread memory access
- Optional semantic search with embeddings

**Memory Operations:**
- User preferences (key-value)
- Episodic memories (timestamped)
- Namespace organisation (users/org/context)

---

#### 27 - Prebuilt ReAct Agent
**File:** `27_prebuilt_react_agent.py`

Rapid agent development with prebuilt components.

**Key Concepts:**
- `create_react_agent()` one-liner
- Custom system prompts
- `ToolNode` and `tools_condition`
- State modifiers for pre/post processing

**Tools Demonstrated:**
- Weather lookup
- HR policy search
- Meeting scheduling
- PTO balance calculation

---

#### 28 - Dynamic Breakpoints
**File:** `28_dynamic_breakpoints.py`

Human-in-the-loop interrupt patterns.

**Key Concepts:**
- `interrupt_before`: Pause before a node
- `interrupt_after`: Pause after a node
- `interrupt()`: Dynamic interrupt from within
- `Command` for resumption with user input

**Use Cases:**
- Human approval before sensitive actions
- User input collection mid-workflow
- Review and edit agent outputs
- Multi-step approval workflows

---

#### 29 - Command Routing
**File:** `29_command_routing.py`

Dynamic control flow with Command objects.

**Key Concepts:**
- `Command(goto=...)`: Navigate to specific nodes
- `Command(update=...)`: Modify state from within
- Combined goto + update operations
- Error recovery and retry logic

**Patterns:**
- Dynamic router without explicit edges
- Retry with counter and max attempts
- Tiered escalation workflows

---

#### 30 - Functional API
**File:** `30_functional_api.py`

Alternative to StateGraph with functional style.

**Key Concepts:**
- `@entrypoint` decorator for workflows
- `@task` decorator for individual functions
- `.result()` for awaiting task completion
- `entrypoint.final()` for return values

**Advantages:**
- More Pythonic feel
- Simpler mental model
- Automatic state persistence
- Easy parallel task execution

---

#### 31 - Map-Reduce Send
**File:** `31_map_reduce_send.py`

Dynamic fan-out with the Send API.

**Key Concepts:**
- `Send()` for creating parallel instances
- `operator.add` reducers for aggregation
- Variable-length parallel processing
- Multi-stage map-reduce pipelines

**Demos:**
- Basic item processing
- Parallel LLM calls for multiple perspectives
- Chained map-reduce stages
- Document analysis with variable sections

---

#### 32 - Streaming Modes
**File:** `32_streaming_modes.py`

Comprehensive streaming capabilities.

**Key Concepts:**
- `stream_mode="values"`: Full state after each step
- `stream_mode="updates"`: Delta changes only
- `stream_mode="messages"`: Token-by-token LLM output
- `get_stream_writer()`: Custom streaming

**Use Cases:**
- Real-time UI updates
- Progress indicators
- Token streaming for chat interfaces
- Debug observation

---

#### 33 - Subgraph Composition
**File:** `33_subgraph_composition.py`

Modular nested graph design.

**Key Concepts:**
- Shared state schemas between parent and child
- Different schemas with state transformation
- Multi-agent systems using subgraphs
- Streaming from nested graphs

**Patterns:**
- Parent-child with same schema
- Schema transformation wrappers
- Three-level nesting (grandparent → parent → child)

---

#### 34 - Retry Error Handling
**File:** `34_retry_error_handling.py`

Robust fault tolerance patterns.

**Key Concepts:**
- `RetryPolicy` configuration
- Exponential backoff
- Custom error handling within nodes
- Circuit breaker patterns

**Strategies:**
- Automatic retry on transient failures
- Graceful degradation with fallbacks
- Error state tracking
- Selective retry based on exception type

---

### Domain Applications (35-38)

#### 35 - Culture Amp Employee Experience
**File:** `35_culture_amp_employee_experience.py`

Employee experience workflows inspired by Culture Amp.

**Demos (7):**
1. **Engagement Survey Analysis**: Quantitative scores, sentiment analysis, benchmark comparison
2. **Retention Risk Identification**: Flight risk scoring, intervention recommendations
3. **DEI Insights**: Representation analysis, equity metrics, inclusion indices
4. **Action Plan Generation**: Priority actions, quick wins, strategic initiatives
5. **Manager Coaching**: Leadership recommendations, team health insights
6. **Pulse Survey Orchestration**: Trend analysis, alert generation
7. **Comment Summarisation**: Theme extraction, sentiment clustering

---

#### 36 - File Operations
**File:** `36_file_operations.py`

Intelligent file system operations.

**Demos (8):**
1. **Batch Processing**: Discovery, validation, transformation
2. **Content Classification**: AI-powered file categorisation
3. **Duplicate Detection**: Hash-based deduplication
4. **Intelligent Organisation**: Auto-folder creation and sorting
5. **Directory Sync**: Source-to-target synchronisation
6. **Validation Pipeline**: Schema and content validation
7. **Backup Workflows**: Incremental backup with verification
8. **Real File Operations**: Actual filesystem manipulation

---

#### 37 - EVOLVE Coaching Model
**File:** `37_evolve_coaching_model.py`

Culture Amp's EVOLVE coaching framework.

**EVOLVE Stages:**
- **E**xplore: Understand current situation and context
- **V**ision: Define desired future state and aspirations
- **O**ptions: Generate possible paths and strategies
- **L**earn: Identify skills, knowledge, and resources needed
- **V**alidate: Check alignment and commitment
- **E**xecute: Create actionable plans and next steps

**Demos (4):**
1. Full EVOLVE coaching session
2. Stage-specific deep dives
3. AI-assisted insight generation
4. Session summarisation

---

#### 38 - Organisational Psychology Engine
**File:** `38_organisational_psychology_engine.py`

Comprehensive psychological assessment system.

**Frameworks Integrated:**
1. **JD-R Model** (Bakker & Demerouti): Job demands vs resources, burnout prediction
2. **Psychological Safety** (Edmondson): Team learning, voice climate, interpersonal risk
3. **Self-Determination Theory** (Deci & Ryan): Autonomy, competence, relatedness
4. **Organisational Justice**: Distributive, procedural, interactional fairness
5. **Maslach Burnout Inventory**: Exhaustion, cynicism, professional efficacy
6. **Team Development** (Tuckman + Lencioni): Stages and dysfunctions

**Pipeline Stages (10):**
1. Survey data collection and preprocessing
2. JD-R model analysis
3. Psychological safety assessment
4. Self-determination theory evaluation
5. Organisational justice measurement
6. Burnout inventory scoring
7. Team dynamics analysis
8. Integrated risk factor identification
9. AI-powered diagnosis and insights
10. Intervention planning and executive summary

---

## Architecture Patterns

### State Management
```python
class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-merging message list
    results: Annotated[list, operator.add]   # Reducer for aggregation
    user_id: str
```

### Conditional Routing
```python
graph.add_conditional_edges(
    "router",
    route_function,
    {"option_a": "node_a", "option_b": "node_b"}
)
```

### Tool Integration
```python
@tool
def my_tool(param: str) -> str:
    """Tool description for the LLM."""
    return result

llm_with_tools = llm.bind_tools([my_tool])
```

### Checkpointing
```python
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user-123"}}
```

---

## Requirements

- Python 3.10+
- langgraph >= 0.2.0
- langchain-openai >= 0.2.0
- langchain-core >= 0.3.0
- streamlit (for the UI app)

Optional:
- aiosqlite (for async SQLite)
- pyyaml (for YAML processing demos)

---

## Environment Variables

- `OPENAI_API_KEY` — required to use `ChatOpenAI`
  - macOS/Linux: `export OPENAI_API_KEY=sk-...`
  - Windows (PowerShell): `$Env:OPENAI_API_KEY = "sk-..."`

---

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install langgraph langchain-openai langchain-core streamlit
   ```

3. Set your OpenAI key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

---

## Running the Examples

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI scripts:**
```bash
python 01_basic_coach.py
python 24_ultimate_hr_platform.py
python 38_organisational_psychology_engine.py
```

---

## Contributing

Each file is self-contained and can be run independently. When adding new examples:

1. Follow the numbered naming convention
2. Include docstrings explaining key concepts
3. Use the `print_banner()` and `log()` helpers for consistent output
4. Demonstrate multiple patterns per file where sensible
5. Use British spelling throughout

---

## Troubleshooting

- **Import errors** like "No module named 'langgraph'" usually mean dependencies are not installed:
  ```bash
  which python && python -V
  pip show langgraph langchain-openai langchain-core streamlit
  ```

- **API errors** usually indicate a missing/invalid `OPENAI_API_KEY`

---

## Licence

MIT

---

## Acknowledgements

- LangGraph team for the excellent framework
- Culture Amp for HR domain inspiration
- Anthropic for agentic pattern documentation
