#!/usr/bin/env python3
"""
LangGraph Advanced Example 34: Retry and Error Handling
========================================================

This example demonstrates robust error handling and fault tolerance in LangGraph:
- Built-in retry policies with RetryPolicy
- Custom error handling within nodes
- Graceful degradation patterns
- Circuit breaker patterns
- Error recovery and fallback strategies

Building resilient agent systems requires handling failures gracefully,
especially when dealing with external services and LLM APIs.

Author: LangGraph Examples
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
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
    print(f"{prefix}→ {message}")


# =============================================================================
# Demo 1: Basic Retry Policy
# =============================================================================

def demo_basic_retry():
    """
    Basic retry policy for handling transient failures.
    
    Demonstrates:
    - Configuring RetryPolicy
    - Automatic retries on failure
    - Exponential backoff
    """
    print_banner("Demo 1: Basic Retry Policy")
    
    # Track call attempts
    call_count = {"value": 0}
    
    class RetryState(TypedDict):
        input: str
        output: str
    
    def flaky_node(state: RetryState) -> dict:
        """Node that fails sometimes (simulating network issues)."""
        call_count["value"] += 1
        attempt = call_count["value"]
        
        log(f"Attempt {attempt}: Processing...", indent=1)
        
        # Fail on first 2 attempts
        if attempt < 3:
            log(f"Attempt {attempt}: Simulating failure!", indent=1)
            raise ConnectionError(f"Transient failure on attempt {attempt}")
        
        log(f"Attempt {attempt}: Success!", indent=1)
        return {"output": f"Processed: {state['input']}"}
    
    # Configure retry policy
    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=0.1,  # Start with 100ms
        backoff_factor=2.0,    # Double each time
        max_interval=2.0,      # Max 2 seconds
        retry_on=(ConnectionError,)  # Retry on these exceptions
    )
    
    # Build graph with retry policy
    builder = StateGraph(RetryState)
    builder.add_node("process", flaky_node, retry=retry_policy)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    # Reset counter
    call_count["value"] = 0
    
    result = graph.invoke({"input": "test data", "output": ""})
    
    log(f"\nFinal result: {result['output']}")
    log(f"Total attempts: {call_count['value']}")


# =============================================================================
# Demo 2: Custom Error Handling in Nodes
# =============================================================================

def demo_custom_error_handling():
    """
    Handle errors within node logic for fine-grained control.
    
    Demonstrates:
    - Try-catch within nodes
    - Error state tracking
    - Custom error messages
    """
    print_banner("Demo 2: Custom Error Handling in Nodes")
    
    class ErrorHandlingState(TypedDict):
        data: str
        result: str
        error: str | None
        success: bool
    
    def risky_operation(state: ErrorHandlingState) -> dict:
        """Perform operation with error handling."""
        try:
            log("Attempting risky operation...")
            
            # Simulate various conditions
            if "invalid" in state["data"].lower():
                raise ValueError("Invalid data format")
            
            if "timeout" in state["data"].lower():
                raise TimeoutError("Operation timed out")
            
            result = f"Processed: {state['data'].upper()}"
            log(f"Success: {result}", indent=1)
            return {"result": result, "success": True, "error": None}
            
        except ValueError as e:
            log(f"Validation error: {e}", indent=1)
            return {"result": "", "success": False, "error": f"Validation: {e}"}
            
        except TimeoutError as e:
            log(f"Timeout error: {e}", indent=1)
            return {"result": "", "success": False, "error": f"Timeout: {e}"}
            
        except Exception as e:
            log(f"Unexpected error: {e}", indent=1)
            return {"result": "", "success": False, "error": f"Unknown: {e}"}
    
    def handle_result(state: ErrorHandlingState) -> dict:
        """Process based on success/failure."""
        if state["success"]:
            log("Result handler: Operation succeeded")
        else:
            log(f"Result handler: Operation failed - {state['error']}")
        return {}
    
    # Build graph
    builder = StateGraph(ErrorHandlingState)
    builder.add_node("risky", risky_operation)
    builder.add_node("handler", handle_result)
    builder.add_edge(START, "risky")
    builder.add_edge("risky", "handler")
    builder.add_edge("handler", END)
    
    graph = builder.compile()
    
    # Test various scenarios
    test_cases = ["good data", "invalid input", "timeout scenario"]
    
    for test in test_cases:
        log(f"\nTesting: '{test}'")
        result = graph.invoke({
            "data": test,
            "result": "",
            "error": None,
            "success": False
        })
        log(f"Final state: success={result['success']}, error={result['error']}", indent=1)


# =============================================================================
# Demo 3: Fallback Pattern
# =============================================================================

def demo_fallback_pattern():
    """
    Implement fallback nodes when primary operations fail.
    
    Demonstrates:
    - Primary/fallback node pattern
    - Conditional routing based on errors
    - Graceful degradation
    """
    print_banner("Demo 3: Fallback Pattern")
    
    class FallbackState(TypedDict):
        query: str
        response: str
        used_fallback: bool
        attempts: list[str]
    
    def primary_service(state: FallbackState) -> dict:
        """Primary service - might fail."""
        log("Trying primary service...")
        
        # Simulate 70% failure rate
        if random.random() < 0.7:
            log("Primary service failed!", indent=1)
            return {
                "response": "",
                "attempts": state["attempts"] + ["primary_failed"]
            }
        
        log("Primary service succeeded!", indent=1)
        return {
            "response": f"[PRIMARY] Response to: {state['query']}",
            "used_fallback": False,
            "attempts": state["attempts"] + ["primary_success"]
        }
    
    def route_after_primary(state: FallbackState) -> str:
        """Route to fallback if primary failed."""
        if state["response"]:
            return END
        return "fallback"
    
    def fallback_service(state: FallbackState) -> dict:
        """Fallback service - more reliable but slower."""
        log("Using fallback service...")
        time.sleep(0.1)  # Simulate slower fallback
        
        return {
            "response": f"[FALLBACK] Response to: {state['query']}",
            "used_fallback": True,
            "attempts": state["attempts"] + ["fallback_success"]
        }
    
    # Build graph
    builder = StateGraph(FallbackState)
    builder.add_node("primary", primary_service)
    builder.add_node("fallback", fallback_service)
    builder.add_edge(START, "primary")
    builder.add_conditional_edges("primary", route_after_primary)
    builder.add_edge("fallback", END)
    
    graph = builder.compile()
    
    # Run multiple times to see fallback behavior
    random.seed(42)  # For reproducibility
    
    for i in range(5):
        log(f"\n--- Run {i+1} ---")
        result = graph.invoke({
            "query": f"Query #{i+1}",
            "response": "",
            "used_fallback": False,
            "attempts": []
        })
        log(f"Response: {result['response'][:50]}...")
        log(f"Used fallback: {result['used_fallback']}")
        log(f"Attempts: {result['attempts']}")


# =============================================================================
# Demo 4: Circuit Breaker Pattern
# =============================================================================

def demo_circuit_breaker():
    """
    Implement circuit breaker to prevent cascading failures.
    
    Demonstrates:
    - Circuit breaker states (closed, open, half-open)
    - Failure threshold tracking
    - Automatic recovery
    """
    print_banner("Demo 4: Circuit Breaker Pattern")
    
    # Circuit breaker state (would typically be persistent/shared)
    circuit_breaker = {
        "failures": 0,
        "threshold": 3,
        "state": "closed",  # closed, open, half-open
        "last_failure_time": 0,
        "recovery_timeout": 2.0
    }
    
    class CircuitState(TypedDict):
        request: str
        response: str
        circuit_state: str
        was_blocked: bool
    
    def check_circuit(state: CircuitState) -> dict:
        """Check if circuit allows request."""
        current_time = time.time()
        
        # Check if we should try half-open
        if circuit_breaker["state"] == "open":
            if current_time - circuit_breaker["last_failure_time"] > circuit_breaker["recovery_timeout"]:
                circuit_breaker["state"] = "half-open"
                log("Circuit: Moving to half-open state", indent=1)
        
        return {"circuit_state": circuit_breaker["state"]}
    
    def route_by_circuit(state: CircuitState) -> str:
        """Route based on circuit state."""
        if state["circuit_state"] == "open":
            return "blocked"
        return "service"
    
    def call_service(state: CircuitState) -> dict:
        """Call the external service."""
        log(f"Calling service (circuit: {circuit_breaker['state']})...")
        
        # Simulate 60% failure rate
        if random.random() < 0.6:
            circuit_breaker["failures"] += 1
            circuit_breaker["last_failure_time"] = time.time()
            
            log(f"Service failed! (failures: {circuit_breaker['failures']})", indent=1)
            
            if circuit_breaker["failures"] >= circuit_breaker["threshold"]:
                circuit_breaker["state"] = "open"
                log("Circuit: OPENED due to failures!", indent=1)
            
            return {
                "response": "[ERROR] Service unavailable",
                "circuit_state": circuit_breaker["state"],
                "was_blocked": False
            }
        
        # Success - reset if half-open
        if circuit_breaker["state"] == "half-open":
            circuit_breaker["state"] = "closed"
            circuit_breaker["failures"] = 0
            log("Circuit: CLOSED after successful call", indent=1)
        
        return {
            "response": f"[SUCCESS] Processed: {state['request']}",
            "circuit_state": circuit_breaker["state"],
            "was_blocked": False
        }
    
    def blocked_response(state: CircuitState) -> dict:
        """Handle blocked request."""
        log("Request blocked by circuit breaker!", indent=1)
        return {
            "response": "[BLOCKED] Circuit open - service unavailable",
            "was_blocked": True
        }
    
    # Build graph
    builder = StateGraph(CircuitState)
    builder.add_node("check", check_circuit)
    builder.add_node("service", call_service)
    builder.add_node("blocked", blocked_response)
    builder.add_edge(START, "check")
    builder.add_conditional_edges("check", route_by_circuit)
    builder.add_edge("service", END)
    builder.add_edge("blocked", END)
    
    graph = builder.compile()
    
    # Run multiple requests to see circuit breaker behavior
    random.seed(123)
    
    for i in range(10):
        log(f"\n--- Request {i+1} ---")
        result = graph.invoke({
            "request": f"Request #{i+1}",
            "response": "",
            "circuit_state": "",
            "was_blocked": False
        })
        log(f"Response: {result['response']}")
        log(f"Circuit state: {result['circuit_state']}, Blocked: {result['was_blocked']}")
        
        # Small delay between requests
        time.sleep(0.5)


# =============================================================================
# Demo 5: Graceful Degradation
# =============================================================================

def demo_graceful_degradation():
    """
    Degrade service quality gracefully when resources are limited.
    
    Demonstrates:
    - Multiple quality levels
    - Resource-aware processing
    - Graceful reduction in capabilities
    """
    print_banner("Demo 5: Graceful Degradation")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class DegradationState(TypedDict):
        query: str
        response: str
        quality_level: str  # "full", "reduced", "minimal"
        resource_available: bool
    
    def check_resources(state: DegradationState) -> dict:
        """Check available resources and determine quality level."""
        # Simulate resource checking (would check real metrics in production)
        available = random.random() > 0.3  # 70% chance resources available
        
        if available:
            quality = "full"
        else:
            # Degrade based on load
            quality = random.choice(["reduced", "minimal"])
        
        log(f"Resources available: {available}, Quality: {quality}")
        return {"resource_available": available, "quality_level": quality}
    
    def route_by_quality(state: DegradationState) -> str:
        """Route to appropriate handler based on quality level."""
        return state["quality_level"]
    
    def full_quality_response(state: DegradationState) -> dict:
        """Full quality LLM response."""
        log("Processing at FULL quality", indent=1)
        response = llm.invoke(
            f"Provide a detailed answer to: {state['query']}"
        )
        return {"response": f"[FULL] {response.content}"}
    
    def reduced_quality_response(state: DegradationState) -> dict:
        """Reduced quality - shorter response."""
        log("Processing at REDUCED quality", indent=1)
        response = llm.invoke(
            f"Briefly answer in 1-2 sentences: {state['query']}"
        )
        return {"response": f"[REDUCED] {response.content}"}
    
    def minimal_quality_response(state: DegradationState) -> dict:
        """Minimal quality - cached/static response."""
        log("Processing at MINIMAL quality", indent=1)
        return {
            "response": f"[MINIMAL] Your query '{state['query']}' has been received. "
                       "Due to high demand, please try again later for a detailed response."
        }
    
    # Build graph
    builder = StateGraph(DegradationState)
    builder.add_node("check", check_resources)
    builder.add_node("full", full_quality_response)
    builder.add_node("reduced", reduced_quality_response)
    builder.add_node("minimal", minimal_quality_response)
    builder.add_edge(START, "check")
    builder.add_conditional_edges("check", route_by_quality)
    builder.add_edge("full", END)
    builder.add_edge("reduced", END)
    builder.add_edge("minimal", END)
    
    graph = builder.compile()
    
    # Test multiple times
    random.seed(42)
    
    for i in range(3):
        log(f"\n--- Query {i+1} ---")
        result = graph.invoke({
            "query": "What is machine learning?",
            "response": "",
            "quality_level": "",
            "resource_available": True
        })
        log(f"Quality: {result['quality_level']}")
        log(f"Response: {result['response'][:100]}...")


# =============================================================================
# Demo 6: Error Recovery with Checkpoints
# =============================================================================

def demo_error_recovery():
    """
    Use checkpointing to recover from errors mid-workflow.
    
    Demonstrates:
    - Saving progress at checkpoints
    - Resuming from last successful step
    - Error recovery strategies
    """
    print_banner("Demo 6: Error Recovery with Checkpoints")
    
    # Simulate checkpoint storage
    checkpoints = {}
    
    class RecoveryState(TypedDict):
        task_id: str
        steps_completed: list[str]
        current_step: int
        result: str
        failed: bool
    
    def step_one(state: RecoveryState) -> dict:
        """First step - always succeeds."""
        log("Step 1: Initializing...")
        # Save checkpoint
        checkpoints[state["task_id"]] = {
            "step": 1,
            "data": "step1_data"
        }
        return {"steps_completed": ["step1"], "current_step": 1}
    
    def step_two(state: RecoveryState) -> dict:
        """Second step - might fail."""
        log("Step 2: Processing...")
        
        # Check if resuming from checkpoint
        checkpoint = checkpoints.get(state["task_id"], {})
        if checkpoint.get("step", 0) >= 2:
            log("  Resuming from checkpoint!", indent=1)
        
        # Simulate occasional failure
        if random.random() < 0.4:
            log("  Step 2 FAILED!", indent=1)
            return {"failed": True}
        
        # Save checkpoint
        checkpoints[state["task_id"]] = {
            "step": 2,
            "data": "step2_data"
        }
        return {
            "steps_completed": state["steps_completed"] + ["step2"],
            "current_step": 2
        }
    
    def should_continue(state: RecoveryState) -> str:
        """Check if we should continue or handle error."""
        if state.get("failed"):
            return "error_handler"
        return "step_three"
    
    def step_three(state: RecoveryState) -> dict:
        """Third step - finalize."""
        log("Step 3: Finalizing...")
        return {
            "steps_completed": state["steps_completed"] + ["step3"],
            "current_step": 3,
            "result": "Task completed successfully!"
        }
    
    def error_handler(state: RecoveryState) -> dict:
        """Handle errors and attempt recovery."""
        log("Error handler: Attempting recovery...")
        
        checkpoint = checkpoints.get(state["task_id"], {})
        log(f"  Last checkpoint: step {checkpoint.get('step', 0)}", indent=1)
        
        return {
            "result": f"Failed at step {state['current_step'] + 1}. "
                     f"Recovery checkpoint at step {checkpoint.get('step', 0)}."
        }
    
    # Build graph
    builder = StateGraph(RecoveryState)
    builder.add_node("step1", step_one)
    builder.add_node("step2", step_two)
    builder.add_node("step3", step_three)
    builder.add_node("error_handler", error_handler)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_conditional_edges("step2", should_continue)
    builder.add_edge("step3", END)
    builder.add_edge("error_handler", END)
    
    graph = builder.compile()
    
    # Run multiple tasks
    random.seed(42)
    
    for i in range(4):
        task_id = f"task_{i+1}"
        log(f"\n--- {task_id} ---")
        
        result = graph.invoke({
            "task_id": task_id,
            "steps_completed": [],
            "current_step": 0,
            "result": "",
            "failed": False
        })
        
        log(f"Steps completed: {result['steps_completed']}")
        log(f"Result: {result['result']}")


# =============================================================================
# Demo 7: Timeout Handling
# =============================================================================

async def demo_timeout_handling():
    """
    Handle long-running operations with timeouts.
    
    Demonstrates:
    - Async timeout handling
    - Cancellation of slow operations
    - Timeout fallbacks
    """
    print_banner("Demo 7: Timeout Handling")
    
    class TimeoutState(TypedDict):
        query: str
        response: str
        timed_out: bool
    
    async def slow_operation(state: TimeoutState) -> dict:
        """Potentially slow operation."""
        delay = random.uniform(0.5, 2.5)  # Random delay 0.5-2.5 seconds
        log(f"Starting operation (will take {delay:.1f}s)...")
        
        try:
            # Use asyncio.timeout for Python 3.11+ or asyncio.wait_for
            await asyncio.sleep(delay)
            log(f"Operation completed in {delay:.1f}s", indent=1)
            return {
                "response": f"Processed: {state['query']}",
                "timed_out": False
            }
        except asyncio.CancelledError:
            log("Operation was cancelled!", indent=1)
            raise
    
    async def run_with_timeout(state: TimeoutState) -> dict:
        """Wrapper that enforces timeout."""
        timeout = 1.5  # 1.5 second timeout
        
        try:
            result = await asyncio.wait_for(
                slow_operation(state),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            log(f"Operation timed out after {timeout}s!", indent=1)
            return {
                "response": f"[TIMEOUT] Query '{state['query']}' took too long",
                "timed_out": True
            }
    
    # Build graph
    builder = StateGraph(TimeoutState)
    builder.add_node("process", run_with_timeout)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    graph = builder.compile()
    
    # Run multiple times
    random.seed(42)
    
    for i in range(5):
        log(f"\n--- Query {i+1} ---")
        result = await graph.ainvoke({
            "query": f"Query #{i+1}",
            "response": "",
            "timed_out": False
        })
        log(f"Response: {result['response']}")
        log(f"Timed out: {result['timed_out']}")


# =============================================================================
# Demo 8: Validation and Error Prevention
# =============================================================================

def demo_validation():
    """
    Prevent errors through input validation.
    
    Demonstrates:
    - Pre-validation of inputs
    - Type checking
    - Constraint validation
    """
    print_banner("Demo 8: Validation and Error Prevention")
    
    class ValidationState(TypedDict):
        input_data: dict
        validation_errors: list[str]
        is_valid: bool
        processed_result: str
    
    def validate_input(state: ValidationState) -> dict:
        """Validate input data before processing."""
        errors = []
        data = state["input_data"]
        
        log("Validating input...")
        
        # Required fields
        required = ["name", "age", "email"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Type validation
        if "age" in data:
            if not isinstance(data["age"], int):
                errors.append("Age must be an integer")
            elif data["age"] < 0 or data["age"] > 150:
                errors.append("Age must be between 0 and 150")
        
        # Format validation
        if "email" in data:
            if "@" not in str(data["email"]):
                errors.append("Invalid email format")
        
        # Length validation
        if "name" in data:
            if len(str(data["name"])) < 2:
                errors.append("Name must be at least 2 characters")
        
        is_valid = len(errors) == 0
        log(f"Validation result: {'PASSED' if is_valid else 'FAILED'}", indent=1)
        if errors:
            for error in errors:
                log(f"  - {error}", indent=2)
        
        return {"validation_errors": errors, "is_valid": is_valid}
    
    def route_after_validation(state: ValidationState) -> str:
        """Route based on validation result."""
        if state["is_valid"]:
            return "process"
        return "reject"
    
    def process_data(state: ValidationState) -> dict:
        """Process valid data."""
        log("Processing valid data...")
        data = state["input_data"]
        return {
            "processed_result": f"Welcome, {data['name']}! Account created for {data['email']}."
        }
    
    def reject_data(state: ValidationState) -> dict:
        """Handle invalid data."""
        log("Rejecting invalid data...")
        return {
            "processed_result": f"Validation failed: {', '.join(state['validation_errors'])}"
        }
    
    # Build graph
    builder = StateGraph(ValidationState)
    builder.add_node("validate", validate_input)
    builder.add_node("process", process_data)
    builder.add_node("reject", reject_data)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validation)
    builder.add_edge("process", END)
    builder.add_edge("reject", END)
    
    graph = builder.compile()
    
    # Test cases
    test_cases = [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "B", "age": -5, "email": "invalid"},
        {"age": 25, "email": "test@test.com"},  # Missing name
        {"name": "Valid Name", "age": "not_a_number", "email": "valid@email.com"},
    ]
    
    for i, test_data in enumerate(test_cases):
        log(f"\n--- Test Case {i+1}: {test_data} ---")
        result = graph.invoke({
            "input_data": test_data,
            "validation_errors": [],
            "is_valid": False,
            "processed_result": ""
        })
        log(f"Result: {result['processed_result']}")


# =============================================================================
# Demo 9: Multi-Service Error Handling
# =============================================================================

def demo_multi_service_errors():
    """
    Handle errors across multiple dependent services.
    
    Demonstrates:
    - Coordinating multiple service calls
    - Partial failure handling
    - Aggregating errors across services
    """
    print_banner("Demo 9: Multi-Service Error Handling")
    
    class MultiServiceState(TypedDict):
        request: str
        service_a_result: str | None
        service_b_result: str | None
        service_c_result: str | None
        errors: list[str]
        partial_success: bool
    
    def call_service_a(state: MultiServiceState) -> dict:
        """Call Service A."""
        log("Calling Service A...")
        
        if random.random() < 0.3:
            log("  Service A failed!", indent=1)
            return {
                "service_a_result": None,
                "errors": state["errors"] + ["Service A unavailable"]
            }
        
        log("  Service A succeeded", indent=1)
        return {"service_a_result": "Result from A"}
    
    def call_service_b(state: MultiServiceState) -> dict:
        """Call Service B."""
        log("Calling Service B...")
        
        if random.random() < 0.3:
            log("  Service B failed!", indent=1)
            return {
                "service_b_result": None,
                "errors": state["errors"] + ["Service B unavailable"]
            }
        
        log("  Service B succeeded", indent=1)
        return {"service_b_result": "Result from B"}
    
    def call_service_c(state: MultiServiceState) -> dict:
        """Call Service C."""
        log("Calling Service C...")
        
        if random.random() < 0.3:
            log("  Service C failed!", indent=1)
            return {
                "service_c_result": None,
                "errors": state["errors"] + ["Service C unavailable"]
            }
        
        log("  Service C succeeded", indent=1)
        return {"service_c_result": "Result from C"}
    
    def aggregate_results(state: MultiServiceState) -> dict:
        """Aggregate results from all services."""
        log("Aggregating results...")
        
        results = [
            state["service_a_result"],
            state["service_b_result"],
            state["service_c_result"]
        ]
        
        successful = [r for r in results if r is not None]
        partial_success = len(successful) > 0 and len(successful) < 3
        
        log(f"  Successful services: {len(successful)}/3", indent=1)
        
        return {"partial_success": partial_success}
    
    # Build graph with parallel service calls
    builder = StateGraph(MultiServiceState)
    builder.add_node("service_a", call_service_a)
    builder.add_node("service_b", call_service_b)
    builder.add_node("service_c", call_service_c)
    builder.add_node("aggregate", aggregate_results)
    
    # Parallel service calls
    builder.add_edge(START, "service_a")
    builder.add_edge(START, "service_b")
    builder.add_edge(START, "service_c")
    builder.add_edge("service_a", "aggregate")
    builder.add_edge("service_b", "aggregate")
    builder.add_edge("service_c", "aggregate")
    builder.add_edge("aggregate", END)
    
    graph = builder.compile()
    
    # Run multiple times
    random.seed(42)
    
    for i in range(4):
        log(f"\n--- Request {i+1} ---")
        result = graph.invoke({
            "request": f"Request #{i+1}",
            "service_a_result": None,
            "service_b_result": None,
            "service_c_result": None,
            "errors": [],
            "partial_success": False
        })
        
        results = [result["service_a_result"], result["service_b_result"], result["service_c_result"]]
        successful = len([r for r in results if r is not None])
        
        log(f"Services succeeded: {successful}/3")
        log(f"Errors: {result['errors']}")
        log(f"Partial success: {result['partial_success']}")


# =============================================================================
# Demo 10: Comprehensive Error Handling Strategy
# =============================================================================

def demo_comprehensive_error_handling():
    """
    Comprehensive error handling with multiple strategies combined.
    
    Demonstrates:
    - Layered error handling
    - Combining retries, fallbacks, and validation
    - Production-ready error management
    """
    print_banner("Demo 10: Comprehensive Error Handling Strategy")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    class ComprehensiveState(TypedDict):
        user_input: str
        validated_input: str
        llm_response: str
        final_response: str
        errors: list[str]
        strategy_used: str
    
    def validate_and_sanitize(state: ComprehensiveState) -> dict:
        """Validate and sanitize user input."""
        log("Step 1: Validating input...")
        
        user_input = state["user_input"].strip()
        
        # Basic validation
        if len(user_input) < 3:
            return {
                "errors": ["Input too short"],
                "strategy_used": "validation_failed"
            }
        
        if len(user_input) > 1000:
            user_input = user_input[:1000]
            log("  Input truncated to 1000 chars", indent=1)
        
        # Sanitize
        sanitized = user_input.replace("<", "&lt;").replace(">", "&gt;")
        
        log("  Validation passed", indent=1)
        return {"validated_input": sanitized}
    
    def route_after_validation(state: ComprehensiveState) -> str:
        if state.get("errors"):
            return "error_response"
        return "primary_llm"
    
    def call_primary_llm(state: ComprehensiveState) -> dict:
        """Primary LLM call with error handling."""
        log("Step 2: Calling primary LLM...")
        
        try:
            response = llm.invoke(
                f"Respond helpfully to: {state['validated_input']}"
            )
            log("  LLM call succeeded", indent=1)
            return {
                "llm_response": response.content,
                "strategy_used": "primary_llm"
            }
        except Exception as e:
            log(f"  LLM call failed: {e}", indent=1)
            return {
                "errors": state["errors"] + [f"LLM error: {str(e)}"],
                "strategy_used": "primary_failed"
            }
    
    def route_after_llm(state: ComprehensiveState) -> str:
        if state.get("llm_response"):
            return "finalize"
        return "fallback_response"
    
    def fallback_response(state: ComprehensiveState) -> dict:
        """Provide fallback response when LLM fails."""
        log("Step 3: Using fallback response...")
        
        return {
            "llm_response": f"I received your message about '{state['validated_input'][:50]}...'. "
                           "I'm currently experiencing issues but have logged your request.",
            "strategy_used": "fallback"
        }
    
    def error_response(state: ComprehensiveState) -> dict:
        """Handle validation errors."""
        log("Generating error response...")
        return {
            "final_response": f"Unable to process request: {', '.join(state['errors'])}",
            "strategy_used": "error"
        }
    
    def finalize_response(state: ComprehensiveState) -> dict:
        """Finalize and format response."""
        log("Step 4: Finalizing response...")
        
        response = state["llm_response"]
        
        # Post-processing
        if len(response) > 500:
            response = response[:500] + "..."
        
        return {
            "final_response": response
        }
    
    # Build graph
    builder = StateGraph(ComprehensiveState)
    builder.add_node("validate", validate_and_sanitize)
    builder.add_node("primary_llm", call_primary_llm)
    builder.add_node("fallback_response", fallback_response)
    builder.add_node("error_response", error_response)
    builder.add_node("finalize", finalize_response)
    
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validation)
    builder.add_conditional_edges("primary_llm", route_after_llm)
    builder.add_edge("fallback_response", "finalize")
    builder.add_edge("error_response", END)
    builder.add_edge("finalize", END)
    
    graph = builder.compile()
    
    # Test cases
    test_cases = [
        "What is the capital of France?",
        "Hi",  # Too short
        "Tell me about artificial intelligence and its applications in modern society",
    ]
    
    for i, test in enumerate(test_cases):
        log(f"\n--- Test {i+1}: '{test[:30]}...' ---")
        result = graph.invoke({
            "user_input": test,
            "validated_input": "",
            "llm_response": "",
            "final_response": "",
            "errors": [],
            "strategy_used": ""
        })
        log(f"Strategy used: {result['strategy_used']}")
        log(f"Response: {result['final_response'][:100]}...")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all error handling demonstrations."""
    print("\n" + "🛡️" * 35)
    print(" LangGraph Retry & Error Handling ".center(70))
    print("🛡️" * 35)
    
    # Run synchronous demos
    demo_basic_retry()
    demo_custom_error_handling()
    demo_fallback_pattern()
    demo_circuit_breaker()
    demo_graceful_degradation()
    demo_error_recovery()
    demo_validation()
    demo_multi_service_errors()
    demo_comprehensive_error_handling()
    
    # Run async demo
    asyncio.run(demo_timeout_handling())
    
    print("\n" + "=" * 70)
    print(" All Error Handling Demonstrations Complete! ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
