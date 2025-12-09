# Contributing

Thank you for your interest in contributing to LangGraph HR Coaching Examples!

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Check existing issues before creating a new one
- Include clear reproduction steps for bugs

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Ensure your code follows the project conventions (see below)
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

## Code Conventions

### File Naming

- Demo files follow the pattern: `{number}_{descriptive_name}.py`
- Use snake_case for file names
- Number new demos sequentially

### Spelling

- Use **British English** throughout:
  - analyse (not analyze)
  - organise (not organize)
  - behaviour (not behavior)
  - colour (not color) — unless in CSS/code syntax
  - favourite (not favorite)
  - specialise (not specialize)

### Style

- **No emoticons** in code or documentation
- Use `print_banner()` and `log()` helpers for console output
- Include comprehensive docstrings explaining key concepts
- Each file should be self-contained and runnable independently

### Code Structure

Each demo file should include:

1. Module docstring explaining what it demonstrates
2. Helper utilities section
3. State definitions
4. Node functions
5. Graph construction
6. Demo/main execution

Example structure:

```python
#!/usr/bin/env python3
"""
LangGraph Example {N}: {Title}
===============================

This example demonstrates {description}.

Key concepts:
- Concept 1
- Concept 2
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

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


# =============================================================================
# State Definition
# =============================================================================

class DemoState(TypedDict):
    ...


# =============================================================================
# Demo
# =============================================================================

def demo_function():
    ...


if __name__ == "__main__":
    demo_function()
```

## Testing

- Test your changes locally before submitting
- Ensure all examples run without errors
- Verify with: `python {your_file}.py`

## Questions?

Feel free to open an issue for any questions about contributing.
