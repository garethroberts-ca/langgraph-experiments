#!/bin/bash
# ============================================================
# Culture Amp HR Platform - Installation Script
# ============================================================
# This script installs CLI commands and shell functions for
# easy access to the HR Coaching Platform from anywhere.
#
# Usage: ./install.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_SCRIPT="$SCRIPT_DIR/24_ultimate_hr_platform.py"
BIN_DIR="$HOME/.local/bin"
SHELL_RC="$HOME/.zshrc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Culture Amp HR Platform - Installation                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if platform script exists
if [ ! -f "$PLATFORM_SCRIPT" ]; then
    echo -e "${RED}Error: Platform script not found at $PLATFORM_SCRIPT${NC}"
    exit 1
fi

# Create bin directory if it doesn't exist
mkdir -p "$BIN_DIR"

echo -e "${YELLOW}Installing commands to $BIN_DIR...${NC}"

# ============================================================
# Create main CLI wrapper
# ============================================================
cat > "$BIN_DIR/hr-coach" << EOF
#!/bin/bash
# Culture Amp HR Coaching Platform CLI
# Usage: hr-coach [options]

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"
# Trim OpenAI key if present (handles accidental whitespace)
if [ -n "\$OPENAI_API_KEY" ]; then
  export OPENAI_API_KEY="\$(echo "\$OPENAI_API_KEY" | tr -d ' \n\r')"
fi
python3 "$PLATFORM_SCRIPT" "\$@"
EOF
chmod +x "$BIN_DIR/hr-coach"
echo -e "${GREEN} Created: hr-coach${NC}"

# ============================================================
# Create interactive mode shortcut
# ============================================================
cat > "$BIN_DIR/hr-chat" << EOF
#!/bin/bash
# Start HR Coaching in interactive chat mode
# Usage: hr-chat

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"
# Trim OpenAI key if present (handles accidental whitespace)
if [ -n "\$OPENAI_API_KEY" ]; then
  export OPENAI_API_KEY="\$(echo "\$OPENAI_API_KEY" | tr -d ' \n\r')"
fi

# Prefer calling the interactive helper directly if available
python3 - <<PY
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
# Load the platform script into this interpreter
with open('${PLATFORM_SCRIPT}', 'r', encoding='utf-8') as f:
    code = f.read()
ns = {}
exec(code, ns)

run_interactive = ns.get('run_interactive_mode')
if callable(run_interactive):
    run_interactive()
else:
    print('Interactive mode helper not found; running default CLI...')
    main = ns.get('main') or ns.get('main_cli')
    if callable(main):
        main()
    else:
        print('No suitable entrypoint found in script.')
PY
EOF
chmod +x "$BIN_DIR/hr-chat"
echo -e "${GREEN} Created: hr-chat${NC}"

# ============================================================
# Create demo runner
# ============================================================
cat > "$BIN_DIR/hr-demo" << EOF
#!/bin/bash
# Run HR Platform demos
# Usage: hr-demo [demo_name]
# Examples:
#   hr-demo --list-demos           # List available demos
#   hr-demo conversation           # Run specific demo
#   hr-demo --all                  # Run all demos

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"
if [ "\$1" == "--list-demos" ] || [ "\$1" == "-l" ]; then
    python3 "$PLATFORM_SCRIPT" --list-demos
elif [ "\$1" == "--all" ] || [ "\$1" == "-a" ]; then
    python3 "$PLATFORM_SCRIPT" --all
elif [ -n "\$1" ]; then
    python3 "$PLATFORM_SCRIPT" --demo "\$1"
else
    echo "HR Platform Demos"
    echo "Usage: hr-demo [option|demo_name]"
    echo ""
    echo "Options:"
    echo "  --list-demos, -l    List available demos"
    echo "  --all, -a           Run all demos"
    echo "  <demo_name>         Run specific demo"
    echo ""
    python3 "$PLATFORM_SCRIPT" --list-demos
fi
EOF
chmod +x "$BIN_DIR/hr-demo"
echo -e "${GREEN} Created: hr-demo${NC}"

# ============================================================
# Create health check command
# ============================================================
cat > "$BIN_DIR/hr-health" << 'HEALTHEOF'
#!/bin/bash
# Run HR Platform health check
# Usage: hr-health

python3 -c "
import os, sys, json
from datetime import datetime

status = {
    'status': 'healthy',
    'version': '2.0.0',
    'timestamp': datetime.now().isoformat(),
    'checks': {}
}

# Check OpenAI API key
api_key = os.getenv('OPENAI_API_KEY', '')
status['checks']['openai_api_key'] = 'present' if api_key and len(api_key) > 20 else 'missing'

# Check imports
try:
    from langgraph.graph import StateGraph
    status['checks']['langgraph'] = 'available'
except ImportError:
    status['checks']['langgraph'] = 'missing'
    status['status'] = 'unhealthy'

try:
    from langchain_openai import ChatOpenAI
    status['checks']['langchain_openai'] = 'available'
except ImportError:
    status['checks']['langchain_openai'] = 'missing'
    status['status'] = 'unhealthy'

print(json.dumps(status, indent=2))
sys.exit(0 if status['status'] == 'healthy' else 1)
"
HEALTHEOF
chmod +x "$BIN_DIR/hr-health"
echo -e "${GREEN} Created: hr-health${NC}"

# ============================================================
# Create quick feedback improver
# ============================================================
cat > "$BIN_DIR/hr-feedback" << EOF
#!/bin/bash
# Quick feedback improvement using SBI model
# Usage: hr-feedback "Your feedback text here"
# Or pipe: echo "feedback" | hr-feedback

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"

if [ -n "\$1" ]; then
    INPUT="\$1"
elif [ ! -t 0 ]; then
    INPUT=\$(cat)
else
    echo "Usage: hr-feedback \"Your feedback text\""
    echo "   or: echo \"feedback\" | hr-feedback"
    exit 1
fi

python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from langchain_core.messages import HumanMessage

# Suppress banner
import os
os.environ['HR_PLATFORM_DEBUG'] = 'false'

exec(open('${PLATFORM_SCRIPT}').read())

platform = build_ultimate_platform()
state = create_default_state()
state['messages'] = [HumanMessage(content='Help me improve this feedback using SBI model: \$INPUT')]
state['request_type'] = RequestType.FEEDBACK_WRITING.value

result = platform.invoke(state, {'configurable': {'thread_id': 'quick-feedback'}})
print(result['messages'][-1].content)
"
EOF
chmod +x "$BIN_DIR/hr-feedback"
echo -e "${GREEN} Created: hr-feedback${NC}"

# ============================================================
# Create OKR helper
# ============================================================
cat > "$BIN_DIR/hr-goals" << EOF
#!/bin/bash
# Quick OKR/goal setting helper
# Usage: hr-goals "Your goal context"

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"

if [ -n "\$1" ]; then
    INPUT="\$1"
else
    echo "Usage: hr-goals \"Your goal context or role description\""
    exit 1
fi

python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from langchain_core.messages import HumanMessage

exec(open('${PLATFORM_SCRIPT}').read())

platform = build_ultimate_platform()
state = create_default_state()
state['messages'] = [HumanMessage(content='Help me create OKRs for: \$INPUT')]
state['request_type'] = RequestType.GOAL_SETTING.value

result = platform.invoke(state, {'configurable': {'thread_id': 'quick-goals'}})
print(result['messages'][-1].content)
"
EOF
chmod +x "$BIN_DIR/hr-goals"
echo -e "${GREEN} Created: hr-goals${NC}"

# ============================================================
# Create 1:1 agenda generator
# ============================================================
cat > "$BIN_DIR/hr-1on1" << EOF
#!/bin/bash
# Generate 1:1 meeting agenda
# Usage: hr-1on1 "Context about the meeting"

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"

if [ -n "\$1" ]; then
    INPUT="\$1"
else
    INPUT="General 1:1 meeting with my direct report"
fi

python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from langchain_core.messages import HumanMessage

exec(open('${PLATFORM_SCRIPT}').read())

platform = build_ultimate_platform()
state = create_default_state()
state['messages'] = [HumanMessage(content='Generate a 1:1 agenda for: \$INPUT')]
state['request_type'] = RequestType.ONE_ON_ONE.value

result = platform.invoke(state, {'configurable': {'thread_id': 'quick-1on1'}})
print(result['messages'][-1].content)
"
EOF
chmod +x "$BIN_DIR/hr-1on1"
echo -e "${GREEN} Created: hr-1on1${NC}"

# ============================================================
# Create shoutout generator
# ============================================================
cat > "$BIN_DIR/hr-shoutout" << EOF
#!/bin/bash
# Generate a recognition shoutout
# Usage: hr-shoutout "What the person did"

export PYTHONPATH="${SCRIPT_DIR}:\$PYTHONPATH"

if [ -n "\$1" ]; then
    INPUT="\$1"
else
    echo "Usage: hr-shoutout \"Description of what the person did\""
    exit 1
fi

python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from langchain_core.messages import HumanMessage

exec(open('${PLATFORM_SCRIPT}').read())

platform = build_ultimate_platform()
state = create_default_state()
state['messages'] = [HumanMessage(content='Create a shoutout for: \$INPUT')]
state['request_type'] = RequestType.SHOUTOUT.value

result = platform.invoke(state, {'configurable': {'thread_id': 'quick-shoutout'}})
print(result['messages'][-1].content)
"
EOF
chmod +x "$BIN_DIR/hr-shoutout"
echo -e "${GREEN} Created: hr-shoutout${NC}"

# ============================================================
# Add to PATH if not already there
# ============================================================
echo ""
echo -e "${YELLOW}Checking PATH configuration...${NC}"

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo -e "${YELLOW}Adding $BIN_DIR to PATH in $SHELL_RC...${NC}"
    
    # Add PATH export to shell config
    cat >> "$SHELL_RC" << EOF

# Culture Amp HR Platform - Added by install.sh
export PATH="\$HOME/.local/bin:\$PATH"

# HR Platform aliases
alias hrc='hr-coach'
alias hrh='hr-health'
alias hrd='hr-demo'
EOF
    
    echo -e "${GREEN} PATH updated in $SHELL_RC${NC}"
    echo -e "${YELLOW}Run 'source $SHELL_RC' or restart your terminal to apply.${NC}"
else
    echo -e "${GREEN} $BIN_DIR already in PATH${NC}"
fi

# ============================================================
# Print summary
# ============================================================
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Available commands:"
echo ""
echo -e "  ${GREEN}hr-coach${NC}     - Main CLI (hr-coach --help for options)"
echo -e "  ${GREEN}hr-chat${NC}      - Start interactive chat mode"
echo -e "  ${GREEN}hr-demo${NC}      - Run platform demos"
echo -e "  ${GREEN}hr-health${NC}    - Run health check"
echo -e "  ${GREEN}hr-feedback${NC}  - Quick feedback improvement"
echo -e "  ${GREEN}hr-goals${NC}     - Quick OKR generator"
echo -e "  ${GREEN}hr-1on1${NC}      - Generate 1:1 agenda"
echo -e "  ${GREEN}hr-shoutout${NC}  - Create recognition shoutout"
echo ""
echo "Aliases:"
echo "  hrc  -> hr-coach"
echo "  hrh  -> hr-health"
echo "  hrd  -> hr-demo"
echo ""
echo -e "${YELLOW}To apply changes now, run:${NC}"
echo "  source $SHELL_RC"
echo ""
echo -e "${YELLOW}Quick test:${NC}"
echo "  hr-health"
echo ""
