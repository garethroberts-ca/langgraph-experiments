"""
HR Coaching Assistant - Culture Amp Style UI
Run with: streamlit run app.py
"""

import streamlit as st
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HR Coach | Culture Amp",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CLEAN CULTURE AMP STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* FORCE LIGHT MODE */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
}

[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background-color: #FAFAFA !important;
    color: #1A1A1A !important;
}

/* Force all text to be dark */
p, span, label, div, h1, h2, h3, h4, h5, h6 {
    color: #1A1A1A !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #1A1A1A !important;
}

/* Reset and base */
*, *::before, *::after { box-sizing: border-box; }

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 800px;
    background-color: #FFFFFF !important;
}

/* Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FAFAFA;
    border-right: 1px solid #EAEAEA;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Logo area */
.logo-container {
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #EAEAEA;
    margin-bottom: 1.5rem;
}

.logo-text {
    font-size: 1.25rem;
    font-weight: 700;
    color: #E5533D;
    letter-spacing: -0.02em;
}

/* Section headers in sidebar */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 1.5rem 0 0.75rem 0;
}

/* Main header */
.main-header {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1A1A1A;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.main-subheader {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}

/* Chat container */
.chat-container {
    background: #FAFAFA;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    min-height: 400px;
    border: 1px solid #EAEAEA;
}

/* Messages */
.message-row {
    display: flex;
    margin-bottom: 1rem;
}

.message-row.user {
    justify-content: flex-end;
}

.message-row.assistant {
    justify-content: flex-start;
}

.message-bubble {
    max-width: 85%;
    padding: 0.875rem 1.125rem;
    border-radius: 16px;
    font-size: 0.9375rem;
    line-height: 1.5;
}

.message-bubble.user {
    background: #E5533D !important;
    color: white !important;
    border-bottom-right-radius: 4px;
}

.message-bubble.user * {
    color: white !important;
}

.message-bubble.assistant {
    background: white;
    color: #1A1A1A;
    border: 1px solid #EAEAEA;
    border-bottom-left-radius: 4px;
}

/* Avatar */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
    flex-shrink: 0;
}

.avatar.user {
    background: #E5533D !important;
    color: white !important;
    margin-left: 0.75rem;
}

.avatar.assistant {
    background: #6B5B95 !important;
    color: white !important;
    margin-right: 0.75rem;
}

/* Input area */
.input-container {
    background: white;
    border: 1px solid #EAEAEA;
    border-radius: 12px;
    padding: 0.25rem;
    display: flex;
    gap: 0.5rem;
}

/* Streamlit input overrides */
.stTextInput > div > div > input {
    border: 1px solid #EAEAEA !important;
    background: #FFFFFF !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important;
    color: #1A1A1A !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input:focus {
    box-shadow: none !important;
    border-color: #E5533D !important;
}

.stTextInput > div > div > input::placeholder {
    color: #888 !important;
}

/* Button overrides */
.stButton > button {
    background: #E5533D !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: background 0.15s ease !important;
}

.stButton > button:hover {
    background: #D14432 !important;
    color: white !important;
}

.stButton > button:active {
    background: #C13A2A !important;
}

/* Scenario buttons */
.scenario-btn {
    background: white;
    border: 1px solid #EAEAEA;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #1A1A1A;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
    width: 100%;
    margin-bottom: 0.5rem;
}

.scenario-btn:hover {
    border-color: #E5533D;
    background: #FEF7F6;
}

/* Radio buttons */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio > div > label {
    background: white;
    border: 1px solid #EAEAEA;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    cursor: pointer;
    transition: all 0.15s ease;
}

.stRadio > div > label:hover {
    border-color: #E5533D;
}

.stRadio > div > label[data-checked="true"] {
    border-color: #E5533D;
    background: #FEF7F6;
}

/* Select box */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* Info box */
.stAlert {
    background: #F0EFFA;
    border: none;
    border-radius: 8px;
    color: #6B5B95;
}

/* Metrics */
.metric-container {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    background: #F5F5F5;
    padding: 0.375rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8125rem;
    margin-right: 0.5rem;
    margin-top: 0.5rem;
}

.metric-label {
    color: #888;
}

.metric-value {
    color: #1A1A1A;
    font-weight: 600;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #E5533D !important;
}

/* Divider */
.divider {
    height: 1px;
    background: #EAEAEA;
    margin: 1.5rem 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #888;
}

.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.empty-state-text {
    font-size: 0.9375rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# COACHING GRAPHS
# ============================================================

class CoachingState(TypedDict):
    messages: Annotated[list, add_messages]
    coaching_style: str


def build_basic_coach(style: str = "supportive"):
    STYLE_PROMPTS = {
        "supportive": """You are a warm, empathetic HR coach. Focus on validating feelings, asking reflective questions, and empowering the employee. Be encouraging and supportive. Keep responses concise but helpful.""",
        "directive": """You are a direct, action-oriented HR coach. Focus on clear advice, specific steps, and measurable outcomes. Be concise and practical.""",
        "socratic": """You are a Socratic HR coach. Ask thought-provoking questions to help employees discover insights. Help them think through problems rather than giving direct answers."""
    }
    
    def coach(state: CoachingState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        style = state.get("coaching_style", "supportive")
        system = STYLE_PROMPTS.get(style, STYLE_PROMPTS["supportive"])
        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    graph = StateGraph(CoachingState)
    graph.add_node("coach", coach)
    graph.add_edge(START, "coach")
    graph.add_edge("coach", END)
    return graph.compile()


class ReflectiveState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    critique: str
    score: float
    iteration: int


def build_reflective_coach():
    def generate(state: ReflectiveState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        query = state["messages"][-1].content
        response = llm.invoke([HumanMessage(content=f"You are an HR coach. Respond helpfully and concisely to: {query}")])
        return {"draft": response.content, "iteration": 1}
    
    def critique(state: ReflectiveState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = f"""Rate this HR coaching response 0-1 and give brief critique.
Query: {state["messages"][-1].content}
Response: {state["draft"]}
JSON only: {{"score": 0.8, "critique": "brief issues"}}"""
        result = llm.invoke([HumanMessage(content=prompt)])
        try:
            parsed = json.loads(result.content)
            return {"score": parsed.get("score", 0.7), "critique": parsed.get("critique", "")}
        except:
            return {"score": 0.7, "critique": ""}
    
    def refine(state: ReflectiveState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
        prompt = f"""Improve this response based on critique. Keep it concise.
Original: {state["draft"]}
Critique: {state["critique"]}
Provide only the improved response."""
        result = llm.invoke([HumanMessage(content=prompt)])
        return {"draft": result.content, "iteration": state["iteration"] + 1}
    
    def should_refine(state: ReflectiveState) -> Literal["refine", "done"]:
        return "done" if state["score"] >= 0.8 or state["iteration"] >= 2 else "refine"
    
    def finalize(state: ReflectiveState) -> dict:
        return {"messages": [AIMessage(content=state["draft"])]}
    
    graph = StateGraph(ReflectiveState)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("refine", refine)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_refine, {"refine": "refine", "done": "finalize"})
    graph.add_edge("refine", "critique")
    graph.add_edge("finalize", END)
    return graph.compile()


# ============================================================
# APP
# ============================================================

def render_message(role: str, content: str, metadata: dict = None):
    """Render a chat message."""
    if role == "user":
        st.markdown(f'''
        <div class="message-row user">
            <div class="message-bubble user">{content}</div>
            <div class="avatar user">You</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="message-row assistant">
            <div class="avatar assistant">CA</div>
            <div class="message-bubble assistant">{content}</div>
        </div>
        ''', unsafe_allow_html=True)
        if metadata and (metadata.get("score") or metadata.get("iterations")):
            metrics_html = '<div style="margin-left: 44px;">'
            if metadata.get("score"):
                metrics_html += f'<span class="metric-container"><span class="metric-label">Quality</span><span class="metric-value">{metadata["score"]:.0%}</span></span>'
            if metadata.get("iterations"):
                metrics_html += f'<span class="metric-container"><span class="metric-label">Iterations</span><span class="metric-value">{metadata["iterations"]}</span></span>'
            metrics_html += '</div>'
            st.markdown(metrics_html, unsafe_allow_html=True)


def main():
    # Initialize state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="logo-container"><span class="logo-text">🧡 Culture Amp</span></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">Coaching Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode", ["Basic Coach", "Reflective Coach"], label_visibility="collapsed")
        
        if mode == "Basic Coach":
            st.markdown('<div class="sidebar-section">Style</div>', unsafe_allow_html=True)
            style = st.selectbox("Style", ["supportive", "directive", "socratic"], 
                                format_func=str.title, label_visibility="collapsed")
        else:
            style = "supportive"
            st.info(" Auto-critiques and refines responses")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Quick Scenarios</div>', unsafe_allow_html=True)
        
        scenarios = [
            ("Career Growth", "I've been in my role for 3 years and feel stuck. How do I grow?"),
            ("Feedback", "My manager doesn't give me feedback. How do I ask for it?"),
            ("Burnout", "I'm burning out from overwork. What should I do?"),
            ("Promotion", "I was passed over for promotion. How should I handle this?"),
            ("Conflict", "Two people on my team aren't getting along. How can I help?")
        ]
        
        for name, prompt in scenarios:
            if st.button(name, use_container_width=True, key=f"scenario_{name}"):
                st.session_state.pending_input = prompt
                st.rerun()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.markdown('<h1 class="main-header">HR Coaching Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subheader">Get guidance for workplace challenges</p>', unsafe_allow_html=True)
    
    # Chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        st.markdown('''
        <div class="empty-state">
            <div class="empty-state-icon"></div>
            <div class="empty-state-text">Start a conversation or pick a scenario from the sidebar</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"], msg.get("metadata"))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input
    pending = st.session_state.pop("pending_input", "")
    
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("Message", value=pending, placeholder="Type your question...", 
                                   label_visibility="collapsed", key="chat_input")
    with col2:
        send = st.button("Send", use_container_width=True)
    
    # Process
    if send and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            try:
                if mode == "Basic Coach":
                    graph = build_basic_coach(style)
                    result = graph.invoke({"messages": [HumanMessage(content=user_input)], "coaching_style": style})
                    response, metadata = result["messages"][-1].content, {}
                else:
                    graph = build_reflective_coach()
                    result = graph.invoke({"messages": [HumanMessage(content=user_input)], "draft": "", "critique": "", "score": 0, "iteration": 0})
                    response = result["messages"][-1].content
                    metadata = {"score": result.get("score", 0), "iterations": result.get("iteration", 1)}
                
                st.session_state.messages.append({"role": "assistant", "content": response, "metadata": metadata})
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.rerun()


if __name__ == "__main__":
    main()
