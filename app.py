"""
Phishing Detector - Streamlit Interface
========================================
A modern, visually appealing web interface for the Phishing Detection Orchestrator.

Run with: streamlit run app.py
"""

import streamlit as st
import time
from typing import Optional

# Must be first Streamlit command
st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .verdict-safe {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    
    .verdict-phishing {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    }
    
    /* Model result badges */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        margin: 0.3rem;
    }
    
    .badge-ml { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; }
    .badge-dl { background: linear-gradient(135deg, #8b5cf6, #6d28d9); color: white; }
    .badge-genai { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    
    /* Confidence meter */
    .confidence-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
        transition: width 0.5s ease;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 0.8rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .analyzing {
        animation: pulse 1.5s infinite;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None


@st.cache_resource
def load_orchestrator():
    """Load the PhishingOrchestrator (cached)."""
    try:
        from phishing_detector import PhishingOrchestrator
        return PhishingOrchestrator()
    except Exception as e:
        st.error(f"Failed to load orchestrator: {e}")
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Phishing Detector</h1>
        <p>AI-powered URL analysis using ML, Deep Learning & GenAI ensemble</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with info and stats."""
    with st.sidebar:
        st.markdown("## 🔍 How It Works")
        st.markdown("""
        This system uses **3 AI models** working together:
        
        1. **🤖 ML Model** (XGBoost)
           - Analyzes URL features
           - Pattern-based detection
        
        2. **🧠 DL Model** (LSTM)
           - Character-level analysis
           - Sequence patterns
        
        3. **✨ GenAI** (Groq LLM)
           - Semantic understanding
           - Context-aware analysis
        
        ---
        
        **Weighted Scoring**: Final verdict based on weighted confidence scores (GenAI: 60%, DL: 20%, ML: 20%).
        """)
        
        st.markdown("---")
        st.markdown("## 📊 Session Stats")
        
        total = len(st.session_state.history)
        phishing = sum(1 for h in st.session_state.history if h["verdict"] == "phishing")
        legitimate = total - phishing
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">URLs Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #ef4444;">{phishing}</div>
                <div class="stat-label">Phishing Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧹 Clear History")
        if st.button("Clear All", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_result(result, url: str):
    """Render the classification result."""
    is_phishing = result.final_verdict.lower() == "phishing"
    verdict_class = "verdict-phishing" if is_phishing else "verdict-safe"
    verdict_icon = "🚨" if is_phishing else "✅"
    verdict_text = "PHISHING DETECTED" if is_phishing else "LEGITIMATE URL"
    
    st.markdown(f"""
    <div class="result-card">
        <div class="{verdict_class}">
            {verdict_icon} {verdict_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence meter
    confidence_pct = result.confidence * 100
    st.markdown(f"""
    <div class="confidence-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: rgba(255,255,255,0.7);">Confidence</span>
            <span style="font-weight: 600; color: white;">{confidence_pct:.1f}%</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); border-radius: 6px; overflow: hidden;">
            <div class="confidence-bar" style="width: {confidence_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model breakdown
    st.markdown("### 🎯 Model Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ml_pred = result.ml_result.prediction if result.ml_result else "N/A"
        ml_conf = f"{result.ml_result.confidence:.1%}" if result.ml_result else "N/A"
        ml_icon = "✅" if ml_pred == "legitimate" else "🚨" if ml_pred == "phishing" else ""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">🤖 ML (XGBoost)</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">{ml_icon} {ml_pred.upper()}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Confidence: {ml_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        dl_pred = result.dl_result.prediction if result.dl_result else "N/A"
        dl_conf = f"{result.dl_result.confidence:.1%}" if result.dl_result else "N/A"
        dl_icon = "✅" if dl_pred == "legitimate" else "🚨" if dl_pred == "phishing" else ""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #8b5cf6, #6d28d9); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">🧠 DL (LSTM)</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">{dl_icon} {dl_pred.upper()}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Confidence: {dl_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        genai_pred = result.genai_result.prediction if result.genai_result else "N/A"
        genai_conf = f"{result.genai_result.confidence:.1%}" if result.genai_result else "N/A"
        genai_icon = "✅" if genai_pred == "legitimate" else "🚨" if genai_pred == "phishing" else ""
        failover_note = " (Failover)" if result.failover_used else ""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f59e0b, #d97706); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">✨ GenAI (Groq){failover_note}</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">{genai_icon} {genai_pred.upper()}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Confidence: {genai_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if result.failover_used:
        st.warning("⚠️ GenAI was unavailable. Result based on weighted ML (50%) + DL (50%) decision.")


def render_history():
    """Render analysis history."""
    if not st.session_state.history:
        return
    
    st.markdown("---")
    st.markdown("## 📜 Analysis History")
    
    for i, entry in enumerate(reversed(st.session_state.history[-10:])):
        is_phishing = entry["verdict"] == "phishing"
        icon = "🚨" if is_phishing else "✅"
        color = "#ef4444" if is_phishing else "#10b981"
        
        with st.expander(f"{icon} {entry['url'][:60]}{'...' if len(entry['url']) > 60 else ''}", expanded=False):
            st.markdown(f"""
            - **Verdict:** <span style="color: {color}; font-weight: 600;">{entry['verdict'].upper()}</span>
            - **Confidence:** {entry['confidence']:.1%}
            - **ML:** {entry['ml']} | **DL:** {entry['dl']} | **GenAI:** {entry['genai']}
            """, unsafe_allow_html=True)


def main():
    """Main application."""
    render_header()
    render_sidebar()
    
    # Load orchestrator
    with st.spinner("Loading AI models..."):
        orchestrator = load_orchestrator()
    
    if orchestrator is None:
        st.error("❌ Failed to initialize the Phishing Detector. Please check your setup.")
        st.info("Make sure all model files are in the `models/` directory and dependencies are installed.")
        return
    
    # URL Input
    st.markdown("### 🔗 Enter URL to Analyze")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input(
            "URL",
            placeholder="e.g., https://example.com or suspicious-site.xyz",
            label_visibility="collapsed"
        )
    with col2:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    
    # Quick test URLs
    st.markdown("**Quick Test:**")
    quick_cols = st.columns(4)
    quick_urls = [
        ("✅ Google", "www.google.com"),
        ("✅ GitHub", "github.com/user/repo"),
        ("🚨 Suspicious", "paypal.com.login-verify.xyz"),
        ("🚨 IP-based", "http://192.168.1.1:8080/admin"),
    ]
    
    for i, (label, test_url) in enumerate(quick_urls):
        with quick_cols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                url = test_url
                analyze_btn = True
    
    # Analysis
    if analyze_btn and url:
        url = url.strip()
        if not url:
            st.warning("Please enter a valid URL.")
            return
        
        with st.spinner("🔄 Analyzing URL with AI ensemble..."):
            try:
                start_time = time.time()
                result = orchestrator.classify(url)
                elapsed = time.time() - start_time
                
                render_result(result, url)
                
                st.caption(f"⏱️ Analysis completed in {elapsed:.2f}s")
                
                # Add to history
                st.session_state.history.append({
                    "url": url,
                    "verdict": result.final_verdict,
                    "confidence": result.confidence,
                    "ml": result.ml_result.prediction if result.ml_result else "N/A",
                    "dl": result.dl_result.prediction if result.dl_result else "N/A",
                    "genai": result.genai_result.prediction if result.genai_result else "N/A",
                })
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
    
    render_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.6; font-size: 0.85rem;">
        Built with Streamlit • Powered by XGBoost, LSTM & Groq LLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
