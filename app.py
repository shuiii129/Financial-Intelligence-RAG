import streamlit as st
import os
import tempfile
import json
import re
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Executive Financial RAG Platform", layout="wide", page_icon="📈")

# Inject Custom CSS for Premium Design
css_styles = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    padding-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #9CA3AF;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #F3F4F6;
    border-bottom: 2px solid #374151;
    padding-bottom: 0.5rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* Glassmorphic Cards */
.metric-container {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}

.metric-card {
    flex: 1;
    min-width: 180px;
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 10px 25px rgba(16, 185, 129, 0.08);
}

.metric-label {
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

.metric-val-curr {
    font-size: 1.7rem;
    font-weight: 700;
    color: #F9FAFB;
    margin: 0.3rem 0;
}

.metric-val-prior {
    font-size: 0.85rem;
    color: #6B7280;
}

.metric-change {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.2rem;
}

.change-up {
    color: #10B981;
}

.change-down {
    color: #EF4444;
}

.change-neutral {
    color: #9CA3AF;
}

/* SWOT cards */
.swot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

@media (max-width: 768px) {
    .swot-grid {
        grid-template-columns: 1fr;
    }
}

.swot-card {
    border-radius: 12px;
    padding: 1.25rem;
    background: rgba(17, 24, 39, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 5px solid;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.swot-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.swot-card.s { border-left-color: #10B981; }
.swot-card.w { border-left-color: #EF4444; }
.swot-card.o { border-left-color: #3B82F6; }
.swot-card.t { border-left-color: #F59E0B; }

.swot-title.s { color: #10B981; }
.swot-title.w { color: #EF4444; }
.swot-title.o { color: #3B82F6; }
.swot-title.t { color: #F59E0B; }

.swot-list {
    margin: 0;
    padding-left: 1.2rem;
    color: #E5E7EB;
    font-size: 0.9rem;
}

.swot-list li {
    margin-bottom: 0.3rem;
}

/* Custom Alert Banner */
.custom-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
    color: #E5E7EB;
}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# Title & Intro
st.markdown("<h1 class='main-title'>📈 Executive Financial RAG Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload corporate filings or financial statements to automatically extract dashboard statistics and chat with our context-aware agent in English, Malay, or Mandarin.</p>", unsafe_allow_html=True)

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}
if "extracted_dashboard" not in st.session_state:
    st.session_state.extracted_dashboard = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Helpers for PDF Processing & Extractor
def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get('page', -1)
        if isinstance(page, int) and page >= 0:
            page += 1
        else:
            page = 'Unknown'
        formatted.append(f"[Page {page}]:\n{doc.page_content}")
    return "\n\n".join(formatted)

def parse_num(val):
    if val is None or val == "N/A":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val)
    # Remove everything except digits, dots, and negative sign
    cleaned = re.sub(r"[^\d\.\-]", "", val_str)
    try:
        num = float(cleaned)
        # Scale for chart representation
        if "B" in val_str or "b" in val_str:
            num *= 1000
        elif "M" in val_str or "m" in val_str:
            num *= 1
        elif "K" in val_str or "k" in val_str:
            num /= 1000
        return num
    except ValueError:
        return None

def robust_json_parser(text):
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = text.strip("` \n\t")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common trailing comma issues
        text_fixed = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(text_fixed)
        except json.JSONDecodeError as e:
            raise e

def extract_financial_data(docs, api_key, model_name="gemini-2.5-flash"):
    # Select first 20 pages + last 5 pages to grab main financial highlights
    selected_docs = docs[:20]
    if len(docs) > 25:
        selected_docs += docs[-5:]
    
    full_text = "\n".join([d.page_content for d in selected_docs])
    
    prompt_text = """You are an expert financial analyst. Analyze the following text extracted from a financial report.
Extract the following information:
1. A brief executive summary (3-4 sentences max, professional tone).
2. Key financial metrics (Revenue, Net Income, Operating Income/EBITDA, Total Assets, Total Liabilities) comparing the latest available period (year or quarter) to the prior period. For each metric, extract:
   - Metric Name
   - Current Period Value (with currency, e.g., "$15.4B" or "RM 420M")
   - Prior Period Value (with currency, e.g., "$13.2B" or "RM 390M")
   - Change Percentage (with positive/negative signs, e.g., "+16.7%" or "-3.2%")
3. A concise SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) with 2-3 bullet points for each section.

Return the result STRICTLY as a JSON object with this exact structure:
{{
  "executive_summary": "...",
  "metrics": [
    {{"metric": "Revenue", "value_current": "...", "value_prior": "...", "change_pct": "..."}},
    {{"metric": "Net Income", "value_current": "...", "value_prior": "...", "change_pct": "..."}},
    {{"metric": "Operating Income / EBITDA", "value_current": "...", "value_prior": "...", "change_pct": "..."}},
    {{"metric": "Total Assets", "value_current": "...", "value_prior": "...", "change_pct": "..."}},
    {{"metric": "Total Liabilities", "value_current": "...", "value_prior": "...", "change_pct": "..."}}
  ],
  "swot": {{
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "opportunities": ["...", "..."],
    "threats": ["...", "..."]
  }}
}}

Be extremely precise. If any metric is not available, write "N/A" for its values and change percentage. Do not make up numbers. Output the JSON only.

Text content:
{text_content}
"""
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, google_api_key=api_key)
        prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
        chain = prompt | llm | StrOutputParser()
        res_text = chain.invoke({"text_content": full_text[:80000]}) # limit text to avoid token overload
    except Exception as e:
        if model_name != "gemini-2.5-flash":
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)
                prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
                chain = prompt | llm | StrOutputParser()
                res_text = chain.invoke({"text_content": full_text[:80000]})
            except Exception as e_flash:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_list_str = ", ".join(models)
                except Exception as list_err:
                    model_list_str = f"Error listing models: {list_err}"
                st.warning(f"Could not extract structural dashboard data (fallback failed): {e_flash}. Available models: {model_list_str}")
                return None
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                model_list_str = ", ".join(models)
            except Exception as list_err:
                model_list_str = f"Error listing models: {list_err}"
            st.warning(f"Could not extract structural dashboard data: {e}. Available models: {model_list_str}")
            return None
            
    try:
        return robust_json_parser(res_text)
    except Exception as parse_err:
        st.warning(f"Error parsing dashboard json: {parse_err}")
        return None

def condense_question(user_question, chat_history, api_key):
    if not chat_history:
        return user_question
    
    history_text = ""
    for msg in chat_history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
        
    system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Ensure the standalone question is in the same language as the user's latest question (English, Malay, or Mandarin)."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Chat History:\n{chat_history}\n\nLatest Question: {question}")
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)
    chain = prompt | llm | StrOutputParser()
    
    try:
        standalone = chain.invoke({"chat_history": history_text, "question": user_question})
        return standalone.strip()
    except Exception as e:
        return user_question

def clean_html(html_str):
    if not html_str:
        return ""
    return "\n".join([line.strip() for line in html_str.split("\n")])

def render_metric_cards(metrics):
    if not metrics:
        return ""
    
    cards_html = '<div class="metric-container">'
    for m in metrics:
        name = m.get("metric", "Metric") or "Metric"
        curr = m.get("value_current", "N/A") or "N/A"
        prior = m.get("value_prior", "N/A") or "N/A"
        change = m.get("change_pct", "N/A") or "N/A"
        
        # Determine green/red class based on change string
        change_class = "change-neutral"
        if change != "N/A":
            try:
                change_str = str(change)
                if "+" in change_str or (change_str and not change_str.startswith("-") and re.sub(r"[^\d\.]", "", change_str)):
                    change_class = "change-up"
                elif "-" in change_str:
                    change_class = "change-down"
            except Exception:
                pass
                
        cards_html += f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-val-curr">{curr}</div>
            <div class="metric-val-prior">Prior: {prior}</div>
            <div class="metric-change {change_class}">YoY: {change}</div>
        </div>
        """
    cards_html += '</div>'
    return clean_html(cards_html)

def create_plotly_charts(metrics):
    chart_data = []
    for m in metrics:
        name = m.get("metric", "Metric")
        val_curr = m.get("value_current", "N/A")
        val_prior = m.get("value_prior", "N/A")
        
        curr_num = parse_num(val_curr)
        prior_num = parse_num(val_prior)
        
        if curr_num is not None and prior_num is not None:
            chart_data.append({
                "Metric": name,
                "Current Period": curr_num,
                "Prior Period": prior_num,
                "Label Current": val_curr,
                "Label Prior": val_prior
            })
            
    if not chart_data:
        return None
        
    df = pd.DataFrame(chart_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Metric"],
        y=df["Prior Period"],
        name="Prior Period",
        marker_color="#3B82F6",
        text=df["Label Prior"],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=df["Metric"],
        y=df["Current Period"],
        name="Current Period",
        marker_color="#10B981",
        text=df["Label Current"],
        textposition="auto"
    ))
    
    fig.update_layout(
        title="Key Metrics Comparison (Scaled to Millions)",
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Outfit, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        
    st.subheader("🤖 Model Settings")
    selected_model = st.selectbox(
        "Chat & Analysis Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        help="Use gemini-2.5-flash if you are on the free tier (gemini-2.5-pro has 0 quota for free tier accounts)."
    )
    
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type=["pdf"])
    
    # Handle File Reset on upload change
    if uploaded_file:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.vectorstore = None
            st.session_state.ensemble_retriever = None
            st.session_state.chat_history = []
            st.session_state.doc_metadata = {}
            st.session_state.extracted_dashboard = None
            st.session_state.uploaded_file_name = uploaded_file.name
            
    # Process PDF and populate state
    if uploaded_file and api_key and st.session_state.vectorstore is None:
        with st.spinner("Processing document & initializing hybrid search..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # 1. Load document
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # 2. Chunk data (Optimized size for fewer chunks/lower API load)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=400,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                splits = text_splitter.split_documents(docs)
                
                # 3. Dense Embeddings and FAISS Index (Robust Fallback & Diagnostic Setup)
                model_options = [
                    "models/text-embedding-004", 
                    "models/text-embedding-004", 
                    "models/embedding-001", 
                    "models/gemini-embedding-2-preview",
                    "text-embedding-004",
                    "text-embedding-004"
                ]
                embeddings = None
                
                last_err = None
                for model_name in model_options:
                    try:
                        temp_embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
                        # Quick validation call
                        temp_embeddings.embed_query("test query")
                        embeddings = temp_embeddings
                        break
                    except Exception as embed_err:
                        last_err = embed_err
                        continue
                
                if embeddings is None:
                    # Fetch available models supporting embedding
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        available_embed_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
                        model_list_str = ", ".join(available_embed_models)
                    except Exception as list_err:
                        model_list_str = f"Error listing models: {list_err}"
                        
                    raise ValueError(
                        f"All embedding models failed. Last error: {last_err}. "
                        f"Available embedding models for your API key: {model_list_str}"
                    )
                
                # Create vectorstore in sequential rate-limited steps to prevent 429 quota errors
                import time
                vectorstore = None
                
                progress_text = "Generating embeddings with rate-limiting (0.75s delay) to respect API quotas..."
                progress_bar = st.progress(0, text=progress_text)
                
                for idx, chunk in enumerate(splits):
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents([chunk], embeddings)
                    else:
                        vectorstore.add_documents([chunk])
                    
                    pct = min(1.0, (idx + 1) / len(splits))
                    progress_bar.progress(pct, text=f"Generated embeddings for {idx+1} of {len(splits)} chunks...")
                    
                    if idx + 1 < len(splits):
                        time.sleep(0.75)  # 0.75 seconds sleep guarantees max 80 API calls/minute
                        
                progress_bar.empty()
                faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # 4. Sparse Indexing (BM25)
                bm25_retriever = BM25Retriever.from_documents(splits)
                bm25_retriever.k = 4
                
                # 5. Build Hybrid Retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                
                # 6. Automatic structural extraction
                with st.spinner("Analyzing executive summary & YoY metrics..."):
                    dashboard_data = extract_financial_data(docs, api_key, selected_model)
                
                # Populate state
                st.session_state.vectorstore = vectorstore
                st.session_state.ensemble_retriever = ensemble_retriever
                st.session_state.extracted_dashboard = dashboard_data
                st.session_state.doc_metadata = {
                    "num_pages": len(docs),
                    "num_chunks": len(splits),
                    "chunk_size": text_splitter._chunk_size
                }
                
                st.success("Document processed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    st.divider()
    
    st.header("📄 Document Status")
    if st.session_state.vectorstore is not None:
        st.success("✅ Hybrid search active")
        if st.session_state.extracted_dashboard:
            st.success("📊 Dashboard data loaded")
    else:
        st.warning("⚠️ Pending PDF upload & API key")
        
    st.header("🔍 Index Metadata")
    if st.session_state.doc_metadata:
        st.write(f"**Total Pages:** {st.session_state.doc_metadata['num_pages']}")
        st.write(f"**Total Chunks:** {st.session_state.doc_metadata['num_chunks']}")
        st.write(f"**Chunk Size:** {st.session_state.doc_metadata['chunk_size']} characters")
    else:
        st.write("No document indexed.")
        
    if st.button("Clear Chat Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

# --- MAIN SCREEN INTERFACE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "💬 Conversational Agent", "🔍 Document Inspector"])

# TAB 1: EXECUTIVE DASHBOARD
with tab1:
    if st.session_state.extracted_dashboard:
        data = st.session_state.extracted_dashboard
        
        # Summary Card
        st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
        summary = data.get("executive_summary", "No executive summary available.")
        st.markdown(f"<div class='custom-banner'>💡 {summary}</div>", unsafe_allow_html=True)
        
        # Metric Scorecards
        st.markdown("<div class='section-title'>Key Financial Metrics</div>", unsafe_allow_html=True)
        st.caption("ℹ️ *Note: If metrics display as N/A, please ensure the uploaded PDF contains consolidated financial statements (e.g. 10-K or 10-Q reports). Proxy statements (DEF 14A) generally do not contain these figures.*")
        metrics = data.get("metrics", [])
        st.markdown(render_metric_cards(metrics), unsafe_allow_html=True)
        
        # Plotly comparison
        fig = create_plotly_charts(metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
        # SWOT Analysis Grid
        st.markdown("<div class='section-title'>SWOT Analysis</div>", unsafe_allow_html=True)
        swot = data.get("swot", {})
        s_list = "".join([f"<li>{item}</li>" for item in swot.get("strengths", [])])
        w_list = "".join([f"<li>{item}</li>" for item in swot.get("weaknesses", [])])
        o_list = "".join([f"<li>{item}</li>" for item in swot.get("opportunities", [])])
        t_list = "".join([f"<li>{item}</li>" for item in swot.get("threats", [])])
        
        swot_html = f"""
        <div class="swot-grid">
            <div class="swot-card s">
                <div class="swot-title s">💪 Strengths</div>
                <ul class="swot-list">{s_list or "<li>N/A</li>"}</ul>
            </div>
            <div class="swot-card w">
                <div class="swot-title w">⚠️ Weaknesses</div>
                <ul class="swot-list">{w_list or "<li>N/A</li>"}</ul>
            </div>
            <div class="swot-card o">
                <div class="swot-title o">🚀 Opportunities</div>
                <ul class="swot-list">{o_list or "<li>N/A</li>"}</ul>
            </div>
            <div class="swot-card t">
                <div class="swot-title t">⚡ Threats</div>
                <ul class="swot-list">{t_list or "<li>N/A</li>"}</ul>
            </div>
        </div>
        """
        st.markdown(clean_html(swot_html), unsafe_allow_html=True)
    else:
        st.info("👋 Welcome! Please upload a financial PDF report in the sidebar to populate the Executive Dashboard.")

# TAB 2: CONVERSATIONAL AGENT
with tab2:
    if st.session_state.ensemble_retriever:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        user_question = st.chat_input("Ask a question about this report (English, Malay, or Mandarin)...")
        
        if user_question:
            if not api_key:
                st.error("Please provide a Google API Key in the sidebar.")
                st.stop()
                
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document using hybrid search..."):
                    # 1. Condense question using chat memory
                    condensed = condense_question(user_question, st.session_state.chat_history[:-1], api_key)
                    
                    # 2. Retrieve sources using Ensemble
                    retrieved_docs = st.session_state.ensemble_retriever.invoke(condensed)
                    context_text = format_docs(retrieved_docs)
                    
                    # 3. Formulate RAG generation
                    system_prompt = """You are a highly capable multilingual financial assistant.
                    
First, identify the language of the user's question (it will be English, Malay, or Mandarin).
Second, formulate your response entirely in that same identified language.

Use ONLY the provided context below to answer the user's question. If the answer cannot be found in the context, clearly state "I do not know the answer based on the provided document" in the detected language. Do not make up information or use outside knowledge.

At the end of your response, you MUST provide explicit source citations referencing the exact page numbers from the context used to form your answer.

Context:
{context}
"""
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{question}")
                    ])
                    try:
                        # Attempt to use selected model, fallback to gemini-2.5-flash
                        try:
                            llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.1, google_api_key=api_key)
                            chain = prompt | llm | StrOutputParser()
                            response = chain.invoke({"question": user_question, "context": context_text})
                        except Exception as pro_err:
                            if selected_model != "gemini-2.5-flash":
                                st.warning(f"{selected_model} failed. Falling back to gemini-2.5-flash...")
                                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)
                                chain = prompt | llm | StrOutputParser()
                                response = chain.invoke({"question": user_question, "context": context_text})
                            else:
                                raise pro_err
                                
                        st.markdown(response)
                        
                        # Display sources in expander
                        with st.expander("🔍 View Retrieved Hybrid Chunks"):
                            for i, doc in enumerate(retrieved_docs):
                                page = doc.metadata.get('page', -1)
                                if isinstance(page, int) and page >= 0:
                                    page += 1
                                else:
                                    page = 'Unknown'
                                st.markdown(f"**Chunk {i+1} (Page {page})**")
                                st.text(doc.page_content[:350] + "...")
                                
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        # Fetch available models supporting content generation
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            available_gen_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            model_list_str = ", ".join(available_gen_models)
                        except Exception as list_err:
                            model_list_str = f"Error listing models: {list_err}"
                            
                        st.error(f"Generation error: {e}. \n\nAvailable generation models for your API key: {model_list_str}")
    else:
        st.warning("⚠️ Chat agent is locked. Please upload a PDF and set your API key in the sidebar to begin chatting.")

# TAB 3: DOCUMENT INSPECTOR
with tab3:
    if st.session_state.ensemble_retriever:
        st.markdown("<div class='section-title'>Document Index Information</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", st.session_state.doc_metadata.get("num_pages", 0))
        with col2:
            st.metric("Document Chunks", st.session_state.doc_metadata.get("num_chunks", 0))
        with col3:
            st.metric("Base Chunk Size (characters)", st.session_state.doc_metadata.get("chunk_size", 0))
            
        st.markdown("<div class='section-title'>Direct Document Explorer</div>", unsafe_allow_html=True)
        st.markdown("Search specifically for key terms or metrics to return and read matching snippets directly.")
        search_query = st.text_input("Enter search keywords (e.g. 'Operating Margin', 'Deferred Revenue'):")
        
        if search_query:
            with st.spinner("Searching document..."):
                search_results = st.session_state.ensemble_retriever.invoke(search_query)
                st.write(f"Showing top {len(search_results)} relevant matching chunks:")
                for idx, doc in enumerate(search_results):
                    page = doc.metadata.get('page', -1)
                    if isinstance(page, int) and page >= 0:
                        page += 1
                    else:
                        page = 'Unknown'
                    with st.container():
                        st.markdown(f"📄 **Passage {idx+1} (Page {page})**")
                        st.markdown(f"<div style='background:rgba(255,255,255,0.03); padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05); margin-bottom:1rem;'>{doc.page_content}</div>", unsafe_allow_html=True)
    else:
        st.info("💡 Please upload a document in the sidebar to explore the page metrics and retrieve direct chunks.")
