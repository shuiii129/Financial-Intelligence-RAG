#  Executive Financial RAG Platform

A premium corporate filings and annual reports analysis agent. It automatically extracts key financial metrics, generates a dynamic SWOT analysis, visualizes year-over-year trends, and provides a context-aware conversational agent supporting English, Malay, and Mandarin.

---

##  Features

- ** Automated Executive Dashboard**: Instantaneous generation of an executive summary, key Year-over-Year (YoY) financial metrics table, and SWOT analysis quadrants immediately upon document upload.
- ** Interactive Trend Visualizations**: Groups and charts primary metrics (Revenue, Net Income, Operating Income/EBITDA, Assets, Liabilities) using Plotly.
- ** Conversational Agent with Memory**: Resolve follow-up questions (e.g. *"What is the revenue?"* followed by *"Did it grow?"*) via a query-condensing memory chain.
- ** Hybrid Dense/Sparse Search**: Employs LangChain's `EnsembleRetriever` combining dense semantic search (`FAISS`) and sparse keyword matching (`BM25`) for maximum accuracy with financial tables and specific numbers.
- ** Rate-Limit Resilient Architecture**: Automatically schedules text embeddings sequentially (30-chunk batches, 0.75s sleeps) to remain safely under Gemini API free tier limits (429 protection).
- ** Multi-Model Fallbacks**: Automatically falls back from `gemini-2.5-pro` to `gemini-2.5-flash` if your API key has restricted access levels, guaranteeing continuous functionality.
- ** Multilingual Support**: Seamlessly identifies and replies to inquiries in English, Malay, or Mandarin.

---

##  Technology Stack

- **Core**: Python 3.10+
- **Front-end**: Streamlit (Premium Custom Glassmorphic Styling)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Sparse Retriever**: BM25 (via `rank-bm25`)
- **LLM/Embeddings**: Gemini API (`gemini-2.5-pro` / `gemini-2.5-flash` / `text-embedding-004`)
- **Orchestration**: LangChain

---

##  Installation & Setup

### 1. Clone the repository and navigate inside:
```bash
cd financial_rag_agent
```

### 2. Install dependencies:
Make sure you have python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. API Key Setup:
Create a `.env` file in the root directory to store your Google Gemini API Key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
*(Alternatively, you can paste your API key directly in the sidebar inside the web application).*

---

##  Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will launch in your default web browser at **`http://localhost:8501`**.

---

##  Usage & Testing

1. **Upload a Document**: Drop a financial PDF report (annual reports, proxy statements, 10-K filings, etc.) in the sidebar upload box.
2. **Dashboard Tab**: Check the generated **Executive Summary**, the metric scorecards (which color-code growth trends automatically), the interactive Plotly bar chart, and the **SWOT Analysis** quadrant.
3. **Conversational Tab**: Ask direct questions about the document in English, Malay, or Mandarin.
4. **Document Inspector Tab**: Review parsing metadata and use the **Direct Document Explorer** to search for specific key-phrases matching exact source passages.
