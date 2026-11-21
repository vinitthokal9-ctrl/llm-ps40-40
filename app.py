"""Streamlit interface for the Product Manual Assistant."""

from __future__ import annotations

import hashlib
import html
import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import ManualAssistant, ManualError

load_dotenv()

st.set_page_config(
    page_title="Product Manual Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --blue: #155eef; --navy: #102a56; --pale: #f2f7ff; }
      .stApp { background: #f7faff; }
      [data-testid="stHeader"] { background: rgba(247,250,255,.9); }
      .hero { background: linear-gradient(115deg,#123d83,#2169d5); color:white;
        padding: 1.5rem 1.8rem; border-radius: 18px; margin: .3rem 0 1.2rem; }
      .hero h1 { margin:0; font-size:2rem; }
      .hero p { margin:.45rem 0 0; opacity:.92; }
      .source-card { background:#eef5ff; border-left:4px solid #2169d5;
        border-radius:8px; padding:.7rem .9rem; margin:.4rem 0; }
      .stChatMessage { border:1px solid #e4ebf5; border-radius:12px; }
      [data-testid="stSidebar"] { background:#f0f5fc; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "file_digest" not in st.session_state:
    st.session_state.file_digest = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    '<div class="hero"><h1>📘 Product Manual Assistant</h1>'
    '<p>Ask questions about your appliance manual and get clear, page-linked guidance.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Your manual")
    st.caption("Upload a PDF to build a searchable knowledge base for this session.")
    uploaded = st.file_uploader("Choose a product manual", type=["pdf"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        if digest != st.session_state.file_digest:
            st.session_state.assistant = None
            st.session_state.messages = []
            st.session_state.file_digest = digest
        if st.session_state.assistant is None:
            if st.button("Process manual", type="primary", use_container_width=True):
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your .env file, then restart the app.")
                else:
                    try:
                        with st.spinner("Reading pages and creating the searchable index…"):
                            st.session_state.assistant = ManualAssistant.from_pdf(file_bytes)
                        st.success(
                            f"Ready: {st.session_state.assistant.page_count} pages, "
                            f"{st.session_state.assistant.chunk_count} searchable sections."
                        )
                    except ManualError as exc:
                        st.error(str(exc))
                    except Exception as exc:  # show a useful message without exposing secrets
                        st.error(f"Could not process this PDF ({type(exc).__name__}). Check the file and try again.")
        elif st.session_state.assistant:
            st.success(
                f"Indexed {st.session_state.assistant.page_count} pages "
                f"({st.session_state.assistant.chunk_count} sections)."
            )
    st.divider()
    st.markdown("**How it works**")
    st.caption("PDF pages → text chunks → Sentence Transformer embeddings → FAISS search → Gemini answer")
    st.info("Answers rely on the uploaded manual. Always follow the manufacturer’s safety instructions.")

if st.session_state.assistant is None:
    st.info("Upload a PDF manual in the sidebar and select **Process manual** to begin.")
    st.markdown("#### Designed for practical troubleshooting")
    st.write("Ask in everyday language. The assistant searches your manual, cites the page numbers it used, and says when the manual does not contain enough information.")
else:
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                st.markdown("**Manual pages used**")
                for source in item["sources"]:
                    st.markdown(
                        f'<div class="source-card"><strong>Page {source["page"]}</strong> · '
                        f'{html.escape(str(source["preview"]))}</div>',
                        unsafe_allow_html=True,
                    )

    question = st.chat_input("For example: What should I check if the washer will not drain?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the manual and preparing an answer…"):
                    answer, sources = st.session_state.assistant.answer(question)
                st.markdown(answer)
                if sources:
                    st.markdown("**Manual pages used**")
                    for source in sources:
                        st.markdown(
                            f'<div class="source-card"><strong>Page {source["page"]}</strong> · '
                            f'{html.escape(str(source["preview"]))}</div>',
                            unsafe_allow_html=True,
                        )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ManualError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"I couldn’t complete that request ({type(exc).__name__}). Please try again.")
