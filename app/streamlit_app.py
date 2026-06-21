import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

load_dotenv(ROOT / ".env")

from rag.agent import answer_question
from rag.chunking import recursive_chunks
from rag.ingest import ingest_folder
from rag.vectorstore import build_chroma

st.set_page_config(page_title="Biomedical RAG Assistant", layout="wide")

st.title("Biomedical RAG Assistant")
st.caption(
    "Literature-review assistant with citations, guardrails, and trace IDs. "
    "Not for clinical advice."
)

if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history = [
    x for x in st.session_state.history if isinstance(x, dict)
]

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Add biomedical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        raw_dir = ROOT / "data" / "raw_pdfs"
        processed_dir = ROOT / "data" / "processed"
        vector_dir = ROOT / "vectorstore" / "chroma"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for uploaded in uploaded_files:
            path = raw_dir / uploaded.name

            if uploaded.name not in st.session_state.indexed_files:
                path.write_bytes(uploaded.getbuffer())
                st.session_state.indexed_files.add(uploaded.name)
                saved_count += 1

        if saved_count > 0:
            with st.spinner(f"Indexing {saved_count} PDF(s). Please wait..."):
                ingest_folder(
                    str(raw_dir),
                    str(processed_dir / "documents.jsonl"),
                )

                recursive_chunks(
                    str(processed_dir / "documents.jsonl"),
                    str(processed_dir / "chunks.jsonl"),
                )

                build_chroma(
                    str(processed_dir / "chunks.jsonl"),
                    str(vector_dir),
                )

            st.success(f"{saved_count} PDF(s) indexed successfully.")
        else:
            st.info("These PDFs are already uploaded in this session.")

    st.divider()

    if st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()

    st.caption(
        "Upload 50+ PDFs together. The app will index all PDFs from data/raw_pdfs."
    )

question = st.chat_input("Ask a biomedical literature question...")

if question:
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is missing. Create a .env file in the project root and add your key."
        )
    else:
        with st.spinner("Retrieving and verifying answer..."):
            ans, chunks, trace_id, latency = answer_question(question)

        st.session_state.history.append(
            {
                "question": question,
                "answer": ans,
                "chunks": chunks,
                "trace_id": trace_id,
                "latency": latency,
            }
        )

for item in reversed(st.session_state.history):
    if item is None:
        continue

    if not isinstance(item, dict):
        continue

    if "question" not in item:
        continue

    question = item.get("question", "")
    ans = item.get("answer", None)
    chunks = item.get("chunks", [])
    trace_id = item.get("trace_id", "N/A")
    latency = item.get("latency", 0)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if ans is None:
            st.error("No answer available.")
            continue

        try:
            st.write(ans.answer)
        except Exception:
            st.write(str(ans))

        st.caption(
            f"Confidence: {getattr(ans, 'confidence', 'unknown')} | "
            f"Trace ID: {trace_id} | "
            f"Latency: {latency:.2f}s"
        )

        citations = getattr(ans, "citations", [])

        if citations:
            st.subheader("Citations")
            for c in citations:
                try:
                    st.markdown(
                        f"- **{c.title}**, page {c.page_number}, chunk `{c.chunk_id}`"
                    )
                except Exception:
                    st.write(str(c))

        with st.expander("Retrieved Chunks"):
            for c in chunks:
                try:
                    st.markdown(
                        f"**{c.title} — page {c.page_number} — score {c.score:.3f} — chunk `{c.chunk_id}`**"
                    )
                    st.write(c.text[:1200])
                except Exception:
                    st.write(str(c))

        col1, col2 = st.columns(2)

        feedback_file = ROOT / "data" / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        with col1:
            if st.button("👍", key=f"up-{trace_id}"):
                with open(feedback_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "trace_id": trace_id,
                                "question": question,
                                "feedback": "up",
                            }
                        )
                        + "\n"
                    )

                st.success("Feedback saved.")

        with col2:
            if st.button("👎", key=f"down-{trace_id}"):
                with open(feedback_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "trace_id": trace_id,
                                "question": question,
                                "feedback": "down",
                            }
                        )
                        + "\n"
                    )

                st.success("Feedback saved.")