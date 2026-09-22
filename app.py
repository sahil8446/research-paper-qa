"""Tiny Streamlit interface for the Phase 2 baseline: ask a question, see the cited answer
and the passages it was built from. Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import config  # noqa: E402
from rag.answer import answer_question  # noqa: E402
from rag.vectorstore import get_client  # noqa: E402

st.set_page_config(page_title="Research Paper Q&A", page_icon="📄")
st.title("Research Paper Q&A Assistant")
st.caption(
    f"Meaning search over {config.QDRANT_COLLECTION} + {config.ANSWER_MODEL} -- Phase 2 baseline, "
    "no reranker or agent loop yet."
)


@st.cache_resource
def _client():
    return get_client()


question = st.text_input("Ask a question about the paper library")

if question:
    with st.spinner("Searching and writing an answer..."):
        try:
            answered = answer_question(question, client=_client())
        except RuntimeError as err:
            st.error(str(err))
            st.stop()

    if answered.result.not_found:
        st.warning(answered.result.answer)
    else:
        st.markdown(answered.result.answer)

    st.subheader("Sources")
    for i, chunk in enumerate(answered.sources, start=1):
        used = i in answered.result.citations
        tag = " (cited)" if used else ""
        label = f"[{i}]{tag} {chunk.title} -- {chunk.section}, p.{chunk.page_start}"
        with st.expander(label, expanded=used):
            st.caption(f"{', '.join(chunk.authors)} ({chunk.year})")
            st.write(chunk.text)
