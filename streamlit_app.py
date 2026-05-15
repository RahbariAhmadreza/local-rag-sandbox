"""Streamlit UI for Local Document Intelligence.

Run:
    streamlit run streamlit_app.py

Full-corpus Q&A over the local Chroma index — no metadata filters in the UI.
Calls local_rag_sandbox.qa.answer_question with source/year/topic=None.
"""

import streamlit as st

from local_rag_sandbox.config import CHROMA_DIR
from local_rag_sandbox.qa import (
    DEPTH_TOP_K,
    Depth,
    QAResult,
    RetrievedChunk,
    answer_question,
)

# Streamlit radio keys -> internal depth keys passed to qa.answer_question
DEPTH_UI_KEYS: tuple[str, ...] = ("brief", "analytical", "deep_rd")
DEPTH_UI_TO_INTERNAL: dict[str, Depth] = {
    "brief": "concise",
    "analytical": "standard",
    "deep_rd": "learning",
}
DEPTH_UI_LABELS: dict[str, str] = {
    "brief": "Brief",
    "analytical": "Analytical",
    "deep_rd": "Deep R&D",
}

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What were the main hydrogen bottlenecks mentioned across the indexed reports?",
    "What electrolyser technologies appear most often, and how do reports describe their trade-offs?",
    "What investment, offtake, and FID risks do the reports associate with hydrogen projects?",
    "How do the reports describe growth in low-emission hydrogen project pipelines over time?",
)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def _index_is_ready() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def _render_evidence(chunks: list[RetrievedChunk]) -> None:
    """Single collapsed expander; chunks listed inside without nested expanders."""
    label = f"Evidence — {len(chunks)} retrieved chunk(s)"
    with st.expander(label, expanded=False):
        if not chunks:
            st.caption("No chunks were retrieved for this answer.")
            return
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i} — {chunk.filename}, p. {chunk.page_display}**")
            st.markdown(
                f"- **Source:** `{chunk.source}`\n"
                f"- **Year:** `{chunk.year}`\n"
                f"- **Topic:** `{chunk.topic}`\n"
                f"- **File:** `{chunk.filename}`\n"
                f"- **Page (display):** `{chunk.page_display}`\n"
                f"- **Page (raw loader value):** `{chunk.page_raw}`"
            )
            st.markdown("**Content:**")
            st.text(chunk.content)
            if i < len(chunks):
                st.divider()


def _render_assistant_turn(
    result: QAResult | None,
    progress_lines: list[str],
) -> None:
    if result is None:
        st.error(
            "No relevant chunks found for that question. Try rephrasing "
            "or using a deeper answer mode."
        )
        if progress_lines:
            with st.expander("Pipeline log", expanded=False):
                st.code("\n".join(progress_lines))
        return

    st.caption(result.filter_context)
    st.markdown(result.answer)
    _render_evidence(result.retrieved_chunks)


def _generate_assistant_reply(
    question: str,
    depth: Depth,
) -> tuple[QAResult | None, list[str]]:
    progress_lines: list[str] = []

    def on_progress(message: str) -> None:
        progress_lines.append(message)

    with st.status("Working …", expanded=False) as status:
        def on_progress_status(message: str) -> None:
            on_progress(message)
            label = message.strip() or "Working …"
            status.update(label=label)

        result = answer_question(
            query=question,
            source=None,
            year=None,
            topic=None,
            depth=depth,
            on_progress=on_progress_status,
        )
        status.update(label="Done", state="complete")

    return result, progress_lines


def main() -> None:
    st.set_page_config(
        page_title="Local Document Intelligence",
        page_icon="📚",
        layout="wide",
    )
    _init_session_state()

    with st.sidebar:
        st.header("Answer mode")
        depth_ui = st.radio(
            "Mode",
            options=list(DEPTH_UI_KEYS),
            index=1,
            format_func=lambda key: DEPTH_UI_LABELS[key],
            help=(
                "Brief: shorter grounded answer. "
                "Analytical: structured discussion with evidence. "
                "Deep R&D: thematic synthesis across many chunks."
            ),
            label_visibility="collapsed",
        )
        depth = DEPTH_UI_TO_INTERNAL[depth_ui]
        st.caption(
            f"{DEPTH_UI_LABELS[depth_ui]} — retrieves up to "
            f"{DEPTH_TOP_K[depth]} chunks from the full library."
        )

        st.divider()
        if st.button(
            "Clear chat",
            help="Clear conversation history only. Does not affect Chroma or ingested files.",
        ):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    if not _index_is_ready():
        st.warning(
            "No indexed documents found. Run `python scripts/ingest.py` first, "
            "then refresh this page."
        )
        st.stop()

    st.title("Local Document Intelligence")
    st.caption(
        "Ask deeper R&D-style questions over your local document library. "
        "Offline. Evidence-grounded."
    )

    if not st.session_state.messages:
        st.markdown("**Try an example:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for i, (col, example) in enumerate(zip(cols, EXAMPLE_QUESTIONS, strict=True)):
            with col:
                if st.button(
                    example,
                    key=f"example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_prompt = example
                    st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_turn(msg.get("result"), msg.get("progress_lines", []))

    pending = st.session_state.pending_prompt
    if pending:
        st.session_state.pending_prompt = None
    prompt = pending or st.chat_input("Ask a question about your documents")

    if prompt and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        st.rerun()

    messages = st.session_state.messages
    if messages and messages[-1]["role"] == "user":
        question = messages[-1]["content"]
        with st.chat_message("assistant"):
            result, progress_lines = _generate_assistant_reply(
                question=question,
                depth=depth,
            )
            _render_assistant_turn(result, progress_lines)

        st.session_state.messages.append({
            "role": "assistant",
            "result": result,
            "progress_lines": progress_lines,
        })
        st.rerun()


if __name__ == "__main__":
    main()
