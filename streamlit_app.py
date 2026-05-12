"""Streamlit UI for the Local RAG Sandbox.

Run:
    streamlit run streamlit_app.py

Same core RAG pipeline as scripts/ask.py — both call into
local_rag_sandbox.qa.answer_question. This file is just a different
front-end: filters + depth selector in the sidebar, question textbox
and answer + expandable evidence in the main area.
"""

import streamlit as st

from local_rag_sandbox.qa import (
    DEPTH_TOP_K,
    answer_question,
    list_filter_options,
)

ANY_LABEL = "Any"


@st.cache_data(show_spinner=False)
def _filter_options() -> tuple[list[str], list[int], list[str]]:
    """Cached wrapper around qa.list_filter_options() for sidebar dropdowns."""
    return list_filter_options()


def main() -> None:
    st.set_page_config(
        page_title="Local RAG Sandbox",
        page_icon="📚",
        layout="wide",
    )
    st.title("Local RAG Sandbox")
    st.caption(
        "Ask grounded questions over your indexed documents. "
        "Offline. Local. No data leaves your machine."
    )

    sources, years, topics = _filter_options()

    with st.sidebar:
        st.header("Filters")
        source = st.selectbox(
            "Publisher (source)",
            options=[ANY_LABEL] + sources,
            help="The folder name under data/, e.g. dnv, iea.",
        )
        year_choice = st.selectbox(
            "Year",
            options=[ANY_LABEL] + [str(y) for y in years],
            help="Parsed from the YYYY_ filename prefix during ingest.",
        )
        topic = st.selectbox(
            "Topic",
            options=[ANY_LABEL] + topics,
            help="Parsed from the filename body during ingest.",
        )

        st.header("Answer depth")
        depth = st.radio(
            "Depth",
            options=["concise", "standard", "learning"],
            index=1,
            format_func=str.capitalize,
            help=(
                "Concise: 2 sentences + citations. "
                "Standard: 5-section answer. "
                "Learning: 5-section answer with definitions and context."
            ),
            label_visibility="collapsed",
        )
        st.caption(f"Retrieves {DEPTH_TOP_K[depth]} chunks at {depth} depth.")

        st.divider()
        if st.button(
            "Refresh filter options",
            help="Re-read source / year / topic values from Chroma. Use this "
                 "after running scripts/ingest.py while Streamlit is running.",
        ):
            st.cache_data.clear()
            st.rerun()

    if not sources and not years and not topics:
        st.warning(
            "No indexed chunks found. Run `python scripts/ingest.py` first, "
            "then click 'Refresh filter options' in the sidebar."
        )
        st.stop()

    question = st.text_area(
        "Question",
        placeholder=(
            "e.g. What were the main hydrogen bottlenecks mentioned across "
            "the 2024 reports?"
        ),
        height=120,
    )
    ask_clicked = st.button(
        "Ask",
        type="primary",
        disabled=not question.strip(),
    )
    if not ask_clicked:
        st.stop()

    selected_source = None if source == ANY_LABEL else source
    selected_year = None if year_choice == ANY_LABEL else int(year_choice)
    selected_topic = None if topic == ANY_LABEL else topic

    progress_lines: list[str] = []
    with st.status("Working …", expanded=False) as status:
        def on_progress(message: str) -> None:
            progress_lines.append(message)
            label = message.strip() or "Working …"
            status.update(label=label)

        result = answer_question(
            query=question.strip(),
            source=selected_source,
            year=selected_year,
            topic=selected_topic,
            depth=depth,
            on_progress=on_progress,
        )
        status.update(label="Done", state="complete")

    if result is None:
        st.error(
            "No relevant chunks found for that combination. Try broadening "
            "the filters or rephrasing the question."
        )
        with st.expander("Pipeline log"):
            st.code("\n".join(progress_lines))
        st.stop()

    st.info(result.filter_context)
    st.markdown(result.answer)

    st.subheader(f"Evidence — {len(result.retrieved_chunks)} retrieved chunk(s)")
    for i, chunk in enumerate(result.retrieved_chunks, start=1):
        label = f"Chunk {i} — {chunk.filename}, p. {chunk.page_display}"
        with st.expander(label):
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


if __name__ == "__main__":
    main()
