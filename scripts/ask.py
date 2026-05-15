"""Ask questions over the ingested document collection (CLI).

Thin command-line wrapper around `local_rag_sandbox.qa.answer_question`.
The real RAG logic lives in `src/local_rag_sandbox/qa.py` so the same
pipeline can be reused by other front-ends (Streamlit, future test
harness, etc.).

Usage:
    python scripts/ask.py "Your question here"                    # whole corpus
    python scripts/ask.py                                          # interactive prompt
    python scripts/ask.py --source dnv "..."                       # filter by publisher
    python scripts/ask.py --year 2024 "..."                        # filter by year
    python scripts/ask.py --topic global_hydrogen_review "..."     # filter by topic
    python scripts/ask.py --source iea --year 2024 "..."           # combined (AND)
    python scripts/ask.py --top-k 20 "..."                         # broader synthesis
    python scripts/ask.py --top-k 4 "..."                          # tighter context
    python scripts/ask.py --depth learning "..."                   # deep R&D synthesis

Filters use AND semantics and exact-match. --source and --topic are
normalised to lowercase to match the metadata produced during ingestion.
Omitting all filters searches the full corpus (same as the bare command).

--depth controls the prompt template and the default --top-k:
    concise   -> top_k=6,  brief answer (Question/Answer/Evidence)
    standard  -> top_k=15, analytical answer with Discussion (the default)
    learning  -> top_k=20, deep R&D synthesis (6 sections)

An explicit --top-k always wins over the depth default.

Prerequisite: run scripts/ingest.py at least once so that CHROMA_DIR
contains an indexed collection.
"""

import argparse

from local_rag_sandbox.qa import DEPTH_TOP_K, answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--source",
        help="Filter chunks to a specific publisher (e.g. dnv, iea). Case-insensitive.",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter chunks to a specific year (e.g. 2024).",
    )
    parser.add_argument(
        "--topic",
        help="Filter chunks to a specific topic (e.g. global_hydrogen_review). Case-insensitive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of chunks to retrieve. If omitted, chosen by --depth "
            f"({DEPTH_TOP_K['concise']}/{DEPTH_TOP_K['standard']}/{DEPTH_TOP_K['learning']})."
        ),
    )
    parser.add_argument(
        "--depth",
        choices=["concise", "standard", "learning"],
        default="standard",
        help="Answer style. Default: standard.",
    )
    args = parser.parse_args()

    if args.question:
        query = " ".join(args.question).strip()
    else:
        query = input("Ask a question about the indexed documents: ").strip()

    if not query:
        print("Empty question. Exiting.")
        return

    result = answer_question(
        query=query,
        source=args.source,
        year=args.year,
        topic=args.topic,
        depth=args.depth,
        top_k=args.top_k,
        on_progress=print,
    )
    if result is None:
        return

    print(result.answer)


if __name__ == "__main__":
    main()
