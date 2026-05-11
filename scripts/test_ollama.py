"""Smoke test: verify that LangChain can call the local Ollama chat model."""

from langchain_ollama import ChatOllama

from local_rag_sandbox.config import CHAT_MODEL


def main() -> None:
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(
        "Explain grid congestion in the energy transition in three simple bullet points."
    )
    print(response.content)


if __name__ == "__main__":
    main()
