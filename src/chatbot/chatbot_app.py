from __future__ import annotations

import argparse

from src.chatbot.chat_engine import GuardrailedChatEngine


def run_cli(question: str, case_id: str | None = None) -> None:
    response = GuardrailedChatEngine().answer(question, case_id=case_id)
    print(response.answer)
    if response.guardrail_reasons:
        print(f"Guardrail reasons: {response.guardrail_reasons}")


def run_streamlit() -> None:
    try:
        import streamlit as st
    except Exception:
        run_cli("Is this model safe for autonomous HR decisions?")
        return
    st.title("LLM Governance & Audit")
    st.caption("Research prototype: explains governed XAI evidence, not HR decisions.")
    case_id = st.text_input("Case ID", value="")
    question = st.text_area("Ask a guarded model-audit question", value="Why did the model predict this class?")
    if st.button("Ask"):
        response = GuardrailedChatEngine().answer(question, case_id=case_id or None)
        st.write(response.answer)
        if not response.allowed:
            st.warning("Unsafe request refused.")
        st.json({"allowed": response.allowed, "guardrail_reasons": response.guardrail_reasons, "context_keys": response.context_keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guardrailed HR explanation chatbot.")
    parser.add_argument("--question", default="Is this model safe for autonomous HR decisions?")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--streamlit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.streamlit:
        run_streamlit()
    else:
        run_cli(args.question, case_id=args.case_id)

