import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import streamlit as st


VECTOR_DIR = Path(__file__).parent / "scripts" / "vector_rag"
sys.path.insert(0, str(VECTOR_DIR))

from vector_retrieval import (
    load_vector_system,
    retrieve_vector,
)

from vector_rag import (
    load_generator,
    generate_answer,
)


st.set_page_config(
    page_title="Legal Contract RAG",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def load_models():
    index, metadata, embedding_model = load_vector_system()
    tokenizer, generation_model = load_generator()

    return (
        index,
        metadata,
        embedding_model,
        tokenizer,
        generation_model,
    )


def make_question(question_text, document_id):
    """
    retrieve_vector() expects the same structure used
    by the evaluation dataset.
    """

    return {
        "question": question_text,
        "gold_passages": [
            {
                "chunk_id": f"{document_id}::0"
            }
        ],
    }


st.title("⚖️ Legal Contract RAG")

st.write(
    "Ask questions about the legal contracts using "
    "semantic retrieval and Vector RAG."
)

with st.spinner("Loading Vector RAG models..."):
    (
        index,
        metadata,
        embedding_model,
        tokenizer,
        generation_model,
    ) = load_models()


# ---------------------------------------------------------
# Contract selector
# ---------------------------------------------------------

documents = sorted(
    {
        chunk["document_id"]
        for chunk in metadata
    }
)

selected_document = st.selectbox(
    "Select contract",
    documents,
)


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

question_text = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What is the royalty rate "
        "defined in this agreement?"
    ),
    height=100,
)


top_k = st.slider(
    "Number of evidence chunks",
    min_value=1,
    max_value=10,
    value=5,
)


# ---------------------------------------------------------
# Run RAG
# ---------------------------------------------------------

if st.button(
    "Ask Vector RAG",
    type="primary",
    use_container_width=True,
):

    if not question_text.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        question = make_question(
            question_text,
            selected_document,
        )

        with st.spinner(
            "Retrieving contract evidence..."
        ):

            retrieved = retrieve_vector(
                question,
                index,
                metadata,
                embedding_model,
                top_k=top_k,
                restrict_to_document=True,
            )

        if not retrieved:

            st.error(
                "No evidence was retrieved."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question_text,
                    retrieved,
                    tokenizer,
                    generation_model,
                )

            # -------------------------------------------------
            # Answer
            # -------------------------------------------------

            st.subheader("Answer")

            st.success(answer)

            # -------------------------------------------------
            # Evidence
            # -------------------------------------------------

            st.subheader("Retrieved Evidence")

            for result in retrieved:

                title = (
                    f"Rank {result['rank']} "
                    f"• Similarity "
                    f"{result['score']:.3f}"
                )

                with st.expander(title):

                    st.write(
                        "**Pages:**",
                        result["pages"],
                    )

                    st.write(
                        "**Chunk ID:**"
                    )

                    st.code(
                        result["chunk_id"]
                    )

                    st.write(
                        "**Evidence:**"
                    )

                    st.write(
                        result["text"]
                    )


st.divider()

st.caption(
    "Vector retrieval: all-MiniLM-L6-v2 + FAISS | "
    "Generation: FLAN-T5 Base"
)