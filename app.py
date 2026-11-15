import os

# Prevent macOS/PyTorch multiprocessing crashes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import streamlit as st


# =========================================================
# IMPORT PATHS
# =========================================================

ROOT = Path(__file__).parent

VECTOR_DIR = ROOT / "scripts" / "vector_rag"
BM25_DIR = ROOT / "scripts" / "bm25"

sys.path.insert(0, str(VECTOR_DIR))
sys.path.insert(0, str(BM25_DIR))


# =========================================================
# VECTOR IMPORTS
# =========================================================

from vector_retrieval import (
    load_vector_system,
    retrieve_vector,
)

from vector_rag import (
    load_generator,
    generate_answer,
)


# =========================================================
# BM25 IMPORTS
# =========================================================

from bm25_retrieval import (
    CHUNKS_DIR,
    load_chunks,
    build_bm25_indexes,
    retrieve_bm25,
)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Legal Contract Retrieval & RAG",
    page_icon="⚖️",
    layout="wide",
)


# =========================================================
# LOAD SYSTEMS
# =========================================================

@st.cache_resource
def load_systems():

    # Vector system
    index, metadata, embedding_model = (
        load_vector_system()
    )

    tokenizer, generation_model = (
        load_generator()
    )

    # BM25 system
    chunks = load_chunks(CHUNKS_DIR)

    bm25_indexes = build_bm25_indexes(
        chunks
    )

    return (
        index,
        metadata,
        embedding_model,
        tokenizer,
        generation_model,
        bm25_indexes,
    )


with st.spinner("Loading retrieval systems..."):

    (
        vector_index,
        metadata,
        embedding_model,
        tokenizer,
        generation_model,
        bm25_indexes,
    ) = load_systems()


# =========================================================
# HELPERS
# =========================================================

def make_question(
    question_text,
    document_id,
):
    """
    Build question structure expected by both
    BM25 and Vector retrieval functions.
    """

    return {
        "question": question_text,
        "gold_passages": [
            {
                "chunk_id":
                    f"{document_id}::0"
            }
        ],
    }


def display_vector_results(results):

    for result in results:

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


def display_bm25_results(results):

    for result in results:

        # Support the fields returned by
        # the existing BM25 implementation.
        score = result.get(
            "score",
            result.get("bm25_score", 0)
        )

        title = (
            f"Rank {result['rank']} "
            f"• BM25 Score "
            f"{score:.3f}"
        )

        with st.expander(title):

            pages = result.get(
                "pages",
                result.get(
                    "page_numbers",
                    []
                )
            )

            st.write(
                "**Pages:**",
                pages,
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


# =========================================================
# HEADER
# =========================================================

st.title(
    "⚖️ Legal Contract Retrieval & RAG"
)

st.write(
    "Compare lexical BM25 retrieval with "
    "semantic Vector RAG on multi-hop "
    "legal contract questions."
)


# =========================================================
# CONTRACT SELECTOR
# =========================================================

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


# =========================================================
# METHOD
# =========================================================

method = st.radio(
    "Retrieval method",
    [
        "BM25",
        "Vector RAG",
    ],
    horizontal=True,
)


# =========================================================
# QUESTION
# =========================================================

question_text = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What is the royalty rate "
        "defined in this agreement?"
    ),
    height=110,
)


top_k = st.slider(
    "Number of evidence chunks",
    min_value=1,
    max_value=10,
    value=5,
)


# =========================================================
# RUN QUERY
# =========================================================

if st.button(
    "Run Query",
    type="primary",
    use_container_width=True,
):

    if not question_text.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        question = make_question(
            question_text,
            selected_document,
        )

        # =================================================
        # BM25
        # =================================================

        if method == "BM25":

            with st.spinner(
                "Running BM25 retrieval..."
            ):

                results = retrieve_bm25(
                    question,
                    bm25_indexes,
                    top_k=top_k,
                )

            st.subheader(
                "BM25 Retrieval Results"
            )

            if not results:

                st.error(
                    "No BM25 evidence retrieved."
                )

            else:

                st.info(
                    "BM25 is a retrieval baseline. "
                    "It returns the most lexically "
                    "relevant contract evidence."
                )

                display_bm25_results(
                    results
                )

        # =================================================
        # VECTOR RAG
        # =================================================

        else:

            with st.spinner(
                "Retrieving semantic evidence..."
            ):

                results = retrieve_vector(
                    question,
                    vector_index,
                    metadata,
                    embedding_model,
                    top_k=top_k,
                    restrict_to_document=True,
                )

            if not results:

                st.error(
                    "No Vector evidence retrieved."
                )

            else:

                with st.spinner(
                    "Generating grounded answer..."
                ):

                    answer = generate_answer(
                        question_text,
                        results,
                        tokenizer,
                        generation_model,
                    )

                st.subheader(
                    "Vector RAG Answer"
                )

                st.success(
                    answer
                )

                st.subheader(
                    "Retrieved Evidence"
                )

                display_vector_results(
                    results
                )


# =========================================================
# PROJECT RESULTS
# =========================================================

st.divider()

st.subheader(
    "📊 Retrieval Evaluation"
)

st.caption(
    "Results on 100 balanced NLLP-CUAD "
    "multi-hop questions (25 per hop level)."
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### BM25"
    )

    st.metric(
        "Recall@10",
        "0.617"
    )

    st.metric(
        "MRR",
        "0.788"
    )

    st.metric(
        "Full Evidence@10",
        "20%"
    )


with col2:

    st.markdown(
        "### Vector"
    )

    st.metric(
        "Recall@10",
        "0.519"
    )

    st.metric(
        "MRR",
        "0.621"
    )

    st.metric(
        "Full Evidence@10",
        "14%"
    )


st.markdown(
    """
**Vector RAG evaluation**

- Gold Answer Match: **3%**
- Any gold evidence retrieved: **80%**
- Full gold evidence retrieved: **4%**
"""
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "BM25 lexical retrieval | "
    "Vector retrieval: all-MiniLM-L6-v2 + FAISS | "
    "Generation: FLAN-T5 Base"
)