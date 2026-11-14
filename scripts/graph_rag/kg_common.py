"""
Shared paths and helpers for the Knowledge Graph / Graph RAG pipeline.

All scripts are run from the repository root, e.g.

    python scripts/graph_rag/kg_extraction.py
"""

import re
import sys
import unicodedata
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
BM25_DIR = SCRIPTS_DIR / "bm25"

if str(BM25_DIR) not in sys.path:
    sys.path.insert(0, str(BM25_DIR))

from bm25_retrieval import (  # noqa: E402
    CHUNKS_DIR,
    EVALUATION_DIR,
    load_chunks,
    load_evaluation,
    build_bm25_indexes,
    retrieve_bm25,
    get_question_document,
    find_best_gold_mapping,
)

KG_DIR = Path("data/knowledge_graph")
TRIPLES_DIR = KG_DIR / "triples"
GRAPH_FILE = KG_DIR / "graph.json"
GRAPH_STATS_FILE = KG_DIR / "graph_stats.json"
NEO4J_EXPORT_DIR = KG_DIR / "neo4j_import"

RESULTS_DIR = Path("results/graph")

MAPPING_THRESHOLD = 0.50


# ---------------------------------------------------------------
# Entity normalization
# ---------------------------------------------------------------

_QUOTES = "\"'“”‘’`"
_LEADING = re.compile(r"^(the|a|an|this|such|said|each|any|all)\s+", re.I)
_TRAILING_PUNCT = re.compile(r"[\s,;:.()\[\]]+$")
_CORP_SUFFIX = re.compile(
    r"[,\s]+(inc|incorporated|corp|corporation|llc|l\.l\.c|ltd|limited|"
    r"co|company|plc|ag|gmbh|s\.a|l\.p|lp)\.?$",
    re.I,
)


def normalize_entity(name):
    """
    Canonical key for an entity name.

    Lower-cases, removes quotes, leading articles, possessives,
    trailing punctuation and corporate suffixes so that
    "Dynamex Inc.", "DYNAMEX" and "Dynamex's" share one node.
    """

    if not name:
        return ""

    text = unicodedata.normalize("NFKC", str(name))
    text = text.strip().strip(_QUOTES).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"['’]s\b", "", text)
    text = _TRAILING_PUNCT.sub("", text)
    text = _LEADING.sub("", text)

    previous = None
    while previous != text:
        previous = text
        text = _CORP_SUFFIX.sub("", text).strip()

    text = text.strip(_QUOTES).strip()

    return text.lower()


def normalize_section(ref):
    """
    "Section 3.1(v)" -> "3.1", "ARTICLE 7" -> "7", "17.1(c)" -> "17.1".
    """

    match = re.search(r"(\d+(?:\.\d+)*)", ref)

    if not match:
        return ""

    number = match.group(1).rstrip(".")
    number = re.sub(r"\.0$", "", number)
    return number


def normalize_question(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^\w\s.%$]", " ", text)
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def document_slug(document_id):
    return re.sub(r"[^\w.-]+", "_", document_id)[:150]
