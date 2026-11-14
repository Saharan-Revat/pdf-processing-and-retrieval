"""
Knowledge extraction: document chunks -> entities + Subject-Predicate-Object
triples, every one linked to the chunk (and page) it came from.

Two extractors are available and can be combined:

  rules  Deterministic, contract-aware patterns. No model, no network.
         Extracts parties, organisations, aliases ("Purolator"), defined
         terms and their definitions, section headings and cross-references
         ("Section 3.1"), durations / amounts / percentages / dates,
         obligations ("X shall ...") and jurisdiction of incorporation.

  llm    An instruction-tuned LLM behind any OpenAI-compatible
         /chat/completions endpoint (Ollama, vLLM, LM Studio, OpenAI, Groq,
         Together, ...). Returns typed entities and SPO triples as JSON.
         Triples whose subject/object do not occur in the chunk are dropped,
         so every kept triple is grounded in its source passage.

  both   rules + llm merged (recommended when an LLM is available). The rules
         provide the structural skeleton (sections, defined terms); the LLM
         adds semantic relations.

Usage (from repo root):

    python scripts/graph_rag/kg_extraction.py                    # rules
    python scripts/graph_rag/kg_extraction.py --extractor both   # + LLM

LLM configuration (environment variables):

    LLM_BASE_URL  default http://localhost:11434/v1   (Ollama)
    LLM_MODEL     default qwen2.5:7b-instruct
    LLM_API_KEY   default "ollama" (ignored by Ollama)

LLM responses are cached per chunk in data/knowledge_graph/llm_cache/, so an
interrupted run resumes where it stopped.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict

from kg_common import (
    CHUNKS_DIR,
    KG_DIR,
    TRIPLES_DIR,
    document_slug,
    load_chunks,
    normalize_entity,
    normalize_section,
)


# ===============================================================
# Patterns
# ===============================================================

Q_OPEN = "\"“"
Q_CLOSE = "\"”"

QUOTED_TERM = re.compile(
    r"[" + Q_OPEN + r"]\s*([A-Za-z0-9][\w&.,/'’\- ]{0,60}?)\s*[" + Q_CLOSE + r"]"
)

DEFINITION_VERB = re.compile(
    r"^\s*,?\s*(?:shall\s+)?(?:mean|means|shall mean|shall have the meaning|"
    r"has the meaning|have the meaning|refers to|shall refer to|"
    r"is defined|includes|shall include)\b",
    re.I,
)

ALIAS_PREFIX = re.compile(
    r"\(\s*(?:the|each|collectively|individually|together|jointly|"
    r"hereinafter(?:\s+referred\s+to\s+as)?|hereafter|referred\s+to\s+"
    r"(?:herein\s+)?as|herein|a|an|or|and)?[\s,]*(?:the\s+)?$",
    re.I,
)

ORG_SUFFIX = (
    r"(?:Inc|Incorporated|Corp|Corporation|LLC|L\.L\.C|Ltd|Limited|AG|GmbH|"
    r"plc|PLC|S\.A|N\.V|B\.V|L\.P|LP|LLP|Co|Company|INC|CORP|CORPORATION|"
    r"LTD|LIMITED|COMPANY)\.?"
)

ORGANIZATION = re.compile(
    r"\b((?:[A-Z][\w&.\-']*,?\s+){0,5}?[A-Z][\w&.\-']*,?\s+" + ORG_SUFFIX + r")(?=[\s,;:)\"”]|$)"
)

INCORPORATION = re.compile(
    r"\b(?:an?|the)\s+((?:State\s+of\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\s+"
    r"(?:corporation|limited\s+liability\s+company|company|limited\s+partnership|"
    r"partnership|business\s+corporation|corporation\s+organized)",
)

INCORPORATION_LAWS = re.compile(
    r"(?:incorporated|organized|organised|existing|formed)\s+under\s+the\s+laws\s+of\s+"
    r"(?:the\s+)?((?:State|Province|Commonwealth|Republic|Kingdom|Federal Republic)\s+of\s+)?"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)

NUM_WORDS = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"and|-|\s)+"
)

DURATION = re.compile(
    r"\b((?:" + NUM_WORDS + r")?\(?\d+\)?\s*(?:calendar\s+|business\s+|working\s+)?"
    r"(?:days?|months?|years?|weeks?|hours?|minutes?|quarters?))\b"
    r"|\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty-five|sixty|ninety)\s+(?:calendar\s+|business\s+)?"
    r"(?:days?|months?|years?|weeks?))\b",
    re.I,
)

MONEY = re.compile(
    r"((?:US\s*|U\.S\.\s*|CAD\s*|C)?\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:dollars|euros|EUR|USD))",
    re.I,
)

PERCENT = re.compile(
    r"(\b(?:" + NUM_WORDS + r")?\(?\d+(?:\.\d+)?\s*%\)?|\b\d+(?:\.\d+)?\s*percent\b)",
    re.I,
)

MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)

DATE = re.compile(
    r"\b(" + MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?" + MONTHS + r",?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.I,
)

SECTION_REF = re.compile(
    r"\b(?:Sections?|Articles?|Paragraphs?|Clauses?|subsections?|"
    r"sub-sections?|referred\s+(?:to\s+)?in|pursuant\s+to)\s+"
    r"((?:\d{1,2}(?:\.\d{1,2}){0,3}(?:\s*\([a-zA-Z0-9]{1,4}\))*"
    r"(?:\s*(?:,|and|or|through|to|-)\s*)?)+)",
    re.I,
)

SECTION_NUMBER = re.compile(r"\d{1,2}(?:\.\d{1,2}){0,3}")

SECTION_HEADING = re.compile(
    r"(?:^|(?<=[\s;:.]))(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?|\d{1,2}\.0|\d{1,2}\.)"
    r"\s+(?=[A-Z\"“(])"
)

SECTION_TITLE = re.compile(
    r"(?:^|(?<=[\s;:.]))(?:ARTICLE\s+)?(\d{1,2})(?:\.0)?\.?\s+"
    r"([A-Z][A-Z&,'\- ]{3,60}[A-Z])(?=\s|$)"
)

ARTICLE_HEADING = re.compile(r"\bARTICLE\s+(\d{1,2}|[IVXL]+)\b")

OBLIGATION = re.compile(
    r"\b(shall\s+not|shall|will\s+not|will|must\s+not|must|may\s+not|may|"
    r"agrees?\s+not\s+to|agrees?\s+to|is\s+entitled\s+to|has\s+the\s+right\s+to|"
    r"is\s+responsible\s+for|warrants?\s+that|represents?\s+that)\s+"
    r"((?:[\w\-'’/$%(),]+\s+){1,14}?[\w\-'’/$%()]+)(?=[.;:]|,\s+(?:and|or|provided|unless|except)\b|$)",
    re.I,
)

GENERIC_TERMS = {
    "agreement", "this agreement", "parties", "party", "section",
    "schedule", "exhibit", "article", "herein", "hereof", "hereunder",
    "including", "provided", "notwithstanding", "confidential",
}

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


ORG_LEAD_STOP = re.compile(
    r"^.*\b(?:and|between|among|by|with|agreement|copy|exhibit|dated|"
    r"to|for|from|amended|restated)\s+",
    re.I,
)

ORG_LEAD_NOISE = re.compile(
    r"^(?:(?:redacted|version|recitals|whereas|has|been|omitted|because|"
    r"third-party|beneficiary|execution|copy|a|the|by|name|title)\s+)+",
    re.I,
)

ORG_REJECT = re.compile(
    r"\b(?:any|purpose|failure|warranty|remedy|liability|shall|herein|"
    r"agreement|section|president|officer|director|state|side|our)\b",
    re.I,
)


def clean_organization(name):
    name = clean_value(name)
    if "," in name.rsplit(",", 1)[0]:
        # "Marathon, Marathon Oil, Marathon Petroleum Company" -> last item
        name = name.rsplit(",", 2)[-2].strip() + ("," + name.rsplit(",", 1)[-1] if name.count(",") else "")
    # Cut at a sentence boundary inside the match ("Effective Date. CORIO, INC")
    parts = re.split(r"(?<=[a-z]{2})\.\s+|(?<=[A-Z]{3})\.\s+(?=[A-Z])", name)
    name = parts[-1]
    name = ORG_LEAD_STOP.sub("", name).strip(" ,")
    name = ORG_LEAD_NOISE.sub("", name).strip(" ,")
    if ORG_REJECT.search(name) or len(name.split()) < 2:
        return None
    return name


def roman_to_int(text):
    total, prev = 0, 0
    for ch in reversed(text.upper()):
        val = ROMAN.get(ch, 0)
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def split_sentences(text):
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?<=[.;:!?])\s+(?=[A-Z(\"“\d])|\n{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def clean_value(text):
    return re.sub(r"\s+", " ", text).strip(" ,;.")


def obligation_predicate(modal):
    modal = re.sub(r"\s+", " ", modal.lower())

    if "not" in modal:
        return "MUST_NOT" if modal.startswith(("shall", "must", "will", "agree")) else "MAY_NOT"
    if modal.startswith(("may", "is entitled", "has the right")):
        return "MAY"
    if modal.startswith(("warrant", "represent")):
        return "WARRANTS"
    if modal.startswith("is responsible"):
        return "RESPONSIBLE_FOR"
    return "MUST"


# ===============================================================
# Rule-based extractor
# ===============================================================

class RuleExtractor:
    """
    Two passes per document:

      1. build a document vocabulary (organisations, parties, aliases,
         defined terms) from all of its chunks;
      2. extract mentions and triples chunk by chunk using that vocabulary,
         so a term defined in chunk 0 is recognised in chunk 30.
    """

    name = "rules"

    # -----------------------------------------------------------
    # Pass 1: document vocabulary
    # -----------------------------------------------------------

    def build_vocabulary(self, document_chunks):

        vocab = {}          # surface form -> type
        aliases = []        # (full name, alias, chunk_id)
        definitions = []    # (term, definition, chunk_id)

        for chunk in document_chunks:

            text = chunk["text"]

            for match in ORGANIZATION.finditer(text):
                name = clean_organization(match.group(1))
                if name and 4 <= len(name) <= 80 and normalize_entity(name):
                    vocab.setdefault(name, "Organization")

            for match in QUOTED_TERM.finditer(text):

                term = clean_value(match.group(1))
                key = normalize_entity(term)

                if not key or key in GENERIC_TERMS or len(term) < 2:
                    continue
                if not term[0].isupper() and not term[0].isdigit():
                    continue
                if len(term.split()) > 7:
                    continue

                before = text[max(0, match.start() - 60): match.start()]
                after = text[match.end(): match.end() + 40]

                if ALIAS_PREFIX.search(before):

                    target = self._alias_target(before, vocab)

                    if target and normalize_entity(target) != key:
                        vocab.setdefault(term, "Party" if vocab.get(target) == "Organization" else "DefinedTerm")
                        if vocab.get(target) == "Organization":
                            vocab[target] = "Party"
                        aliases.append((target, term, chunk["chunk_id"]))
                    else:
                        vocab.setdefault(term, "DefinedTerm")

                elif DEFINITION_VERB.search(after):

                    vocab.setdefault(term, "DefinedTerm")
                    definition = self._definition_text(text[match.end():])
                    if definition:
                        definitions.append((term, definition, chunk["chunk_id"]))

        return vocab, aliases, definitions

    @staticmethod
    def _alias_target(before, vocab):
        """
        Entity named immediately before '(the "Alias")'.
        """

        head = before[: before.rfind("(")] if "(" in before else before
        head = head.rstrip(" ,")

        best, best_end = None, -1
        for surface, kind in vocab.items():
            if kind in ("Organization", "Party"):
                pos = head.rfind(surface)
                if pos != -1 and pos + len(surface) > best_end and len(head) - (pos + len(surface)) < 70:
                    best, best_end = surface, pos + len(surface)

        if best:
            return best

        # Fallback: capitalised phrase right before the parenthesis
        match = re.search(r"((?:[A-Z][\w&.\-']*\s*){1,6})[,\s]*$", head)
        if match:
            candidate = clean_value(match.group(1))
            if candidate and normalize_entity(candidate) not in GENERIC_TERMS:
                return candidate

        return None

    @staticmethod
    def _definition_text(after):
        after = re.sub(r"\s+", " ", after)
        after = re.sub(r"^\s*,?\s*(?:shall\s+)?(?:means?|shall mean|shall have the meaning|has the meaning|have the meaning|refers to|shall refer to|is defined|includes|shall include)\s*", "", after, flags=re.I)
        stop = re.search(r"(?<=[.;])\s|\s[a-z]\)\s+[\"“]|\s\(\w\)\s+[\"“]|[\"“][A-Z][^\"”]{1,50}[\"”]\s+(?:shall\s+)?means", after)
        definition = after[: stop.start()] if stop else after[:300]
        words = definition.split()
        return " ".join(words[:40]).strip(" ,;.") if words else ""

    # -----------------------------------------------------------
    # Pass 2: chunk-level extraction
    # -----------------------------------------------------------

    def extract_document(self, document_chunks):

        vocab, aliases, definitions = self.build_vocabulary(document_chunks)

        # Longest surface forms first so "Licensed Product" beats "Product"
        surfaces = sorted(vocab, key=len, reverse=True)
        patterns = []
        for surface in surfaces:
            escaped = re.escape(surface)
            # Single-word Title-case defined terms ("Marks", "Act") are matched
            # case-sensitively; ALL-CAPS and multi-word terms are not.
            case_sensitive = (
                vocab[surface] == "DefinedTerm"
                and len(surface.split()) == 1
                and not surface.isupper()
            ) or len(surface) <= 3
            flags = 0 if case_sensitive else re.I
            patterns.append((surface, re.compile(r"(?<![\w])" + escaped + r"(?:['’]s)?(?![\w])", flags)))

        alias_by_chunk = defaultdict(list)
        for target, alias, chunk_id in aliases:
            alias_by_chunk[chunk_id].append((target, alias))

        definitions_by_chunk = defaultdict(list)
        for term, definition, chunk_id in definitions:
            definitions_by_chunk[chunk_id].append((term, definition))

        records = []

        for chunk in document_chunks:
            records.append(
                self.extract_chunk(
                    chunk,
                    vocab,
                    patterns,
                    alias_by_chunk[chunk["chunk_id"]],
                    definitions_by_chunk[chunk["chunk_id"]],
                )
            )

        return records

    def extract_chunk(self, chunk, vocab, patterns, chunk_aliases, chunk_definitions):

        text = chunk["text"]
        entities = {}
        triples = []

        def add_entity(name, kind):
            key = normalize_entity(name)
            if not key or key in GENERIC_TERMS:
                return None
            if key not in entities:
                entities[key] = {"name": clean_value(name), "type": kind}
            return key

        def add_triple(subject, s_type, predicate, obj, o_type, sentence, source="rules"):
            if not subject or not obj:
                return
            if normalize_entity(subject) == normalize_entity(obj):
                return
            add_entity(subject, s_type)
            if o_type not in ("Obligation", "Definition"):
                add_entity(obj, o_type)
            triples.append({
                "subject": clean_value(subject),
                "subject_type": s_type,
                "predicate": predicate,
                "object": clean_value(obj),
                "object_type": o_type,
                "evidence": sentence[:400],
                "source": source,
            })

        # ---- aliases and definitions found in this chunk ----------

        for target, alias in chunk_aliases:
            add_triple(target, vocab.get(target, "Party"), "ALSO_KNOWN_AS",
                       alias, vocab.get(alias, "Party"), f'{target} ("{alias}")')

        for term, definition in chunk_definitions:
            add_triple(term, "DefinedTerm", "DEFINED_AS", definition, "Definition",
                       f'"{term}" means {definition}')

        # ---- section structure ------------------------------------

        sections_defined = []
        section_titles = {}

        for match in SECTION_HEADING.finditer(text):
            preceding = text[max(0, match.start() - 14): match.start()].lower()
            if re.search(r"(section|article|paragraph|clause|in|to|and|of|,)\s*$", preceding):
                continue
            number = normalize_section(match.group(1))
            if number and number != "0":
                sections_defined.append(number)

        for match in SECTION_TITLE.finditer(text):
            number, title = match.group(1), match.group(2).strip()
            if len(title.split()) <= 8:
                section_titles.setdefault(number, title.title())
                sections_defined.append(number)

        for match in ARTICLE_HEADING.finditer(text):
            raw = match.group(1)
            number = raw if raw.isdigit() else str(roman_to_int(raw))
            sections_defined.append(number)

        for heading in chunk.get("sections", []):
            match = re.match(r"\s*(?:ARTICLE\s+)?(\d{1,2}(?:\.\d{1,2})*)\.?\s+(.+)", heading)
            if match:
                number = normalize_section(match.group(1))
                sections_defined.append(number)
                section_titles.setdefault(number, match.group(2).strip().title()[:80])

        sections_referenced = []

        for match in SECTION_REF.finditer(text):
            for number in SECTION_NUMBER.findall(match.group(1)):
                number = normalize_section(number)
                if number and number != "0":
                    sections_referenced.append(number)

        # ---- sentence-level entities and relations ----------------

        for sentence in split_sentences(text):

            mentioned = []
            occupied = []

            for surface, pattern in patterns:
                for m in pattern.finditer(sentence):
                    span = (m.start(), m.end())
                    if any(s < span[1] and span[0] < e for s, e in occupied):
                        continue
                    occupied.append(span)
                    mentioned.append((m.start(), surface, vocab[surface]))

            mentioned.sort()

            for _, surface, kind in mentioned:
                add_entity(surface, kind)

            # Values
            values = []
            for regex, kind in ((DATE, "Date"), (MONEY, "Amount"),
                                (PERCENT, "Percentage"), (DURATION, "Duration")):
                for m in regex.finditer(sentence):
                    value = clean_value(next(g for g in m.groups() if g) if m.groups() else m.group(0))
                    if kind == "Duration" and not re.search(r"\d|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|sixty|ninety", value, re.I):
                        continue
                    if len(value) >= 2:
                        values.append((m.start(), value, kind))

            predicates = {"Date": "HAS_DATE", "Amount": "HAS_AMOUNT",
                          "Percentage": "HAS_PERCENTAGE", "Duration": "HAS_DURATION"}

            for position, value, kind in values:
                subject = self._closest_subject(mentioned, position)
                if subject:
                    add_triple(subject[1], subject[2], predicates[kind], value, kind, sentence)
                else:
                    add_entity(value, kind)

            # Incorporation / jurisdiction
            for m in list(INCORPORATION.finditer(sentence)) + list(INCORPORATION_LAWS.finditer(sentence)):
                place = clean_value(m.groups()[-1])
                place = re.sub(r"^State\s+of\s+", "", place)
                subject = self._closest_subject(
                    [x for x in mentioned if x[2] in ("Organization", "Party")], m.start(), before_only=True
                )
                if subject and place and place.lower() not in ("delaware limited", "the"):
                    add_triple(subject[1], subject[2], "INCORPORATED_IN", place, "Location", sentence)

            # Obligations / rights
            for m in OBLIGATION.finditer(sentence):
                subject = self._closest_subject(
                    [x for x in mentioned if x[2] in ("Organization", "Party", "DefinedTerm")],
                    m.start(), before_only=True, max_distance=60,
                )
                if subject:
                    phrase = clean_value(m.group(2))
                    add_triple(subject[1], subject[2], obligation_predicate(m.group(1)),
                               phrase, "Obligation", sentence)

                    # Entities inside the obligation become direct relations
                    for _, surface, kind in mentioned:
                        if surface != subject[1] and re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", phrase, re.I):
                            add_triple(subject[1], subject[2], "OBLIGATION_INVOLVES", surface, kind, sentence)

            # Co-occurrence (weak relation) between named entities
            named = []
            for _, surface, kind in mentioned:
                if surface not in named:
                    named.append(surface)
            for i in range(len(named)):
                for j in range(i + 1, min(len(named), i + 4)):
                    add_triple(named[i], vocab[named[i]], "CO_OCCURS_WITH",
                               named[j], vocab[named[j]], sentence)

        return {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page_numbers": chunk.get("page_numbers", []),
            "entities": [
                {"key": key, **value} for key, value in entities.items()
            ],
            "triples": triples,
            "sections_defined": sorted(set(sections_defined), key=section_sort_key),
            "sections_referenced": sorted(set(sections_referenced), key=section_sort_key),
            "section_titles": section_titles,
        }

    @staticmethod
    def _closest_subject(mentioned, position, before_only=False, max_distance=None):

        best, best_distance = None, None

        for item in mentioned:
            start, surface, kind = item
            if before_only and start > position:
                continue
            distance = abs(position - start)
            if max_distance is not None and distance > max_distance + len(surface):
                continue
            # Prefer defined terms over generic party names at equal distance
            bias = 0 if kind == "DefinedTerm" else 5
            if best is None or distance + bias < best_distance:
                best, best_distance = item, distance + bias

        return best


def section_sort_key(number):
    return [int(p) for p in number.split(".") if p.isdigit()]


# ===============================================================
# LLM extractor
# ===============================================================

LLM_SYSTEM = """You extract a knowledge graph from legal contract text.
Return ONLY a JSON object with this schema:
{
  "entities": [{"name": str, "type": str}],
  "triples":  [{"subject": str, "predicate": str, "object": str,
                "subject_type": str, "object_type": str}]
}
Entity types: Party, Organization, Person, DefinedTerm, Product, Service,
Obligation, Right, Payment, Date, Duration, Amount, Percentage, Location,
Section, Document, Event, Other.
Predicates are UPPER_SNAKE_CASE verbs, for example: ALSO_KNOWN_AS, DEFINED_AS,
HAS_OBLIGATION, MUST, MUST_NOT, MAY, PAYS, RECEIVES, GRANTS_LICENSE_TO,
LICENSES, SUPPLIES, PROVIDES, HAS_DURATION, HAS_DATE, HAS_AMOUNT,
HAS_PERCENTAGE, TERMINATES_ON, GOVERNED_BY, INCORPORATED_IN, LOCATED_IN,
REFERS_TO_SECTION, PART_OF, NOTIFIES, INDEMNIFIES, OWNS.
Rules:
- Copy entity names exactly as written in the text (keep defined-term
  capitalisation, e.g. "Licensed Product").
- Keep values verbatim (e.g. "one hundred twenty (120) days", "15%").
- Only extract facts stated in the text. Do not infer or invent.
- At most 25 triples."""


class LLMExtractor:

    name = "llm"

    def __init__(self, base_url=None, model=None, api_key=None, timeout=180):

        import requests  # local import: only needed for the LLM path

        self.requests = requests
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        self.timeout = timeout
        self.cache_dir = KG_DIR / "llm_cache" / re.sub(r"[^\w.-]+", "_", self.model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, chunk):
        digest = hashlib.sha1((self.model + chunk["chunk_id"] + chunk["text"]).encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _call(self, text):

        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": f"CONTRACT TEXT:\n{text}\n\nJSON:"},
            ],
        }

        for attempt in range(3):
            try:
                response = self.requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except Exception as error:  # noqa: BLE001
                if attempt == 2:
                    print(f"  LLM call failed: {error}")
                    return {"entities": [], "triples": []}
                time.sleep(2 * (attempt + 1))

    def extract_document(self, document_chunks):

        records = []

        for chunk in document_chunks:

            cache = self._cache_path(chunk)

            if cache.exists():
                raw = json.loads(cache.read_text(encoding="utf-8"))
            else:
                raw = self._call(chunk["text"])
                cache.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

            records.append(ground_llm_output(chunk, raw))

        return records


def parse_json_object(content):
    content = content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return {"entities": [], "triples": []}
    try:
        return json.loads(content[start: end + 1])
    except json.JSONDecodeError:
        return {"entities": [], "triples": []}


def grounded(name, chunk_text_norm):
    key = normalize_entity(name)
    if not key:
        return False
    if key in chunk_text_norm:
        return True
    tokens = [t for t in re.findall(r"\w+", key) if len(t) > 2]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in chunk_text_norm)
    return hits / len(tokens) >= 0.75


def ground_llm_output(chunk, raw):
    """
    Keep only LLM entities / triples whose surface forms are found in the
    chunk text (hallucination filter + provenance guarantee).
    """

    text_norm = " ".join(normalize_entity(chunk["text"]).split())
    text_norm = re.sub(r"['’]s\b", "", text_norm)

    entities = {}
    triples = []

    for entity in raw.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        if name and grounded(name, text_norm):
            key = normalize_entity(name)
            entities.setdefault(key, {"key": key, "name": name, "type": str(entity.get("type", "Other"))})

    for triple in raw.get("triples", []) or []:
        if not isinstance(triple, dict):
            continue
        subject = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        predicate = re.sub(r"[^A-Z0-9_]", "_", str(triple.get("predicate", "RELATED_TO")).upper()).strip("_") or "RELATED_TO"

        if not subject or not obj or not grounded(subject, text_norm):
            continue

        o_type = str(triple.get("object_type") or entities.get(normalize_entity(obj), {}).get("type", "Other"))
        if not grounded(obj, text_norm) and o_type not in ("Obligation", "Right", "Definition"):
            continue

        s_type = str(triple.get("subject_type") or entities.get(normalize_entity(subject), {}).get("type", "Other"))

        for name, kind in ((subject, s_type), (obj, o_type)):
            if kind not in ("Obligation", "Right", "Definition"):
                key = normalize_entity(name)
                entities.setdefault(key, {"key": key, "name": name, "type": kind})

        triples.append({
            "subject": subject, "subject_type": s_type,
            "predicate": predicate,
            "object": obj, "object_type": o_type,
            "evidence": "", "source": "llm",
        })

    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "page_numbers": chunk.get("page_numbers", []),
        "entities": list(entities.values()),
        "triples": triples,
        "sections_defined": [],
        "sections_referenced": [],
        "section_titles": {},
    }


def merge_records(rule_records, llm_records):

    llm_by_chunk = {r["chunk_id"]: r for r in llm_records}
    merged = []

    for record in rule_records:
        other = llm_by_chunk.get(record["chunk_id"])
        if other:
            keys = {e["key"] for e in record["entities"]}
            record["entities"] += [e for e in other["entities"] if e["key"] not in keys]
            seen = {(normalize_entity(t["subject"]), t["predicate"], normalize_entity(t["object"])) for t in record["triples"]}
            for t in other["triples"]:
                signature = (normalize_entity(t["subject"]), t["predicate"], normalize_entity(t["object"]))
                if signature not in seen:
                    record["triples"].append(t)
                    seen.add(signature)
        merged.append(record)

    return merged


# ===============================================================
# Driver
# ===============================================================

def group_by_document(chunks):
    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk["document_id"]].append(chunk)
    for document_id in documents:
        documents[document_id].sort(key=lambda c: int(c["chunk_id"].rsplit("::", 1)[-1]))
    return documents


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractor", choices=["rules", "llm", "both"], default="rules")
    parser.add_argument("--model", default=None, help="Overrides LLM_MODEL")
    parser.add_argument("--base-url", default=None, help="Overrides LLM_BASE_URL")
    parser.add_argument("--limit-docs", type=int, default=None, help="Only process the first N documents (testing)")
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_DIR)
    documents = group_by_document(chunks)

    TRIPLES_DIR.mkdir(parents=True, exist_ok=True)

    rules = RuleExtractor()
    llm = LLMExtractor(base_url=args.base_url, model=args.model) if args.extractor in ("llm", "both") else None

    start = time.perf_counter()
    totals = defaultdict(int)

    for number, (document_id, document_chunks) in enumerate(sorted(documents.items()), start=1):

        if args.limit_docs and number > args.limit_docs:
            break

        t0 = time.perf_counter()

        if args.extractor == "rules":
            records = rules.extract_document(document_chunks)
        elif args.extractor == "llm":
            records = llm.extract_document(document_chunks)
        else:
            records = merge_records(rules.extract_document(document_chunks), llm.extract_document(document_chunks))

        n_entities = sum(len(r["entities"]) for r in records)
        n_triples = sum(len(r["triples"]) for r in records)
        totals["chunks"] += len(records)
        totals["entities"] += n_entities
        totals["triples"] += n_triples

        output = {
            "document_id": document_id,
            "extractor": args.extractor,
            "llm_model": llm.model if llm else None,
            "extraction_time_seconds": time.perf_counter() - t0,
            "chunks": records,
        }

        path = TRIPLES_DIR / f"{document_slug(document_id)}.json"
        path.write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")

        print(f"[{number}/{len(documents)}] {document_id[:70]:70s} chunks={len(records):3d} "
              f"entity-mentions={n_entities:4d} triples={n_triples:4d}")

    elapsed = time.perf_counter() - start

    (KG_DIR / "extraction_info.json").write_text(json.dumps({
        "extractor": args.extractor,
        "llm_model": llm.model if llm else None,
        "llm_base_url": llm.base_url if llm else None,
        **totals,
        "extraction_time_seconds": elapsed,
    }, indent=2), encoding="utf-8")

    print(f"\nExtracted {totals['triples']} triples from {totals['chunks']} chunks in {elapsed:.1f}s "
          f"-> {TRIPLES_DIR}")


if __name__ == "__main__":
    main()
