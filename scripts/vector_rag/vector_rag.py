from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from vector_retrieval import (
    load_vector_system,
    load_evaluation,
    retrieve_vector
)


GENERATION_MODEL = "google/flan-t5-base"

TOP_K = 5


def load_generator():

    print(
        f"Loading generation model: "
        f"{GENERATION_MODEL}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL
    )

    return tokenizer, model

def extract_relevant_window(
    text,
    question,
    tokenizer,
    window_tokens=75
):
    """
    Extract the portion of a retrieved chunk that has
    the greatest lexical overlap with the question.
    """

    sentences = text.replace("\n", " ").split(".")

    question_words = set(
        word.lower().strip(".,!?;:()[]\"'")
        for word in question.split()
        if len(word) > 2
    )

    best_sentence = None
    best_score = -1

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = set(
            word.lower().strip(".,!?;:()[]\"'")
            for word in sentence.split()
        )

        score = len(
            question_words.intersection(
                sentence_words
            )
        )

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        best_sentence = text

    tokens = tokenizer.encode(
        best_sentence,
        add_special_tokens=False,
        truncation=True,
        max_length=window_tokens
    )

    return tokenizer.decode(
        tokens,
        skip_special_tokens=True
    )

def build_context(
    retrieved_chunks,
    question,
    tokenizer
):

    context_parts = []

    for result in retrieved_chunks:

        chunk_id = result["chunk_id"]

        text = extract_relevant_window(
        result["text"],
        question,
        tokenizer,
        window_tokens=75
        )

        context_parts.append(
            f"[SOURCE: {chunk_id}]\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question,
    retrieved_chunks,
    tokenizer,
    generation_model
):

    context = build_context(
        retrieved_chunks,
        question,
        tokenizer
    )

    prompt = f"""
You are answering a question about a legal contract.

Use ONLY the evidence below.

Instructions:
- Answer only the question being asked.
- Do not answer a different question mentioned in the evidence.
- Prefer evidence that directly matches the wording of the question.
- If a duration, date, percentage, amount, entity, or other value is requested, return that value directly.
- Do not explain unrelated contract provisions.
- If the evidence is insufficient, answer exactly: Insufficient evidence.

EVIDENCE:
{context}

QUESTION:
{question}

SHORT ANSWER:
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

if __name__ == "__main__":

    # Load vector retrieval
    index, metadata, embedding_model = (
        load_vector_system()
    )

    questions = load_evaluation()

    # Same test question
    test_question = questions[0]

    print("\n" + "=" * 80)
    print("VECTOR RAG TEST")
    print("=" * 80)

    print("\nQuestion:")
    print(test_question["question"])

    print("\nGold answer:")
    print(test_question["gold_answer"])

    # Retrieve evidence
    retrieved = retrieve_vector(
        test_question,
        index,
        metadata,
        embedding_model,
        top_k=TOP_K,
        restrict_to_document=True
    )

    print("\nRetrieved evidence:")

    for result in retrieved:

        print(
            f"Rank {result['rank']}: "
            f"{result['chunk_id']} "
            f"(score={result['score']:.4f})"
        )

    # Load LLM
    tokenizer, generation_model = (
        load_generator()
    )

    # Generate grounded answer
    answer = generate_answer(
        test_question["question"],
        retrieved,
        tokenizer,
        generation_model
    )

    print("\n" + "=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)

    print(answer)

    print("\n" + "=" * 80)
    print("SOURCE ATTRIBUTION")
    print("=" * 80)

    for result in retrieved:

        print(
            f"- {result['chunk_id']} "
            f"(pages={result['pages']})"
        )