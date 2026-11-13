def calculate_retrieval_metrics(
    retrieved_ids,
    relevant_ids,
    k
):
    """
    Calculate retrieval metrics for any retrieval system.

    Parameters
    ----------
    retrieved_ids:
        Ranked list of retrieved project chunk IDs.

    relevant_ids:
        Set of project chunk IDs considered relevant.

    k:
        Number of retrieved results to evaluate.

    Returns
    -------
    Dictionary containing Precision@K,
    Recall@K, Reciprocal Rank and
    Full Evidence Recovery.
    """

    top_k = retrieved_ids[:k]

    relevant_in_top_k = [
        chunk_id
        for chunk_id in top_k
        if chunk_id in relevant_ids
    ]

    precision = (
        len(relevant_in_top_k) / k
    )

    recall = (
        len(set(relevant_in_top_k))
        / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    full_evidence_recovery = (
        1
        if relevant_ids
        and relevant_ids.issubset(set(top_k))
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "evidence_recovery":
            full_evidence_recovery
    }