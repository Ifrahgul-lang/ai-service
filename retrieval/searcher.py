"""
3.3 Retrieval
Queries the Qdrant vector DB Abdul-haq populated. Falls back to a small
mocked dataset when the real DB isn't reachable / populated yet, so
this service can be built and demoed independently — per the "freeze
the contract, mock the rest" plan in section 4 of the guide.

NOTE: Pinecone support is commented out below (not deleted) since this
project is only using Qdrant. If we switch to Pinecone later, just
uncomment the _pinecone_search function and the pinecone branch in
retrieve_chunks(), and set VECTOR_DB=pinecone in .env.
"""
from embeddings.embedder import embed_text
from config import settings

MOCK_CHUNKS = [
    {
        "text": "Newton's Second Law states that force equals mass times "
                "acceleration (F = m * a). The net force on an object is "
                "directly proportional to its acceleration.",
        "page": 42,
        "topic": "Forces and Motion",
        "subject": "physics",
    },
    {
        "text": "Momentum is the product of an object's mass and velocity "
                "(p = m * v). In a closed system, total momentum is "
                "conserved before and after a collision.",
        "page": 58,
        "topic": "Momentum",
        "subject": "physics",
    },
    {
        "text": "Ohm's Law relates voltage, current, and resistance in a "
                "circuit: V = I * R. Resistance is measured in ohms.",
        "page": 91,
        "topic": "Electricity",
        "subject": "physics",
    },
]


def _mock_search(query_text: str, subject: str, limit: int) -> list[dict]:
    subject_matches = [c for c in MOCK_CHUNKS if c["subject"] == subject.lower()]
    pool = subject_matches or MOCK_CHUNKS
    return pool[:limit]


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def _qdrant_search(query_vector: list[float], subject: str, limit: int) -> list[dict]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant = _get_qdrant_client()
    # NOTE: newer qdrant-client versions removed .search() — use
    # .query_points() instead. Results come back as response.points.
    response = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="subject", match=MatchValue(value=subject))]
        ),
        limit=limit,
    )
    return [
        {
            "text": r.payload.get("text", ""),
            "page": r.payload.get("page"),
            "topic": r.payload.get("topic"),
        }
        for r in response.points
    ]


# ---------------------------------------------------------------------
# PINECONE SUPPORT — commented out, not deleted. Only Qdrant is used in
# this project right now. If Abdul-haq's data ends up in Pinecone
# instead, uncomment this function AND the pinecone branch inside
# retrieve_chunks() below, then set VECTOR_DB=pinecone in .env.
# ---------------------------------------------------------------------
# def _pinecone_search(query_vector: list[float], subject: str, limit: int) -> list[dict]:
#     from pinecone import Pinecone
#
#     pc = Pinecone(api_key=settings.PINECONE_API_KEY)
#     index = pc.Index(settings.PINECONE_INDEX)
#     results = index.query(
#         vector=query_vector,
#         top_k=limit,
#         filter={"subject": {"$eq": subject}},
#         include_metadata=True,
#     )
#     return [
#         {
#             "text": m["metadata"].get("text", ""),
#             "page": m["metadata"].get("page"),
#             "topic": m["metadata"].get("topic"),
#         }
#         for m in results.get("matches", [])
#     ]


def retrieve_chunks(question_text: str, subject: str, limit: int = 3) -> list[dict]:
    """
    Returns up to `limit` textbook chunks relevant to question_text,
    filtered by subject. Tries Qdrant first; if that fails (not
    populated yet, connection error, etc.) it transparently falls back
    to the mock dataset so the pipeline keeps working end-to-end.
    """
    try:
        query_vector = embed_text(question_text)

        # --- Pinecone branch (commented out — Qdrant only for now) ---
        # if settings.VECTOR_DB == "pinecone":
        #     chunks = _pinecone_search(query_vector, subject, limit)
        # else:
        #     chunks = _qdrant_search(query_vector, subject, limit)
        chunks = _qdrant_search(query_vector, subject, limit)

        if chunks:
            return chunks
        # Real DB reachable but returned nothing (e.g. not populated yet)
        return _mock_search(question_text, subject, limit)

    except Exception as e:
        print(f"[retrieval] Falling back to mock data — vector DB error: {e}")
        return _mock_search(question_text, subject, limit)