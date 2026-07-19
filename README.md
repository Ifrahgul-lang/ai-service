# AI / RAG Core Intelligence Layer — Socratic AI Tutor

Implements the full pipeline from the guide: **photo → OCR → embed →
retrieve → Socratic hint → adaptive scoring**, exposed as a FastAPI
service behind the exact contract frozen with Umer's backend.

## 1. Setup

```bash
cd ai-service
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your real GEMINI_API_KEY
```

If you don't have Qdrant running yet, that's fine — `/retrieve` will
automatically fall back to a small built-in mock dataset (see
`retrieval/searcher.py`) so you can build and demo end-to-end without
waiting on Abdul-haq's vector DB.

To run Qdrant locally in the meantime:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 2. Run the service

```bash
uvicorn main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## 3. Test each endpoint

**Health check**
```bash
curl http://localhost:8000/health
```

**OCR** (needs a real image URL or a base64 data URL)
```bash
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/homework-photo.jpg"}'
```

**Retrieve** (works immediately via mock fallback, no DB needed)
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"questionText": "What is Newtons second law?", "subject": "physics"}'
```

**Socratic reply**
```bash
curl -X POST http://localhost:8000/socratic-reply \
  -H "Content-Type: application/json" \
  -d '{
        "questionText": "A 2kg block is pushed with 10N force. Find acceleration.",
        "chunks": [{"text": "F = m * a", "page": 42, "topic": "Forces and Motion"}],
        "studentLevel": "intermediate",
        "history": [],
        "sessionId": "student123-session1"
      }'
```

**Score update**
```bash
curl -X POST http://localhost:8000/score-update \
  -H "Content-Type: application/json" \
  -d '{"studentId": "student123", "topic": "Forces and Motion", "correct": true, "hintsUsed": 1}'
```

## 4. How each guide requirement is covered

| Deliverable | Where |
|---|---|
| OCR extracts clean question text | `ocr/reader.py`, GPT-4o-mini Vision, raises a clear error on unreadable images |
| Retrieval returns relevant chunks | `retrieval/searcher.py`, same embedding model as ingestion, subject-filtered, mock fallback so it never blocks the demo |
| Socratic engine never leaks the answer | `socratic_engine/tutor.py` — strict system prompt + a lexical safety-net that intercepts any reply that slips and states "the answer is..." |
| Full pipeline runs in a few seconds | Each stage is a separate, independently-testable function; no unnecessary round-trips |
| Adaptive scoring logs every decision | `adaptive_scoring/scorer.py` writes one JSON line per decision to `adaptive_scoring/decision_log.jsonl` — this is your future ML training data |

## 5. Notes on "real" vs mocked behavior

- **OCR and Socratic reply always call the real Google Gemini API** — there is
  no mocking there, so you need a valid `GEMINI_API_KEY` for those two
  endpoints to return real results.
- **Retrieval** calls the real vector DB first and only falls back to
  mock chunks if Qdrant/Pinecone is unreachable or empty — so once
  Abdul-haq's DB is populated, real textbook content flows through
  automatically with no code changes needed.
- **Score update** is intentionally rule-based (not ML) per the guide,
  since there's no training data yet — but every decision is logged so
  a real model can be trained later on real usage data.

## 6. Project structure

```
ai-service/
  main.py                        # FastAPI app, the 4 contracted endpoints
  config.py                      # loads .env, shared settings
  ocr/reader.py                  # 3.1 photo -> question text
  embeddings/embedder.py         # 3.2 text -> vector
  retrieval/searcher.py          # 3.3 vector DB search + mock fallback
  socratic_engine/tutor.py       # 3.4 hint generation with thread memory
  adaptive_scoring/scorer.py     # 3.5 rule-based difficulty + logging
  requirements.txt
  .env.example
```
