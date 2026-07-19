"""
FastAPI service exposing the pipeline behind the exact contract in
section 4 of the guide, frozen with Umer's backend:

  POST /ocr             { imageUrl }                                   -> { questionText }
  POST /retrieve         { questionText, subject }                      -> { chunks: [{text, page, topic}] }
  POST /socratic-reply   { questionText, chunks, studentLevel, history } -> { reply, isFinalAnswer }
  POST /score-update      { studentId, topic, correct, hintsUsed }       -> { newMasteryLevel, action }

Run locally:
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from ocr.reader import extract_question_text
from retrieval.searcher import retrieve_chunks
from socratic_engine.tutor import generate_socratic_reply
from adaptive_scoring.scorer import record_score_update, _get_history

app = FastAPI(
    title="AI / RAG Core Intelligence Layer",
    description="Socratic AI Tutor — OCR, retrieval, hint generation, adaptive scoring.",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Schemas — field names use camelCase aliases to match the API contract
# exactly, since the JSON request/response bodies are camelCase.
# ---------------------------------------------------------------------

class OcrRequest(BaseModel):
    imageUrl: str


class OcrResponse(BaseModel):
    questionText: str


class RetrieveRequest(BaseModel):
    questionText: str
    subject: str


class Chunk(BaseModel):
    text: str
    page: int | None = None
    topic: str | None = None


class RetrieveResponse(BaseModel):
    chunks: list[Chunk]


class HistoryTurn(BaseModel):
    role: str
    content: str


class SocraticReplyRequest(BaseModel):
    questionText: str
    chunks: list[Chunk]
    studentLevel: str = "intermediate"
    history: list[HistoryTurn] = Field(default_factory=list)
    sessionId: str = "default-session"  # used to route to the right Assistants thread


class SocraticReplyResponse(BaseModel):
    reply: str
    isFinalAnswer: bool


class ScoreUpdateRequest(BaseModel):
    studentId: str
    topic: str
    correct: bool
    hintsUsed: int
    currentLevel: str = "intermediate"


class ScoreUpdateResponse(BaseModel):
    newMasteryLevel: str
    action: str


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr", response_model=OcrResponse)
def ocr(req: OcrRequest):
    try:
        question_text = extract_question_text(req.imageUrl)
        return OcrResponse(questionText=question_text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR provider error: {e}")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        chunks = retrieve_chunks(req.questionText, req.subject, limit=3)
        return RetrieveResponse(chunks=[Chunk(**c) for c in chunks])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval error: {e}")


@app.post("/socratic-reply", response_model=SocraticReplyResponse)
def socratic_reply(req: SocraticReplyRequest):
    try:
        result = generate_socratic_reply(
            question_text=req.questionText,
            chunks=[c.model_dump() for c in req.chunks],
            student_level=req.studentLevel,
            history=[h.model_dump() for h in req.history],
            session_key=req.sessionId,
        )
        return SocraticReplyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Socratic engine error: {e}")


@app.post("/score-update", response_model=ScoreUpdateResponse)
def score_update(req: ScoreUpdateRequest):
    try:
        history = _get_history(req.studentId)
        result = record_score_update(
            student_id=req.studentId,
            topic=req.topic,
            correct=req.correct,
            hints_used=req.hintsUsed,
            current_level=req.currentLevel,
        )
        return ScoreUpdateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.SERVICE_PORT, reload=True)
