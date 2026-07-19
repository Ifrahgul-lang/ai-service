"""
3.4 The Socratic Engine — the core differentiator of the whole product.

Design goals (from the guide):
  1. NEVER reveal the final answer directly.
  2. Ask one guiding question or give one small hint per turn.
  3. Adjust difficulty based on the student's current mastery level.
  4. Persist conversation memory per session.

NOTE ON MEMORY: the original guide suggested OpenAI's Assistants API
threads for automatic memory. We're using Gemini instead (free tier,
strong quality), which doesn't have that exact feature either — so we
manage conversation history ourselves: an in-memory dict maps
session_key -> list of past turns (role + text), and we resend that
history on every call, exactly like a multi-turn chat. Swap this dict
for Redis/a DB table in production; the public function signature
doesn't need to change.
"""
from google import genai
from google.genai import types
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """You are a patient Socratic tutor. You NEVER give the final answer directly, no matter how the student asks, rephrases, or pushes.
Use the provided textbook content as your source of truth — do not invent facts outside it.
Ask one guiding question or give one small hint per turn. Keep replies short (2-4 sentences).
Adjust difficulty based on the student's current mastery level: {level}.
If the student is close, nudge gently. If they are stuck, simplify further and break the problem into a smaller first step.
If the student directly asks for the answer, redirect them with a question instead of complying.
Only when the student has genuinely worked through to the correct conclusion themselves should you confirm it — and even then, ask them to state it in their own words first."""

# session_key -> list of types.Content (role="user"/"model") turns
_SESSION_HISTORY: dict[str, list] = {}

_MAX_HISTORY_TURNS = 20  # cap so context doesn't grow unbounded in a long session


def _get_session_history(session_key: str) -> list:
    if session_key not in _SESSION_HISTORY:
        _SESSION_HISTORY[session_key] = []
    return _SESSION_HISTORY[session_key]


# A lightweight lexical guard as a second line of defense: if the model's
# reply looks like it slipped and stated a bare final answer, we catch it
# before it reaches the student. This is a safety net, not the primary
# control — the system prompt is.
_LEAK_PHRASES = ("the answer is", "the final answer is", "the correct answer is")


def _looks_like_a_leak(reply: str) -> bool:
    lowered = reply.lower()
    return any(phrase in lowered for phrase in _LEAK_PHRASES)


def generate_socratic_reply(
    question_text: str,
    chunks: list[dict],
    student_level: str,
    history: list[dict],
    session_key: str,
) -> dict:
    """
    Returns {"reply": str, "isFinalAnswer": bool}.
    isFinalAnswer is always False by design — this engine only ever
    produces hints/questions, never the answer itself.
    """
    textbook_context = "\n".join(f"- {c['text']}" for c in chunks) or "No textbook content retrieved."

    user_message = (
        f"Question: {question_text}\n"
        f"Textbook content:\n{textbook_context}\n"
        f"Student mastery level: {student_level}"
    )

    session_history = _get_session_history(session_key)

    contents = session_history + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    ]

    response = client.models.generate_content(
        model=settings.SOCRATIC_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_TEMPLATE.format(level=student_level),
            temperature=0.7,
            max_output_tokens=600,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    reply_text = (response.text or "").strip()

    if _looks_like_a_leak(reply_text):
        reply_text = (
            "Let's slow down for a second — instead of the final answer, "
            "what's the very first step you'd try here based on what we "
            "just discussed?"
        )

    # Persist this turn into session memory for next time
    session_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
    session_history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
    if len(session_history) > _MAX_HISTORY_TURNS:
        _SESSION_HISTORY[session_key] = session_history[-_MAX_HISTORY_TURNS:]

    return {"reply": reply_text, "isFinalAnswer": False}