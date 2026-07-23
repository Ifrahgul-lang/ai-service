"""
3.4 The Socratic Engine — the core differentiator of the whole product.

Design goals (from the guide):
  1. NEVER reveal the final answer directly, UNLESS the student has
     already reached it themselves — then confirm it.
  2. Ask one guiding question or give one small hint per turn.
  3. Adjust difficulty based on the student's current mastery level.
  4. Evaluate the student's latest reply for correctness each turn.

NOTE ON MEMORY: per the original API contract, the backend sends the
full conversation `history` on every call — this service is stateless
and trusts that history as the single source of truth, rather than
keeping its own server-side session memory.

NOTE ON TURN STRUCTURE: the question + textbook context live in the
system message, not as a leading user turn. `messages` is built purely
from `history` (student -> "user", tutor -> "assistant") so it always
alternates correctly and the model can clearly tell which message is
the student's latest reply to evaluate. (This mirrors a fix made in
the original Gemini version, where sending the question as a separate
leading "user" turn right before the student's first reply — also
"user" — caused the two to get merged and the student's answer to be
ignored.)

NOTE ON JSON OUTPUT: Groq's strict JSON-schema structured outputs are
only available on a small set of models. `openai/gpt-oss-120b` is used
here with the more widely-supported `json_object` mode instead, which
requires the word "JSON" to appear somewhere in the prompt — the
system prompt below satisfies that.
"""
import json
from groq import Groq
from config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """You are a patient Socratic tutor.

Original question:
{question}

Relevant textbook content:
{context}

Student's current mastery level: {level}

You will be given the conversation so far as a sequence of turns alternating between the student and you (the tutor). Each turn, do this:

1. If the last message in the conversation is from the student, evaluate whether it is a genuinely correct and complete answer to the original question, based on the textbook content above. Judge the substance of what they concluded, even if they think out loud, second-guess themselves, or phrase it informally — what matters is whether their final stated conclusion is correct and complete.
2. If it IS correct and complete: warmly confirm it, briefly restate the key idea in your own words to reinforce it, and set isFinalAnswer to true.
3. If it is NOT yet correct/complete, or there is no student reply yet (this is the first turn): do NOT reveal the final answer. Ask exactly one guiding question or give one small hint that moves the student closer, and set isFinalAnswer to false.
4. Never state the final answer outright unless you are confirming the student already reached it themselves (step 2).

Adjust the difficulty of your hints based on the student's mastery level. If they are close, nudge gently. If they are stuck, simplify further and break the problem into a smaller first step.

Respond ONLY with a JSON object of exactly this shape:
{{"reply": "<your message to the student>", "isFinalAnswer": <true or false>}}"""

_LEAK_PHRASES = ("the answer is", "the final answer is", "the correct answer is")

# Used when there's no history yet, so messages isn't just a bare system
# prompt, and the model still gets a clear "give the opening hint"
# instruction as a proper user turn.
_OPENING_TURN_TEXT = (
    "(The student has not responded yet. Please give them an opening "
    "guiding question or hint to get started, following the rules above.)"
)


def _looks_like_a_leak(reply: str) -> bool:
    lowered = reply.lower()
    return any(phrase in lowered for phrase in _LEAK_PHRASES)


def _build_messages(question_text: str, chunks: list[dict], student_level: str, history: list[dict]) -> list:
    """
    System message carries the question + textbook context + rules.
    The rest of `messages` is built purely from `history`:
    role "student" -> "user", role "assistant" -> "assistant". This
    always alternates correctly and keeps the student's latest reply
    clearly identifiable as its own turn.
    """
    textbook_context = "\n".join(f"- {c['text']}" for c in chunks) or "No textbook content retrieved."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                question=question_text,
                context=textbook_context,
                level=student_level,
            ),
        }
    ]

    for turn in history:
        role = "user" if turn.get("role") == "student" else "assistant"
        content = turn.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    if len(messages) == 1:  # only the system message so far
        messages.append({"role": "user", "content": _OPENING_TURN_TEXT})

    return messages


def generate_socratic_reply(
    question_text: str,
    chunks: list[dict],
    student_level: str,
    history: list[dict],
    session_key: str,
) -> dict:
    """
    Returns {"reply": str, "isFinalAnswer": bool}. isFinalAnswer is true
    only when the student's latest reply (last entry in `history`) is
    evaluated as a correct, complete answer.
    """
    messages = _build_messages(question_text, chunks, student_level, history)

    completion = client.chat.completions.create(
        model=settings.SOCRATIC_MODEL,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw_text = completion.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_text)
        reply_text = str(parsed.get("reply", "")).strip()
        is_final_answer = bool(parsed.get("isFinalAnswer", False))
    except (json.JSONDecodeError, AttributeError, TypeError):
        reply_text = raw_text.strip() or (
            "Let's take it one step at a time — what's the first thing "
            "you'd try here based on what we've covered?"
        )
        is_final_answer = False

    if not is_final_answer and _looks_like_a_leak(reply_text):
        reply_text = (
            "Let's slow down for a second — instead of the final answer, "
            "what's the very first step you'd try here based on what we "
            "just discussed?"
        )

    return {"reply": reply_text, "isFinalAnswer": is_final_answer}