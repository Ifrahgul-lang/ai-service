"""
3.1 OCR: photo -> question text
Uses Gemini's vision model to pull clean question text out of a
photographed (not scanned) textbook/homework page.

NOTE: unlike OpenAI/Groq, the Gemini Developer API does not accept a
plain https image URL directly in the request — it needs the image
bytes (or a pre-uploaded file). So we fetch the image from the given
URL ourselves and pass the raw bytes to Gemini.
"""
import httpx
from google import genai
from google.genai import types
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

EXTRACTION_PROMPT = (
    "You are an OCR engine specialized in student homework photos. "
    "Extract ONLY the question text from this image — no preamble, no "
    "commentary, no page numbers, no answer choices unless they are part "
    "of a multiple-choice question. If the image contains multiple "
    "questions, extract only the one that is circled, underlined, or "
    "otherwise marked; if none is marked, extract the first complete "
    "question. If you cannot read the image clearly, respond with "
    "exactly: UNREADABLE"
)


def _fetch_image_bytes(image_url: str) -> tuple[bytes, str]:
    """Downloads the image and returns (bytes, mime_type)."""
    response = httpx.get(image_url, timeout=15, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    if not content_type.startswith("image/"):
        content_type = "image/jpeg"  # sane fallback
    return response.content, content_type


def extract_question_text(image_url: str) -> str:
    """
    Downloads the image at image_url and calls Gemini's vision model to
    extract clean question text from it.
    """
    image_bytes, mime_type = _fetch_image_bytes(image_url)

    response = client.models.generate_content(
        model=settings.OCR_MODEL,
        contents=[
            EXTRACTION_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=500),
    )

    text = (response.text or "").strip()

    if text == "UNREADABLE" or not text:
        raise ValueError(
            "OCR could not extract a question from this image. "
            "Ask the student to retake the photo with better lighting/focus."
        )

    return text
