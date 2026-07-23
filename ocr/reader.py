"""
3.1 OCR: photo -> question text

Uses Groq's vision model (Llama 3.2 Vision) to pull clean question text
out of a photographed (not scanned) textbook/homework page.

NOTE: Groq's chat completions API is OpenAI-compatible and technically
accepts a plain https image URL directly. We still fetch the image
ourselves and send it as a base64 data URI instead, for two reasons:
  1. Some image hosts (WordPress blogs, CDNs) block Groq's own fetcher
     the same way they blocked plain requests without a browser-like
     User-Agent — fetching it ourselves with a proper User-Agent header
     is more reliable than trusting the remote host to accept Groq's
     request.
  2. It gives us a single, consistent place to raise a clean, graceful
     error if the image can't be downloaded at all.
"""
import base64
import re
import httpx
from groq import Groq
from config import settings

# Safety net: strips any stray <think>...</think> block, in case the model
# still emits internal reasoning despite reasoning_effort="none".
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

client = Groq(api_key=settings.GROQ_API_KEY)

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

_FETCH_HEADERS = {
    # Many image hosts (WordPress blogs, CDNs, etc.) block requests that
    # don't look like they're coming from a browser and return 403.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def _fetch_image_bytes(image_url: str) -> tuple[bytes, str]:
    """Downloads the image and returns (bytes, mime_type)."""
    try:
        response = httpx.get(
            image_url,
            timeout=15,
            follow_redirects=True,
            headers=_FETCH_HEADERS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(
            "We couldn't load that image link — the website hosting it "
            "blocked the request. Please try downloading the image and "
            "uploading it directly, or use a different image link."
        ) from e
    except httpx.RequestError as e:
        raise ValueError(
            "We couldn't reach that image link. Please check the URL and "
            "try again, or upload the image directly."
        ) from e

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type.startswith("image/"):
        raise ValueError(
            "That link doesn't point directly to an image — it looks like "
            "it opens a webpage instead. Please use a direct image link "
            "(usually ending in .jpg, .png, etc.), or upload the photo "
            "directly."
        )
    return response.content, content_type


def extract_question_text(image_url: str) -> str:
    """
    Downloads the image at image_url and calls Groq's vision model to
    extract clean question text from it.
    """
    image_bytes, mime_type = _fetch_image_bytes(image_url)
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_data}"

    completion = client.chat.completions.create(
        model=settings.OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=1024,
        # qwen/qwen3.6-27b supports thinking/non-thinking modes. OCR is a
        # simple extraction task with no need for reasoning, and thinking
        # mode was leaking raw <think>...</think> content into the output.
        reasoning_effort="none",
    )

    choice = completion.choices[0]

    # Detect truncation explicitly instead of silently returning a cut-off answer.
    if choice.finish_reason == "length":
        raise ValueError(
            "This question is quite long, so we couldn't grab all of it in one go. "
            "No worries though — just try cropping the photo a little closer to the "
            "question, or if it has multiple parts, snap them one at a time. "
            "You'll have it sorted in a second try! 🙂"
        )

    text = (choice.message.content or "")
    text = _THINK_TAG_RE.sub("", text).strip()
    if text == "UNREADABLE" or not text:
        raise ValueError(
            "Hmm, we couldn't quite make out a question in this photo. "
            "A couple of quick tips that usually help:\n"
            "• Make sure the photo shows just the question, not notes or a full page,\n"
            "• Try to get good lighting and keep the text in focus,\n"
            "• Crop out anything extra around the question.\n\n"
            "Give it another shot — you'll get it! 🙂"
        )
    return text
