"""
Daily Instagram poster - OpenAI (caption + 4 images) + Zernio (publishing).

Pipeline:
  1. Read the theme prompt from prompt.txt
  2. Ask GPT-4o to pick ONE concept from the reference book list and produce
     4 slide prompts + caption (brand-matched, logo watermark only)
  3. Generate 4 images with gpt-image-2 (sequential calls, 1 image per call)
  4. Save images at 1080x1350 (4:5) — native Instagram portrait size, no cropping
  5. Upload all 4 images to Zernio's media storage (presigned URL flow)
  6. Publish as a 4-slide Instagram carousel post via Zernio's posts API

Required environment variables (set as GitHub repo secrets):
  OPENAI_API_KEY      - OpenAI API key
  ZERNIO_API_KEY      - Zernio API key
  ZERNIO_ACCOUNT_ID   - Zernio's account ID for your connected Instagram account

Files expected in the repo root:
  prompt.txt          - Optional extra context / focus area for today's post
  logo.png            - Your Neuro Reset Studio logo
  sample.png          - (Optional) A sample post for extra style reference

Note: gpt-image-2 requires OpenAI API Organization Verification.
      Complete it at: https://platform.openai.com/settings/organization/general
"""

import base64
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ZERNIO_BASE_URL = "https://zernio.com/api/v1"
PROMPT_FILE     = Path("prompt.txt")
SAMPLE_FILE     = Path("sample.png")
LOGO_FILE       = Path("logo.png")
POSTS_DIR       = Path("posts")
IMAGES_PER_POST = 4

# Reference books — GPT-4o picks ONE concept from these per daily carousel
BOOK_LIST = """
1.  Gut & Mind Connection — Dr. Imran Mayor
2.  The Good Gut — Dr. Justin Sonnenburg & Dr. Erica Sonnenburg
3.  Super Gut — William Davis
4.  Gut — Giulia Enders
5.  10% Human — Alanna Collen
6.  Brain Maker — Dr. David Perlmutter
7.  Grain Brain — Dr. David Perlmutter
8.  Wheat Belly — William Davis
9.  Reset Factor — Dr. Mindy Pelz
10. Fiber Fueled — Dr. Will Bulsiewicz
11. Eat to Beat Disease — Dr. William Li
12. How Not to Die — Michael Greger
13. Obesity Code / Diabetes Code / Cancer Code — Dr. Jason Fung
14. Ultra Mind Solution — Mark Hyman
"""

# Core brand DNA injected into every image prompt
BRAND_STYLE = (
    "Visual style: deep dark navy and rich dark-purple cosmic background, "
    "luminous golden-amber accents and fine gold filigree details, "
    "soft purple and violet nebula-like light blooms, "
    "subtle star fields and ethereal glow particles, "
    "cinematic spiritual-mystical atmosphere, "
    "photorealistic yet dreamlike rendering. "
    "This must feel like it belongs to the same brand world as the Neuro Reset Studio logo: "
    "a glowing purple brain-tree inside concentric gold rings on a dark cosmic background."
)


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def read_theme_prompt() -> str:
    if not PROMPT_FILE.exists():
        return ""  # prompt.txt is now optional — books drive the content
    theme = PROMPT_FILE.read_text().strip()
    return theme


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_image_prompts_and_caption(theme: str, client: OpenAI) -> tuple[list[str], str]:
    """
    Ask GPT-4o to:
      1. Pick ONE compelling concept from the reference book list
      2. Build a 4-slide carousel teaching that concept in depth
      3. Return 4 image prompts + 1 Instagram caption as JSON
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
    content: list = []

    # Attach logo so GPT-4o understands its placement role
    if LOGO_FILE.exists():
        content.append({
            "type": "text",
            "text": (
                "This is the Neuro Reset Studio logo: "
                "a purple glowing brain shaped like a tree with a golden root trunk, "
                "enclosed in concentric gold rings, on a dark cosmic purple background.\n\n"
                "LOGO PLACEMENT RULE — critical: "
                "The logo must NEVER dominate or compete with the main content and must be consistent. "
                "The content must sell itself. Place the logo ONLY as a small, subtle watermark "
                "in one of these positions: bottom-centre, bottom-left, bottom-right, or top-left. "
                "It should look like a professional editorial watermark — tasteful, unobtrusive. "
                "It should be named Neuro Reset Studio"
                "The viewer's eye must land on the concept and message FIRST, logo second."
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(LOGO_FILE)}"},
        })
    else:
        print("Warning: logo.png not found — logo will be described from text only.")

    # Attach sample post for style reference
    if SAMPLE_FILE.exists():
        content.append({
            "type": "text",
            "text": (
                "Here is a sample post from my Instagram feed. "
                "Match its visual style, colour palette, mood, and composition exactly."
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_b64(SAMPLE_FILE)}"},
        })

    optional_focus = f"\nExtra focus from prompt.txt: {theme}\n" if theme else ""

    content.append({
        "type": "text",
        "text": (
            f"Today's date: {today}\n"
            f"{optional_focus}\n"
            f"Brand visual style to strictly follow:\n{BRAND_STYLE}\n\n"

            "=== YOUR REFERENCE BOOK LIST ===\n"
            f"{BOOK_LIST}\n"

            "=== YOUR TASK ===\n"
            "Pick ONE specific, fascinating concept from any of these books — "
            "something rooted in gut health, brain-gut connection, microbiome science, "
            "metabolic health, or nutrition neuroscience. "
            "Choose a concept that is genuinely surprising, educational, and actionable. "
            "Then build a 4-slide Instagram carousel that teaches this concept in depth.\n\n"

            "Respond with ONLY raw JSON (no markdown fences, no commentary) "
            "in exactly this shape:\n"
            '{"book": "Book Title — Author", "concept": "One-line concept summary", '
            '"image_prompts": ["...", "...", "...", "..."], "caption": "..."}\n\n'

            "=== 4-SLIDE NARRATIVE ARC ===\n\n"

            "SLIDE 1 — THE HOOK (Cinematic concept reveal)\n"
            "A breathtaking photorealistic cosmic scene that visually REPRESENTS the concept "
            "(e.g. for 'gut bacteria controlling mood': glowing neural connections flowing from "
            "an illuminated gut upward into a radiant brain, cosmic purple-gold palette). "
            "The scene itself communicates the idea — not just pretty, but meaningful. "
            "Overlay: bold glowing golden headline (7-10 words) at top — this is the scroll-stopper. "
            "Below it: 1 line of soft white text teasing WHY this matters. "
            "Neuro Reset Studio logo as tiny watermark at bottom-centre.\n\n"

            "SLIDE 2 — THE SCIENCE (What's actually happening inside your body)\n"
            "Infographic-style image, same dark cosmic brand palette "
            "(deep navy/purple bg, gold headlines, white body text, purple accent icons). "
            "Explain the SCIENCE behind the concept clearly: what organ/system is involved, "
            "what the research says, a key statistic or finding from the book. "
            "Layout: bold title at top, then 3-4 short fact-based sections with glowing icons, "
            "a source attribution at bottom in small italic gold text (e.g. 'Source: Brain Maker — Dr. Perlmutter'). "
            "Neuro Reset Studio logo watermark at top-left.\n\n"

            "SLIDE 3 — THE PROBLEM (Why most people are unknowingly harming themselves)\n"
            "Split between a moody cinematic visual (left or top half) showing the "
            "negative state (e.g. inflamed gut, foggy brain, disrupted microbiome visualised cosmically) "
            "and an infographic panel (right or bottom half) listing 4-5 common everyday habits "
            "or foods that damage this system — written as short, punchy lines in white/gold text. "
            "Headline: 'Are You Doing This? ❌' or equivalent hook. "
            "Neuro Reset Studio logo watermark at bottom-right.\n\n"

            "SLIDE 4 — THE SOLUTION (Practical steps to reset and heal)\n"
            "Bright, hopeful energy — slightly warmer golden glow compared to previous slides. "
            "Clean infographic layout: bold headline 'How to Reset ✨' at top, "
            "then 4-5 specific, actionable steps drawn from the book's recommendations "
            "(e.g. specific foods, habits, timings, techniques). "
            "Each step has a small glowing icon. "
            "At the bottom: one motivational closing line in italic gold. "
            "Neuro Reset Studio logo watermark at bottom-centre, slightly larger than other slides "
            "(this is the final brand impression slide).\n\n"

            "=== UNIVERSAL RULES FOR ALL 4 PROMPTS ===\n"
            "  • Each prompt: vivid, specific English-language text-to-image instruction "
            "    under 450 characters, written for gpt-image-2.\n"
            "  • All 4 slides must share the same brand palette but have DISTINCT compositions "
            "    and energy — the carousel must feel like a journey, not repetition.\n"
            "  • All text overlaid on images must be legible: high contrast, bold for headlines, "
            "    placed on dark/clean areas of the background — never on busy regions.\n"
            "  • Logo is always a small watermark in a corner — NEVER a focal point.\n"
            "  • Content is king. Every slide must deliver real value the viewer wants to save.\n\n"

            "caption: 3-4 sentence Instagram caption. "
            "Open with a curiosity hook, briefly explain what they'll learn in this carousel, "
            "invite them to save it and share with someone who needs this. "
            "End with 'DM Reset ✨' CTA and 6-8 relevant hashtags. "
            "Tone: warm, trustworthy, educational — like a brilliant friend who happens to be a doctor."
        ),
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=2000,  # increased for 4 detailed prompts
    )

    raw = response.choices[0].message.content

    if raw is None:
        raise RuntimeError(
            f"GPT-4o returned None. Finish reason: {response.choices[0].finish_reason}"
        )

    data = json.loads(raw)

    image_prompts = data.get("image_prompts")
    caption       = data.get("caption")
    book          = data.get("book", "unknown")
    concept       = data.get("concept", "unknown")

    if not image_prompts or len(image_prompts) != IMAGES_PER_POST:
        raise RuntimeError(
            f"Expected {IMAGES_PER_POST} image prompts, got: {image_prompts}"
        )
    if not caption:
        raise RuntimeError(f"Missing caption in GPT-4o response: {data}")

    print(f"📖 Book: {book}")
    print(f"💡 Concept: {concept}")

    return image_prompts, caption


def generate_image(image_prompt: str, out_path: Path, client: OpenAI) -> None:
    """
    Generate a single image with gpt-image-2 at 1024x1280 (native 4:5 portrait).

    1024x1280 is the exact Instagram portrait ratio (4:5) and sits well within
    gpt-image-2's pixel budget — no cropping needed, zero detail lost.

    gpt-image-2 params:
      model         : "gpt-image-2"
      size          : "1024x1280"  — native Instagram 4:5 portrait, no post-processing
      quality       : "medium"     — good balance of quality and cost (~$0.04/image)
      output_format : "png"        — lossless, returned as base64
      n             : 1            — one image per call (called sequentially)
    """
    response = client.images.generate(
        model="gpt-image-2",
        prompt=image_prompt,
        size="1024x1280",
        quality="medium",
        output_format="png",
        n=1,
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    print(f"  Saved 1024x1280 (native Instagram 4:5): {out_path}")


def upload_image_to_zernio(image_path: Path, api_key: str, max_retries: int = 3) -> str:
    """
    Upload an image via Zernio's presigned upload flow and return its public URL.

    Retries up to max_retries times on timeout or transient errors,
    with exponential backoff. PNG files at 1024x1280 can be 3-6 MB,
    so upload timeouts are set generously.
    """
    import time

    content_type = "image/png"
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    # Scale timeout: minimum 120s, +30s per MB above 2 MB
    upload_timeout = max(120, int(120 + max(0, file_size_mb - 2) * 30))
    print(f"  File size: {file_size_mb:.1f} MB — upload timeout: {upload_timeout}s")

    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: get presigned URL
            presign = requests.post(
                f"{ZERNIO_BASE_URL}/media/presign",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"filename": image_path.name, "contentType": content_type},
                timeout=30,
            )
            presign.raise_for_status()
            presign_data = presign.json()

            # Step 2: stream-upload the file to the presigned URL
            with open(image_path, "rb") as f:
                upload = requests.put(
                    presign_data["uploadUrl"],
                    data=f,                          # stream, not f.read() — avoids RAM spike
                    headers={
                        "Content-Type": content_type,
                        "Content-Length": str(image_path.stat().st_size),
                    },
                    timeout=upload_timeout,
                )
            upload.raise_for_status()
            return presign_data["publicUrl"]

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Upload failed after {max_retries} attempts for {image_path.name}: {e}"
                ) from e
            wait = 2 ** attempt  # 2s, 4s, 8s …
            print(f"  ⚠️  Upload attempt {attempt} timed out — retrying in {wait}s...")
            time.sleep(wait)


def post_to_instagram(
    image_urls: list[str], caption: str, account_id: str, api_key: str
) -> dict:
    """Publish a 4-slide carousel post to Instagram via Zernio."""
    media_items = [{"type": "image", "url": url} for url in image_urls]

    response = requests.post(
        f"{ZERNIO_BASE_URL}/posts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "content": caption,
            "mediaItems": media_items,
            "platforms": [{"platform": "instagram", "accountId": account_id}],
            "publishNow": True,
        },
        timeout=60,
    )
    print(f"Zernio response status: {response.status_code}")
    print(f"Zernio response body:   {response.text}")
    response.raise_for_status()
    return response.json()


def main() -> None:
    openai_key        = env("OPENAI_API_KEY")
    zernio_key        = env("ZERNIO_API_KEY")
    zernio_account_id = env("ZERNIO_ACCOUNT_ID")

    POSTS_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=openai_key)

    theme = read_theme_prompt()
    if theme:
        print(f"Extra focus: {theme}")

    image_prompts, caption = generate_image_prompts_and_caption(theme, client)
    print(f"\nCaption:\n{caption}\n")

    today_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    image_urls = []

    for i, prompt in enumerate(image_prompts, start=1):
        print(f"\nGenerating slide {i}/{IMAGES_PER_POST}...")
        print(f"  Prompt: {prompt}")
        image_path = POSTS_DIR / f"{today_str}_{i}.png"
        generate_image(prompt, image_path, client)
        print(f"  Saved:  {image_path}")

        url = upload_image_to_zernio(image_path, zernio_key)
        print(f"  Uploaded: {url}")
        image_urls.append(url)

    result = post_to_instagram(image_urls, caption, zernio_account_id, zernio_key)
    print(f"\n✅ Posted 4-slide carousel to Instagram: {result}")


if __name__ == "__main__":
    main()
