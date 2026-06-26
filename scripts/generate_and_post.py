"""
Daily Instagram poster - OpenAI (caption + 2 images) + Zernio (publishing).

Pipeline:
  1. Read the theme prompt from prompt.txt
  2. Ask GPT-4o for today's image prompt + caption (brand-matched, logo-included)
  3. Generate 2 images with gpt-image-1.5 (sequential calls, 1 image per call)
  4. Crop each image to 1024x1280 (4:5) for Instagram compatibility
  5. Upload both images to Zernio's media storage (presigned URL flow)
  6. Publish as an Instagram carousel post via Zernio's posts API

Required environment variables (set as GitHub repo secrets):
  OPENAI_API_KEY      - OpenAI API key
  ZERNIO_API_KEY      - Zernio API key
  ZERNIO_ACCOUNT_ID   - Zernio's account ID for your connected Instagram account

Files expected in the repo root:
  prompt.txt          - Your theme / topic
  logo.png            - Your Neuro Reset Studio logo
  sample.png          - (Optional) A sample post for extra style reference
"""

import base64
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from PIL import Image

ZERNIO_BASE_URL = "https://zernio.com/api/v1"
PROMPT_FILE     = Path("prompt.txt")
SAMPLE_FILE     = Path("sample.png")
LOGO_FILE       = Path("logo.png")
POSTS_DIR       = Path("posts")
IMAGES_PER_POST = 2

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
        raise RuntimeError("prompt.txt not found. Create it with your theme/topic.")
    theme = PROMPT_FILE.read_text().strip()
    if not theme:
        raise RuntimeError("prompt.txt is empty. Add a theme/topic to it.")
    return theme


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_image_prompts_and_caption(theme: str, client: OpenAI) -> tuple[list[str], str]:
    """
    Ask GPT-4o (with logo + optional sample image as vision input) to produce:
      - 2 distinct image prompts and two beautiful quotes from the books in the prompt,
        each matching the brand style and explicitly instructing the model to include the logo
      - 1 Instagram caption having CTA ,"DM Reset ✨" covering both images
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
    content: list = []

    # Attach logo so GPT-4o can describe it precisely in the prompts
    if LOGO_FILE.exists():
        content.append({
            "type": "text",
            "text": (
                "This is the Neuro Reset Studio logo. "
                "Every image prompt you write MUST instruct the image model to include "
                "this logo visibly and naturally within the scene — for example embedded "
                "in a glowing cosmic background, floating as a holographic emblem, "
                "or integrated as a mystical centrepiece. "
                "Describe the logo accurately in the prompt so the model can render it: "
                "a purple glowing brain shaped like a tree with golden root trunk, "
                "enclosed in concentric gold rings, on a dark cosmic purple background."
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
                "Match its visual style, colour palette, mood, and composition."
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_b64(SAMPLE_FILE)}"},
        })

    content.append({
        "type": "text",
        "text": (
            f"Theme for my Instagram account: {theme}\n"
            f"Today's date: {today}\n\n"
            f"Brand visual style to strictly follow:\n{BRAND_STYLE}\n\n"
            f"Create ONE Instagram carousel post with {IMAGES_PER_POST} slides for today.\n"
            "Respond with ONLY raw JSON (no markdown fences, no commentary) "
            "in exactly this shape:\n"
            '{"image_prompts": ["...", "..."], "caption": "..."}\n\n'
            "image_prompts: a JSON array of exactly 2 strings. Each is a vivid, specific, "
            "English-language text-to-image prompt (under 400 characters) for gpt-image-1.5. "
            "Rules for each prompt:\n"
            "  • Must match the brand visual style (dark cosmic, gold accounts, purple glows).\n"
            "  • Must explicitly include the Neuro Reset Studio logo as described above — "
            "    place it naturally in the scene (glowing emblem, holographic, centrepiece, etc).\n"
            "  • The 2 prompts must be visually distinct — different scenes, compositions, "
            "    or moments — so the carousel feels varied, not repetitive.\n"
            "  • Each image must include a short powerful quote or thought (5-10 words) "
            "    from the books mentioned in the theme, rendered as elegant glowing golden text "
            "    overlaid naturally on the image — like a mystical inscription or illuminated scripture.\n\n"
            "caption: one engaging Instagram caption (2-4 sentences) that works for both "
            "slides, followed by a line of 5-8 relevant hashtags. "
            "Tone: inspiring, thoughtful, aligned with neuroscience/mindset/personal growth."
        ),
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=1000,
    )

    raw = response.choices[0].message.content

    if raw is None:
        raise RuntimeError(
            f"GPT-4o returned None. Finish reason: {response.choices[0].finish_reason}"
        )

    data = json.loads(raw)

    image_prompts = data.get("image_prompts")
    caption = data.get("caption")

    if not image_prompts or not caption:
        raise RuntimeError(f"Unexpected JSON structure from GPT-4o: {data}")

    return image_prompts, caption


def generate_image(image_prompt: str, out_path: Path, client: OpenAI) -> None:
    """
    Generate a single image with gpt-image-1.5 at 1024x1536, then crop
    to 1024x1280 (4:5 ratio) which is the max portrait Instagram accepts.
    """
    response = client.images.generate(
        model="gpt-image-1.5",
        prompt=image_prompt,
        size="1024x1536",  # Closest valid portrait size for gpt-image-1.5
        quality="medium",
        n=1,
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)
    with open(out_path, "wb") as f:
        f.write(image_bytes)

    # Crop to 1024x1280 (4:5) — Instagram's max portrait ratio
    img = Image.open(out_path)
    cropped = img.crop((0, 0, 1024, 1280))
    cropped.save(out_path)
    print(f"Cropped to 1024x1280 for Instagram: {out_path}")


def upload_image_to_zernio(image_path: Path, api_key: str) -> str:
    """Upload an image via Zernio's presigned upload flow and return its public URL."""
    content_type = "image/png"

    presign = requests.post(
        f"{ZERNIO_BASE_URL}/media/presign",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"filename": image_path.name, "contentType": content_type},
        timeout=30,
    )
    presign.raise_for_status()
    presign_data = presign.json()

    with open(image_path, "rb") as f:
        upload = requests.put(
            presign_data["uploadUrl"],
            data=f.read(),
            headers={"Content-Type": content_type},
            timeout=60,
        )
    upload.raise_for_status()

    return presign_data["publicUrl"]


def post_to_instagram(
    image_urls: list[str], caption: str, account_id: str, api_key: str
) -> dict:
    """Publish a carousel post (multiple images) to Instagram via Zernio."""
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
    print(f"Zernio response body: {response.text}")
    response.raise_for_status()
    return response.json()


def main() -> None:
    openai_key        = env("OPENAI_API_KEY")
    zernio_key        = env("ZERNIO_API_KEY")
    zernio_account_id = env("ZERNIO_ACCOUNT_ID")

    POSTS_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=openai_key)

    theme = read_theme_prompt()
    print(f"Theme: {theme}")

    image_prompts, caption = generate_image_prompts_and_caption(theme, client)
    print(f"Caption: {caption}")

    today_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    image_urls = []

    for i, prompt in enumerate(image_prompts, start=1):
        print(f"\nGenerating image {i}/{IMAGES_PER_POST}...")
        print(f"Prompt: {prompt}")
        image_path = POSTS_DIR / f"{today_str}_{i}.png"
        generate_image(prompt, image_path, client)
        print(f"Saved: {image_path}")

        url = upload_image_to_zernio(image_path, zernio_key)
        print(f"Uploaded: {url}")
        image_urls.append(url)

    result = post_to_instagram(image_urls, caption, zernio_account_id, zernio_key)
    print(f"\nPosted carousel to Instagram: {result}")


if __name__ == "__main__":
    main()
