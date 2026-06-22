"""
Daily Instagram poster - OpenAI (caption + image) + Zernio (publishing).

Pipeline:
  1. Read the theme prompt from prompt.txt
  2. Ask GPT-4o for today's specific image prompt + caption, based on that theme
  3. Generate the image with DALL-E 3
  4. Upload the image to Zernio's media storage (presigned URL flow)
  5. Publish the post to Instagram via Zernio's posts API

Required environment variables (set as GitHub repo secrets):
  OPENAI_API_KEY      - OpenAI API key
  ZERNIO_API_KEY      - Zernio API key
  ZERNIO_ACCOUNT_ID   - Zernio's account ID for your connected Instagram account
"""

import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ZERNIO_BASE_URL = "https://zernio.com/api/v1"
PROMPT_FILE = Path("prompt.txt")
POSTS_DIR = Path("posts")


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


def generate_image_prompt_and_caption(theme: str, client: OpenAI) -> tuple[str, str]:
    """Ask GPT-4o to turn the theme into today's specific image prompt + IG caption."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Theme for my Instagram account: {theme}\n"
                    f"Today's date: {today}\n\n"
                    "Create ONE Instagram post for today based on this theme. "
                    "Respond with ONLY raw JSON (no markdown fences, no commentary) in exactly this shape:\n"
                    '{"image_prompt": "...", "caption": "..."}\n\n'
                    "image_prompt: a vivid, specific, English-language text-to-image prompt (under 400 "
                    "characters) describing subject, setting, and visual style for DALL-E 3. "
                    "Give it a fresh, specific angle on the theme so it doesn't feel repetitive day to day.\n"
                    "caption: an engaging Instagram caption (2-4 sentences), followed by a line of 5-8 "
                    "relevant hashtags."
                ),
            }
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return data["image_prompt"], data["caption"]


def generate_image(image_prompt: str, out_path: Path, client: OpenAI) -> None:
    """Generate an image with DALL-E 3 and save it to disk."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="url",
    )

    image_url = response.data[0].url
    image_data = requests.get(image_url, timeout=60).content

    with open(out_path, "wb") as f:
        f.write(image_data)


def upload_image_to_zernio(image_path: Path, api_key: str) -> str:
    """Upload the image via Zernio's presigned upload flow and return its public URL."""
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


def post_to_instagram(image_url: str, caption: str, account_id: str, api_key: str) -> dict:
    response = requests.post(
        f"{ZERNIO_BASE_URL}/posts",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "content": caption,
            "mediaItems": [{"type": "image", "url": image_url}],
            "platforms": [{"platform": "instagram", "accountId": account_id}],
            "publishNow": True,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    openai_key = env("OPENAI_API_KEY")
    zernio_key = env("ZERNIO_API_KEY")
    zernio_account_id = env("ZERNIO_ACCOUNT_ID")

    POSTS_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=openai_key)

    theme = read_theme_prompt()
    print(f"Theme: {theme}")

    image_prompt, caption = generate_image_prompt_and_caption(theme, client)
    print(f"Image prompt: {image_prompt}")
    print(f"Caption: {caption}")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    image_path = POSTS_DIR / f"{today_str}.png"
    generate_image(image_prompt, image_path, client)
    print(f"Image saved to {image_path}")

    image_url = upload_image_to_zernio(image_path, zernio_key)
    print(f"Image uploaded to: {image_url}")

    result = post_to_instagram(image_url, caption, zernio_account_id, zernio_account_id)
    print(f"Posted to Instagram: {result}")


if __name__ == "__main__":
    main()
