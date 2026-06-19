# Daily Instagram Auto-Poster

Generates a themed image + caption every day (entirely with Google's Gemini/Imagen API) and
publishes it to your Instagram account automatically, on a schedule, using GitHub Actions and Zernio.

## How it works
Every day at a set time, GitHub Actions spins up, runs `scripts/generate_and_post.py`, which:
1. Reads your theme from `prompt.txt`
2. Asks Gemini to turn that theme into a specific image prompt + caption for today
3. Generates the image with Google Imagen
4. Uploads the image to Zernio's media storage
5. Publishes it to your Instagram account via Zernio

No Meta developer app, no Graph API setup, no Anthropic key required.

## One-time setup

### 1. Confirm your Instagram account is ready
In Meta Business Suite, make sure your Instagram account is set to **Professional** (Creator).
Zernio still requires a Business or Creator account (personal accounts can't be posted to via
any API) but it handles the Meta OAuth/app side itself.

### 2. Get a Gemini API key
Go to https://aistudio.google.com/apikey, sign in, and create an API key. The free tier covers
one image + one caption a day comfortably.

### 3. Set up Zernio
1. Sign up at https://zernio.com (free tier covers up to 2 connected accounts).
2. Get your API key from the Zernio dashboard.
3. In the dashboard, connect your Instagram account (Connect Account → Instagram → follow the
   OAuth prompts). This is the one place Meta's login screen still shows up, but you're doing it
   through Zernio's flow, not creating your own Meta developer app.
4. Once connected, find your account's ID — either from the dashboard's account list, or by
   calling:
   ```
   GET https://zernio.com/api/v1/accounts
   Authorization: Bearer YOUR_ZERNIO_API_KEY
   ```
   Look for the entry with `"platform": "instagram"` and copy its `accountId`.

### 4. Create a GitHub repo
Create a repo (public or private — Zernio hosts the generated images now, so there's no need for
a public repo like the earlier Graph API version required) and push everything in this folder,
keeping the folder structure: `.github/workflows/daily-post.yml`, `scripts/generate_and_post.py`,
`requirements.txt`, `prompt.txt`.

### 5. Add your secrets
In the repo: Settings → Secrets and variables → Actions → New repository secret. Add three:
- `GEMINI_API_KEY`
- `ZERNIO_API_KEY`
- `ZERNIO_ACCOUNT_ID`

### 6. Set your content theme
Edit `prompt.txt` with whatever you want posted daily, e.g.:
```
Motivational quotes paired with serene nature landscapes (mountains, oceans, forests, sunrises)
```
Gemini uses this as the theme and invents a fresh specific angle each day so posts don't repeat.

### 7. Test it
Go to the repo's **Actions** tab → "Daily Instagram Post" → "Run workflow" to trigger it manually,
then check your Instagram account and the Action's logs.

### 8. Let it run
Once it works, it fires automatically every day at the time set in
`.github/workflows/daily-post.yml` (`cron: '0 13 * * *'` = 13:00 UTC — edit to your preferred time;
GitHub Actions cron always runs in UTC).

## Maintenance
- Zernio manages and refreshes the underlying Instagram tokens itself, so there's no 60-day manual
  token refresh like the direct Graph API approach needed.
- Instagram limits accounts to 100 API-published posts per rolling 24 hours — a non-issue at one
  post/day.
- If a run fails, check the Actions log first — Zernio's error messages (e.g. "media fetch failed",
  "duplicate content detected") are descriptive and usually self-explanatory.
