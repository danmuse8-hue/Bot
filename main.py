import os
import base64
import time
import requests
from io import BytesIO
from PIL import Image
import telebot
from openai import OpenAI
from flask import Flask, request

# ─── ENV ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL         = os.getenv("RENDER_URL")

for _n, _v in [("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
               ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
               ("RENDER_URL", RENDER_URL)]:
    if not _v:
        raise RuntimeError(f"Missing env var: {_n}")

ai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
bot       = telebot.TeleBot(TELEGRAM_TOKEN)
app       = Flask(__name__)

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
🔒 SYSTEM MODE: V7.2 — BETPAWA VIRTUAL 1H TOP SCORER DETECTOR
📊 INPUT TYPE: SCREENSHOT (PAST RESULTS + CURRENT FIXTURES)
🎯 OUTPUT MODE: SINGLE MATCH ONLY

🧠 CORE OBJECTIVE:
Identify the ONE team with the HIGHEST TOTAL FIRST HALF (1H) GOALS SCORED across the available past matchdays, then locate its current fixture.

⚙️ EXECUTION ENGINE:
1️⃣ Scan ALL visible scores in the screenshot — results, matchday history, or any score data shown
2️⃣ If full-time scores are shown but not split by half, treat the full-time score as the basis and estimate 1H contribution
3️⃣ Sum goals scored (not conceded) per team across ALL visible past matches
4️⃣ Pick the team with the highest total (consistency is a tiebreaker)
5️⃣ Find that team's current/upcoming fixture in the screenshot
6️⃣ If no upcoming fixture is visible for that team, pick the next best team that HAS a fixture

⚠️ IMPORTANT RULES:
- NEVER refuse or say you cannot analyze — always produce a result using whatever data is visible
- Work with partial data if that's all that's available (even 1 matchday is enough)
- If scores are shown as (X-X) format, use those as 1H scores
- If only full-time scores are visible, still use them to rank teams by goals scored
- ALWAYS output in the exact format below — no extra text, no explanations

📢 OUTPUT FORMAT (strictly follow this, nothing else):
🔥 [HOME TEAM] vs [AWAY TEAM]
⚡ TEAM: [TEAM NAME]
📊 CONFIDENCE: [HIGH/MEDIUM/LOW]
🎯 ACCURACY: [X/10]
"""

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _resize(img_bytes, max_dim=1280):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()

def _dl_image(file_id):
    info = bot.get_file(file_id)
    resp = requests.get(
        f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{info.file_path}",
        timeout=15)
    resp.raise_for_status()
    return resp.content

def analyze_image(img_bytes):
    b64 = base64.b64encode(_resize(img_bytes)).decode()
    try:
        resp = ai_client.chat.completions.create(
            model="openai/gpt-4o",
            max_tokens=300, temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this screenshot."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as ex:
        print(f"[VISION] {ex}")
        return None

# ─── HANDLERS ────────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start", "help"])
def h_start(msg):
    bot.reply_to(msg,
        "┌─────────────────────────────┐\n"
        "│  🔥 *1H TOP SCORER BOT V7*  │\n"
        "└─────────────────────────────┘\n\n"
        "*📸 How to use:*\n"
        "Send a screenshot containing:\n"
        "• Past match results (with 1H scores)\n"
        "• Current upcoming fixtures\n\n"
        "The bot will identify the team with the *highest 1st half goals* "
        "across the last 3 matchdays and return their current fixture.\n\n"
        "⚠️ _Virtual football only. Not financial advice._",
        parse_mode="Markdown")

@bot.message_handler(content_types=["photo"])
def h_photo(msg):
    status = bot.reply_to(msg, "⏳ Processing screenshot…")
    try:
        img    = _dl_image(msg.photo[-1].file_id)
        result = analyze_image(img)

        if not result:
            bot.edit_message_text(
                "❌ Failed to analyze image. Please try again.",
                status.chat.id, status.message_id)
            return

        output = (
            "┌─────────────────────────────┐\n"
            "│  🔥 *1H TOP SCORER RESULT*  │\n"
            "└─────────────────────────────┘\n\n"
            f"{result}\n\n"
            "⚠️ _Not financial advice._"
        )
        bot.edit_message_text(output, status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as ex:
        print(f"[PHOTO ERROR] {ex}")
        bot.edit_message_text(f"⚠️ Error: {str(ex)[:120]}",
                              status.chat.id, status.message_id)

@bot.message_handler(func=lambda m: True)
def h_text(msg):
    bot.reply_to(msg,
        "📸 *Send a screenshot* to get started.\n"
        "Need help? → /help",
        parse_mode="Markdown")

# ─── WEBHOOK ─────────────────────────────────────────────────────────────────
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    bot.process_new_updates([
        telebot.types.Update.de_json(request.get_data(as_text=True))
    ])
    return "OK", 200

@app.route("/", methods=["GET"])
def health():
    return "1H Top Scorer Bot V7 ✅", 200

def setup_webhook():
    bot.remove_webhook()
    time.sleep(0.5)
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[WEBHOOK] {RENDER_URL}/{TELEGRAM_TOKEN}")

# ─── STARTUP ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
