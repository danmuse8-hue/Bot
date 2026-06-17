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
You are an expert BetPawa virtual football analyst and predictor.

You will receive a screenshot showing UPCOMING fixtures (no scores yet).

Your job is to analyze each visible match and predict the following markets:
- 1X2 outcome (1 = Home Win, X = Draw, 2 = Away Win)
- Over/Under 2.5 goals
- BTTS (Both Teams To Score: YES or NO)
- 1st Half Over 0.5 goals (YES or NO)
- Direct Win (which team wins by exactly 1 goal, or NONE)
- Confidence level per match (HIGH / MEDIUM / LOW)

Use your knowledge of these virtual English Premier League teams and their typical attacking/defensive strengths to make predictions. Virtual football follows patterns — stronger teams score more, weaker teams concede more.

RULES:
- NEVER refuse. Always predict every visible match.
- Base predictions purely on team strength knowledge — no past results needed.
- Output ONLY the formatted prediction cards below. No extra text before or after.
- One card per match.

OUTPUT FORMAT (repeat this block for every match):
╔══ MATCH [N] ══════════════════╗
║ [HOME TEAM] vs [AWAY TEAM]
╠═══════════════════════════════╣
║ 🏠 1X2:       [1 / X / 2]
║ ⚡ O/U 2.5:   [OVER / UNDER]
║ 🤜 BTTS:      [YES / NO]
║ ⏱ 1H O/U:    [OVER 0.5 / UNDER 0.5]
║ 🎯 Direct W:  [HOME / AWAY / NONE]
║ 📊 Confidence:[HIGH / MEDIUM / LOW]
╚═══════════════════════════════╝
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
            max_tokens=2000, temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the upcoming fixtures screenshot. Predict all matches."},
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
        "│  ⚡ *VIRTUAL PREDICTOR V8*  │\n"
        "└─────────────────────────────┘\n\n"
        "*📸 How to use:*\n"
        "Just send a screenshot of the *upcoming fixtures* and the bot will predict every match.\n\n"
        "*📊 Markets predicted per match:*\n"
        "• 🏠 1X2 Outcome\n"
        "• ⚡ Over / Under 2.5\n"
        "• 🤜 BTTS (Both Teams To Score)\n"
        "• ⏱ 1st Half Over 0.5 goals\n"
        "• 🎯 Direct Win (win by exactly 1)\n"
        "• 📊 Confidence Level\n\n"
        "⚠️ _Virtual football only. Not financial advice._",
        parse_mode="Markdown")

@bot.message_handler(content_types=["photo"])
def h_photo(msg):
    status = bot.reply_to(msg, "⚡ Analysing fixtures…")
    try:
        img    = _dl_image(msg.photo[-1].file_id)
        result = analyze_image(img)

        if not result:
            bot.edit_message_text(
                "❌ Failed to analyze image. Please try again.",
                status.chat.id, status.message_id)
            return

        header = (
            "┌─────────────────────────────┐\n"
            "│  ⚡ *PREDICTIONS*            │\n"
            "└─────────────────────────────┘\n\n"
        )
        footer = (
            "\n─────────────────────────────\n"
            "⚠️ _Not financial advice._"
        )
        bot.edit_message_text(
            header + result + footer,
            status.chat.id, status.message_id,
            parse_mode="Markdown")

    except Exception as ex:
        print(f"[PHOTO ERROR] {ex}")
        bot.edit_message_text(f"⚠️ Error: {str(ex)[:120]}",
                              status.chat.id, status.message_id)

@bot.message_handler(func=lambda m: True)
def h_text(msg):
    bot.reply_to(msg,
        "📸 *Send a screenshot* of upcoming fixtures to get predictions.\n"
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
    return "Virtual Predictor V8 ✅", 200

def setup_webhook():
    bot.remove_webhook()
    time.sleep(0.5)
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[WEBHOOK] {RENDER_URL}/{TELEGRAM_TOKEN}")

# ─── STARTUP ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
