import os
import base64
import re
import requests
from io import BytesIO
from PIL import Image
import telebot
from openai import OpenAI
from flask import Flask, request

# =========================
# CONFIG
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL = os.getenv("RENDER_URL")  # e.g. https://your-app.onrender.com

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
🔒 SYSTEM MODE: V8.0 — BETPAWA MATCHDAY SCORE PREDICTOR
📊 INPUT TYPE: SCREENSHOT (BETPAWA FIXTURES / ODDS / PAST RESULTS)
🎯 OUTPUT MODE: FULL MATCHDAY SCORE PREDICTIONS

🧠 CORE OBJECTIVE:
Analyze the BetPawa screenshot and generate SCORE PREDICTIONS for ALL visible matches on the current matchday.

Use a combination of:
- 📉 BetPawa displayed odds (1 / X / 2) → convert to implied probabilities
- 📊 Past result patterns visible in the screenshot (if any)
- ⚽ Known team strength, form, and head-to-head tendencies
- 🔢 Statistical scoring averages per league

⚙️ EXECUTION ENGINE (per match):
1️⃣ Read Home Team, Away Team, and displayed odds (Home / Draw / Away)
2️⃣ Convert odds → implied win probabilities
3️⃣ Determine most likely scoreline based on:
   - Favorite strength (odds gap)
   - Average goals for this fixture type
   - Home advantage factor
4️⃣ Assign confidence level based on odds clarity
5️⃣ Add 1–2 short betting tips per match

🚫 STRICT RULES:
- Analyze EVERY match visible in the screenshot
- NO skipping matches
- NO vague predictions like "could go either way"
- Use REAL scorelines only (e.g. 1-0, 2-1, 0-0, 3-1)
- NO half-time only predictions unless screenshot shows 1H market
- Base EVERY prediction on the odds shown + analysis

📢 OUTPUT FORMAT (repeat for each match):
──────────────────────────
⚽ [HOME TEAM] vs [AWAY TEAM]
🏆 Odds → Home: [X.XX] | Draw: [X.XX] | Away: [X.XX]
📈 Win Probability → Home [XX%] | Draw [XX%] | Away [XX%]
🎯 Predicted Score: [HOME SCORE] – [AWAY SCORE]
📊 Confidence: [HIGH / MEDIUM / LOW]
💡 Tips: [Tip 1] • [Tip 2]
──────────────────────────

After all matches, add:
━━━━━━━━━━━━━━━━━━━━━━━━
🔥 BEST BET OF THE DAY: [Match] → [Predicted Score]
⚡ Reason: [1 sentence why]
━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =========================
# HELPER FUNCTIONS
# =========================
def download_and_compress(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
    img_bytes = requests.get(file_url).content
    img = Image.open(BytesIO(img_bytes))
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()

def validate_output(text):
    """Check that output contains at least one valid prediction block."""
    return "Predicted Score:" in text and "Confidence:" in text

def fallback_output():
    return (
        "❌ Could not extract match data from the screenshot.\n\n"
        "📸 Please make sure your screenshot clearly shows:\n"
        "• Team names\n"
        "• BetPawa odds (1 / X / 2)\n"
        "• Matchday fixtures\n\n"
        "Then send the screenshot again."
    )

def analyze_image(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Model fallback chain: cheapest first → best as backup
    models = [
        "openai/gpt-4o-mini",           # Primary: cheap + fast + no refusals
        "anthropic/claude-3.5-sonnet",   # Backup 1: excellent vision, no refusals
        "openai/gpt-4o",                 # Backup 2: most powerful, higher cost
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this BetPawa screenshot and predict scores for all matches shown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        },
    ]

    for model in models:
        try:
            print(f"[INFO] Trying model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
            )
            result = response.choices[0].message.content.strip()

            if validate_output(result):
                print(f"[SUCCESS] Got valid prediction from {model}")
                return result
            else:
                print(f"[WARN] {model} returned invalid format, trying next...")
                continue

        except Exception as e:
            print(f"[ERROR] {model} failed: {str(e)}")
            continue

    return fallback_output()

# =========================
# BOT HANDLERS
# =========================
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "👋 Welcome to *BetPawa Score Predictor Bot!*\n\n"
        "📸 Send me a *screenshot* of BetPawa fixtures (with odds visible) and I'll predict scores for every match on the matchday.\n\n"
        "🤖 I combine:\n"
        "• BetPawa odds analysis\n"
        "• AI team strength evaluation\n"
        "• Statistical score prediction\n\n"
        "Just drop your screenshot below! ⬇️",
        parse_mode="Markdown"
    )

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    msg = bot.reply_to(message, "⏳ Analyzing BetPawa fixtures... Please wait.")
    try:
        file_id = message.photo[-1].file_id
        image_bytes = download_and_compress(file_id)
        result = analyze_image(image_bytes)
        bot.edit_message_text(result, chat_id=msg.chat.id, message_id=msg.message_id)
    except Exception as e:
        bot.edit_message_text(f"❌ Error: {str(e)}", chat_id=msg.chat.id, message_id=msg.message_id)

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(
        message,
        "📸 Please send a *BetPawa screenshot* with match fixtures and odds to get score predictions.",
        parse_mode="Markdown"
    )

# =========================
# WEBHOOK ROUTE
# =========================
@app.route("/" + TELEGRAM_TOKEN, methods=["POST"])
def get_message():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "!", 200

@app.route("/", methods=["GET"])
def index():
    return "BetPawa Predictor Bot is running! ✅", 200

# =========================
# STARTUP
# =========================
bot.remove_webhook()
bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
