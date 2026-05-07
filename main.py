import os
import base64
import requests
from io import BytesIO
from PIL import Image
import telebot
from openai import OpenAI
from flask import Flask, request

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL = os.getenv("RENDER_URL")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY")
if not RENDER_URL:
    raise RuntimeError("Missing RENDER_URL")

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """
You are a BetPawa match score predictor bot.
Analyze the BetPawa screenshot and generate SCORE PREDICTIONS for ALL visible matches.
Use BetPawa displayed odds, team strength, form, and statistical scoring averages.

For each match output exactly:
──────────────────────────
⚽ [HOME TEAM] vs [AWAY TEAM]
🏆 Odds → Home: [X.XX] | Draw: [X.XX] | Away: [X.XX]
📈 Win Probability → Home [XX%] | Draw [XX%] | Away [XX%]
🎯 Predicted Score: [HOME] – [AWAY]
📊 Confidence: [HIGH / MEDIUM / LOW]
💡 Tips: [Tip 1] • [Tip 2]
──────────────────────────

After all matches add:
━━━━━━━━━━━━━━━━━━━━━━━━
🔥 BEST BET OF THE DAY: [Match] → [Predicted Score]
⚡ Reason: [1 sentence why]
━━━━━━━━━━━━━━━━━━━━━━━━

Rules:
- Analyze EVERY match visible
- NO skipping matches
- Use REAL scorelines only (e.g. 1-0, 2-1, 0-0)
"""

def download_and_compress(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
    img_bytes = requests.get(file_url).content
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()

def analyze_image(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
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
            if "Predicted Score" in result:
                print(f"[SUCCESS] {model}")
                return result
        except Exception as e:
            print(f"[ERROR] {model}: {str(e)}")
            continue
    return (
        "❌ Could not extract match data from the screenshot.\n\n"
        "📸 Please make sure your screenshot clearly shows:\n"
        "• Team names\n"
        "• BetPawa odds (1 / X / 2)\n"
        "• Matchday fixtures\n\n"
        "Then send the screenshot again."
    )

@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "👋 Welcome to *BetPawa Score Predictor Bot!*\n\n"
        "📸 Send me a *screenshot* of BetPawa fixtures with odds and I'll predict scores for every match.\n\n"
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
        if len(result) > 4096:
            result = result[:4090] + "\n..."
        bot.edit_message_text(result, chat_id=msg.chat.id, message_id=msg.message_id)
    except Exception as e:
        print(f"[ERROR] handle_photo: {str(e)}")
        bot.edit_message_text(f"❌ Error: {str(e)}", chat_id=msg.chat.id, message_id=msg.message_id)

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(
        message,
        "📸 Please send a *BetPawa screenshot* with match fixtures and odds.",
        parse_mode="Markdown"
    )

@app.route("/" + TELEGRAM_TOKEN, methods=["POST"])
def get_message():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "!", 200

@app.route("/", methods=["GET"])
def index():
    return "BetPawa Predictor Bot is running! ✅", 200

def setup_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[INFO] Webhook set to {RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    setup_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
else:
    setup_webhook()
