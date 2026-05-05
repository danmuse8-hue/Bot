import os
import base64
import re
import requests
from io import BytesIO
from PIL import Image
import telebot
from openai import OpenAI
from flask import Flask, request

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL = os.getenv("RENDER_URL")

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """
🔒 SYSTEM MODE: V7.1 — BETPAWA VIRTUAL 1H TOP SCORER DETECTOR
📊 INPUT TYPE: SCREENSHOT (PAST RESULTS + CURRENT FIXTURES)
🎯 OUTPUT MODE: SINGLE MATCH ONLY

🧠 CORE OBJECTIVE:
Identify the ONE team with the HIGHEST TOTAL FIRST HALF (1H) GOALS SCORED across the LAST 3 MATCHDAYS, then locate its current fixture.

⚙️ EXECUTION ENGINE:
1️⃣ Extract ONLY first-half scores from LAST 3 matches
2️⃣ Sum ONLY goals scored (not conceded)
3️⃣ Pick team with highest total (consistency priority)
4️⃣ Match with current fixtures
5️⃣ Resolve ties via consistency → fixture order

🚫 RULES:
- NO explanations
- NO multiple picks
- NO full-time data

📢 OUTPUT:
🔥 [HOME TEAM] vs [AWAY TEAM]
⚡ TEAM: [TEAM NAME]
📊 CONFIDENCE: [HIGH/MEDIUM/LOW]
🎯 ACCURACY: [X/10]
"""

def download_and_compress(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
    img_bytes = requests.get(file_url).content
    img = Image.open(BytesIO(img_bytes))
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()

def validate_output(text):
    pattern = r"🔥 .+ vs .+\n⚡ TEAM: .+\n📊 CONFIDENCE: (HIGH|MEDIUM|LOW)\n🎯 ACCURACY: \d+/10"
    return re.search(pattern, text.strip())

def fix_output(text):
    lines = text.strip().split("\n")
    match_line = next((l for l in lines if "vs" in l), "🔥 UNKNOWN vs UNKNOWN")
    team_line = next((l for l in lines if "TEAM" in l), "⚡ TEAM: UNKNOWN")
    conf_line = next((l for l in lines if "CONFIDENCE" in l), "📊 CONFIDENCE: MEDIUM")
    acc_line = next((l for l in lines if "ACCURACY" in l), "🎯 ACCURACY: 5/10")
    return f"{match_line}\n{team_line}\n{conf_line}\n{acc_line}"

def analyze_image(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this screenshot."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]},
            ],
        )
        result = response.choices[0].message.content.strip()
        return result if validate_output(result) else fix_output(result)
    except Exception:
        return "❌ Failed to analyze image. Try again."

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "⏳ Processing...")
    file_id = message.photo[-1].file_id
    image_bytes = download_and_compress(file_id)
    bot.reply_to(message, analyze_image(image_bytes))

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(message, "📸 Send a screenshot.")

@app.route('/' + TELEGRAM_TOKEN, methods=['POST'])
def get_message():
    json_str = request.get_data().decode('UTF-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "!", 200

bot.remove_webhook()
bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
