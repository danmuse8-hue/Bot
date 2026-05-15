import os
import json
import base64
import time
import sqlite3
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import telebot
from flask import Flask, request

# ============ OLD OPENAI VERSION (WORKING) ============
import openai

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL = os.getenv("RENDER_URL")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY")
if not RENDER_URL:
    raise RuntimeError("Missing RENDER_URL")

# Configure OpenAI for OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_API_KEY

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# ============ DATABASE ============
conn = sqlite3.connect('virtual_predictor.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT UNIQUE,
        timestamp INTEGER,
        home_team TEXT,
        away_team TEXT,
        outcome TEXT,
        predicted_outcome TEXT,
        correct BOOLEAN
    )
''')
conn.commit()

# ============ PREDICTOR ============
class VirtualPredictor:
    def __init__(self):
        self.accuracy_window = []
        self.pattern_sequence = []
        self.pattern_position = 0
        self.pattern_length = 0
        
    def record_result(self, game_id, actual_outcome, predicted_outcome):
        correct = (actual_outcome == predicted_outcome)
        self.accuracy_window.append(correct)
        if len(self.accuracy_window) > 100:
            self.accuracy_window.pop(0)
        
        cursor.execute('''
            INSERT OR REPLACE INTO games (game_id, timestamp, outcome, predicted_outcome, correct)
            VALUES (?, ?, ?, ?, ?)
        ''', (game_id, int(time.time()), actual_outcome, predicted_outcome, correct))
        conn.commit()
        
        if len(self.accuracy_window) % 20 == 0:
            self.find_pattern()
    
    def find_pattern(self):
        cursor.execute('SELECT outcome FROM games ORDER BY id DESC LIMIT 100')
        outcomes = [row[0] for row in cursor.fetchall()]
        if len(outcomes) < 20:
            return
        
        for length in range(3, 15):
            pattern = outcomes[:length]
            matches = 0
            for i in range(0, len(outcomes) - length, length):
                if outcomes[i:i+length] == pattern:
                    matches += 1
            if matches >= 2:
                self.pattern_sequence = pattern
                self.pattern_length = length
                self.pattern_position = len(outcomes) % length
                print(f"Pattern found! Length: {length}")
                return
    
    def predict(self, game_id):
        if self.pattern_sequence and self.pattern_length > 0:
            outcome = self.pattern_sequence[self.pattern_position % self.pattern_length]
            self.pattern_position += 1
            return outcome
        return np.random.choice(['1', 'X', '2'], p=[0.48, 0.24, 0.28])
    
    def get_accuracy(self):
        if not self.accuracy_window:
            return 0.5
        return sum(self.accuracy_window) / len(self.accuracy_window)

predictor = VirtualPredictor()

# ============ VISION EXTRACTION ============
EXTRACTION_PROMPT = """
Extract team names from BetPawa virtual screenshot. Return JSON:
{"matches": [{"position": 1, "home": "MCI", "away": "LIV"}]}
DO NOT predict scores. Only extract team codes.
"""

def download_image(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
    img_bytes = requests.get(file_url).content
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()

def extract_teams(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}"}
    ]
    
    try:
        # OLD OPENAI SYNTAX (WORKING)
        response = openai.ChatCompletion.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract team names from this BetPawa virtual screenshot. Return JSON only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1 and end != 0:
            data = json.loads(result[start:end])
            return data.get('matches', [])
    except Exception as e:
        print(f"Extraction error: {e}")
    return None

# ============ BOT COMMANDS ============
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(message, "🎮 Virtual Football Predictor Bot\n\nSend a screenshot of BetPawa virtual games to get predictions.\n\nCommands:\n/stats - View accuracy\n/reset - Reset pattern learning")

@bot.message_handler(commands=["stats"])
def handle_stats(message):
    acc = predictor.get_accuracy() * 100
    bot.reply_to(message, f"📊 Accuracy: {acc:.1f}%\nGames tracked: {len(predictor.accuracy_window)}\nPattern active: {'Yes' if predictor.pattern_length > 0 else 'No'}")

@bot.message_handler(commands=["reset"])
def handle_reset(message):
    predictor.pattern_sequence = []
    predictor.pattern_length = 0
    predictor.pattern_position = 0
    bot.reply_to(message, "🔄 Pattern reset. Learning from scratch.")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    msg = bot.reply_to(message, "📷 Analyzing screenshot...")
    
    try:
        file_id = message.photo[-1].file_id
        image_bytes = download_image(file_id)
        matches = extract_teams(image_bytes)
        
        if not matches:
            bot.edit_message_text("❌ Could not read teams. Please send clearer screenshot.", msg.chat.id, msg.message_id)
            return
        
        response = "🎮 *VIRTUAL PREDICTIONS*\n\n"
        for match in matches:
            game_id = f"VIRT_{match['position']}_{int(time.time())}"
            pred = predictor.predict(game_id)
            
            emoji = "🔮" if pred == '1' else "🤝" if pred == 'X' else "⚡"
            response += f"{emoji} *{match['home']} vs {match['away']}*\n"
            response += f"Prediction: {pred}\n\n"
        
        acc = predictor.get_accuracy() * 100
        response += f"📊 Model accuracy: {acc:.1f}%\n"
        response += "⚠️ Virtual games follow patterns - results not guaranteed"
        
        bot.edit_message_text(response, msg.chat.id, msg.message_id, parse_mode="Markdown")
        
    except Exception as e:
        bot.edit_message_text(f"Error: {str(e)[:100]}", msg.chat.id, msg.message_id)

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(message, "📸 Please send a screenshot of BetPawa virtual games.")

# ============ WEBHOOK ============
@app.route("/" + TELEGRAM_TOKEN, methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode("UTF-8"))
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Bot Active", 200

def setup_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    setup_webhook()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
else:
    setup_webhook()
