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

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_API_KEY

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# ============ DATABASE ============
conn = sqlite3.connect('virtual_predictor.db', check_same_thread=False)
cursor = conn.cursor()

# Updated tables with goals and BTTS tracking
cursor.execute('''
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT UNIQUE,
        timestamp INTEGER,
        home_team TEXT,
        away_team TEXT,
        actual_outcome TEXT,
        actual_score TEXT,
        actual_goals_over TEXT,
        actual_btts TEXT,
        predicted_outcome TEXT,
        predicted_goals TEXT,
        predicted_btts TEXT,
        correct_outcome BOOLEAN,
        correct_goals BOOLEAN,
        correct_btts BOOLEAN
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT,
        match_text TEXT,
        prediction_outcome TEXT,
        prediction_goals TEXT,
        prediction_btts TEXT,
        timestamp INTEGER
    )
''')

conn.commit()

# ============ TEAM STRENGTHS ============
TEAM_STRENGTHS = {
    'MCI': 1.45, 'LIV': 1.38, 'ARS': 1.32, 'CHE': 1.25,
    'MUN': 1.20, 'TOT': 1.15, 'NEW': 1.08, 'AST': 1.02,
    'BHA': 0.98, 'BRE': 0.92, 'WHU': 0.88, 'CRY': 0.85,
    'FUL': 0.82, 'EVE': 0.78, 'WOL': 0.75, 'BOU': 0.72,
    'NOT': 0.68, 'LEE': 0.65, 'BUR': 0.62, 'SUN': 0.58
}

# ============ PREDICTOR WITH FULL ACCURACY ============
class FullPredictor:
    def __init__(self):
        self.outcome_accuracy = []
        self.goals_accuracy = []
        self.btts_accuracy = []
        self.pattern_sequence = []
        self.pattern_position = 0
        self.pattern_length = 0
        
    def calculate_expected_goals(self, home_team, away_team):
        home_strength = TEAM_STRENGTHS.get(home_team, 1.0)
        away_strength = TEAM_STRENGTHS.get(away_team, 1.0)
        
        home_expected = (home_strength / away_strength) * 1.35
        away_expected = (away_strength / home_strength) * 1.05
        
        return home_expected, away_expected
    
    def predict_outcome(self, home_team, away_team):
        # Use pattern if available
        if self.pattern_sequence and self.pattern_length > 0:
            outcome = self.pattern_sequence[self.pattern_position % self.pattern_length]
            self.pattern_position += 1
            return outcome
        
        # Statistical prediction based on team strengths
        home_strength = TEAM_STRENGTHS.get(home_team, 1.0)
        away_strength = TEAM_STRENGTHS.get(away_team, 1.0)
        
        strength_ratio = home_strength / away_strength
        
        if strength_ratio > 1.25:
            return '1'
        elif strength_ratio < 0.8:
            return '2'
        else:
            return np.random.choice(['1', 'X', '2'], p=[0.45, 0.25, 0.30])
    
    def predict_goals(self, home_team, away_team):
        home_expected, away_expected = self.calculate_expected_goals(home_team, away_team)
        total_expected = home_expected + away_expected
        
        # Over/Under 2.5 prediction
        if total_expected > 2.5:
            over_under = "OVER 2.5"
            goals_confidence = "HIGH" if total_expected > 3.0 else "MEDIUM"
        else:
            over_under = "UNDER 2.5"
            goals_confidence = "HIGH" if total_expected < 2.0 else "MEDIUM"
        
        # BTTS prediction
        btts_chance = (home_expected > 0.85 and away_expected > 0.85)
        btts = "YES" if btts_chance else "NO"
        
        # Exact score
        np.random.seed(int(hashlib.md5(f"{home_team}{away_team}{int(time.time()/60)}".encode()).hexdigest()[:8], 16))
        home_score = min(np.random.poisson(home_expected), 5)
        away_score = min(np.random.poisson(away_expected), 5)
        np.random.seed()
        
        return {
            'over_under': over_under,
            'btts': btts,
            'exact_score': f"{home_score}-{away_score}",
            'confidence': goals_confidence
        }
    
    def record_result(self, game_id, actual_outcome, actual_score, predicted_outcome, predicted_goals, predicted_btts):
        # Parse actual score
        try:
            home_actual, away_actual = map(int, actual_score.split('-'))
            total_actual = home_actual + away_actual
            actual_over = "OVER 2.5" if total_actual > 2.5 else "UNDER 2.5"
            actual_btts = "YES" if home_actual > 0 and away_actual > 0 else "NO"
        except:
            actual_over = None
            actual_btts = None
        
        # Check correctness
        correct_outcome = (actual_outcome == predicted_outcome)
        correct_goals = (actual_over == predicted_goals) if actual_over else False
        correct_btts = (actual_btts == predicted_btts) if actual_btts else False
        
        # Update accuracy windows
        self.outcome_accuracy.append(correct_outcome)
        self.goals_accuracy.append(correct_goals)
        self.btts_accuracy.append(correct_btts)
        
        # Keep last 100
        for acc in [self.outcome_accuracy, self.goals_accuracy, self.btts_accuracy]:
            if len(acc) > 100:
                acc.pop(0)
        
        # Save to database
        cursor.execute('''
            INSERT OR REPLACE INTO games 
            (game_id, timestamp, home_team, away_team, actual_outcome, actual_score, 
             actual_goals_over, actual_btts, predicted_outcome, predicted_goals, predicted_btts,
             correct_outcome, correct_goals, correct_btts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id, int(time.time()), 
            game_id.split('_')[0], game_id.split('_')[1],
            actual_outcome, actual_score, actual_over, actual_btts,
            predicted_outcome, predicted_goals, predicted_btts,
            correct_outcome, correct_goals, correct_btts
        ))
        conn.commit()
        
        # Try to find pattern
        if len(self.outcome_accuracy) % 10 == 0:
            self.find_pattern()
        
        return {
            'correct_outcome': correct_outcome,
            'correct_goals': correct_goals,
            'correct_btts': correct_btts
        }
    
    def find_pattern(self):
        cursor.execute('SELECT actual_outcome FROM games WHERE actual_outcome IS NOT NULL ORDER BY id DESC LIMIT 100')
        rows = cursor.fetchall()
        
        if len(rows) < 20:
            return
        
        outcomes = [row[0] for row in rows]
        
        for length in range(3, min(15, len(outcomes)//2)):
            pattern = outcomes[:length]
            matches = 0
            for i in range(0, len(outcomes) - length, length):
                if outcomes[i:i+length] == pattern:
                    matches += 1
            if matches >= 2:
                self.pattern_sequence = pattern
                self.pattern_length = length
                self.pattern_position = len(outcomes) % length
                print(f"[PATTERN] Found! Length: {length}, Confidence: {matches/ (len(outcomes)/length):.1%}")
                return
    
    def get_accuracy(self):
        outcome_acc = sum(self.outcome_accuracy) / len(self.outcome_accuracy) if self.outcome_accuracy else 0.5
        goals_acc = sum(self.goals_accuracy) / len(self.goals_accuracy) if self.goals_accuracy else 0.5
        btts_acc = sum(self.btts_accuracy) / len(self.btts_accuracy) if self.btts_accuracy else 0.5
        
        return {
            'outcome': outcome_acc * 100,
            'goals': goals_acc * 100,
            'btts': btts_acc * 100,
            'overall': ((outcome_acc + goals_acc + btts_acc) / 3) * 100
        }
    
    def get_games_tracked(self):
        cursor.execute('SELECT COUNT(*) FROM games WHERE actual_outcome IS NOT NULL')
        return cursor.fetchone()[0]
    
    def save_prediction(self, game_id, match_text, outcome, goals, btts):
        cursor.execute('''
            INSERT INTO predictions (game_id, match_text, prediction_outcome, prediction_goals, prediction_btts, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (game_id, match_text, outcome, goals, btts, int(time.time())))
        conn.commit()

predictor = FullPredictor()

# ============ VISION EXTRACTION ============
EXTRACTION_PROMPT = """
Extract team names from BetPawa virtual screenshot. Return ONLY JSON:
{"matches": [{"position": 1, "home": "MCI", "away": "LIV"}]}
DO NOT predict scores. Only extract 3-letter team codes.
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
    
    try:
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
    help_text = """
🎮 *FULL VIRTUAL PREDICTOR BOT*

Send a screenshot of BetPawa virtual games.

*Predictions:*
• 🎯 1X2 Outcome (1=Home, X=Draw, 2=Away)
• 📊 Over/Under 2.5 Goals
• ⚽ BTTS (Both Teams To Score)
• 🎲 Exact Score

*Commands:*
/stats - Full accuracy breakdown
/reset - Reset pattern learning
/record game_id outcome score - Record result
/pending - Show pending predictions
/help - This menu

*Example:* `/record MCI_LIV_1734567890 1 2-1`
    """
    bot.reply_to(message, help_text, parse_mode="Markdown")

@bot.message_handler(commands=["stats"])
def handle_stats(message):
    acc = predictor.get_accuracy()
    games = predictor.get_games_tracked()
    pattern_status = "Active" if predictor.pattern_sequence else "Learning"
    
    stats_text = f"""
📊 *FULL ACCURACY REPORT*
━━━━━━━━━━━━━━━━━━━━━━━━
🎮 Games recorded: {games}

*Market Performance:*
🎯 1X2 Outcome: {acc['outcome']:.1f}%
📊 O/U 2.5 Goals: {acc['goals']:.1f}%
⚽ BTTS: {acc['btts']:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━
📈 Overall: {acc['overall']:.1f}%

🔄 Pattern Status: {pattern_status}
📐 Pattern Length: {predictor.pattern_length}

💡 Record results with: `/record game_id outcome score`
    """
    bot.reply_to(message, stats_text, parse_mode="Markdown")

@bot.message_handler(commands=["reset"])
def handle_reset(message):
    predictor.pattern_sequence = []
    predictor.pattern_length = 0
    predictor.pattern_position = 0
    predictor.outcome_accuracy = []
    predictor.goals_accuracy = []
    predictor.btts_accuracy = []
    bot.reply_to(message, "🔄 Full pattern reset. Bot will learn from scratch.")

@bot.message_handler(commands=["record"])
def handle_record(message):
    try:
        parts = message.text.split()
        if len(parts) < 3:
            bot.reply_to(message, "❌ Usage: `/record game_id outcome score`\n\nExample: `/record MCI_LIV_1234567890 1 2-1`\n\nOutcome: 1=Home win, X=Draw, 2=Away win", parse_mode="Markdown")
            return
        
        game_id = parts[1]
        outcome = parts[2].upper()
        score = parts[3] if len(parts) > 3 else None
        
        if outcome not in ['1', 'X', '2']:
            bot.reply_to(message, "❌ Invalid outcome. Use: 1, X, or 2")
            return
        
        if not score or '-' not in score:
            bot.reply_to(message, "❌ Invalid score. Use format: 2-1")
            return
        
        # Get prediction
        cursor.execute('SELECT prediction_outcome, prediction_goals, prediction_btts FROM predictions WHERE game_id = ?', (game_id,))
        pred = cursor.fetchone()
        
        if pred:
            result = predictor.record_result(game_id, outcome, score, pred[0], pred[1], pred[2])
            
            acc = predictor.get_accuracy()
            
            response = f"✅ *Recorded:* {game_id}\n"
            response += f"📊 Result: {outcome} ({score})\n\n"
            response += f"*Prediction Results:*\n"
            response += f"🎯 1X2: {'✓' if result['correct_outcome'] else '✗'}\n"
            response += f"📊 O/U 2.5: {'✓' if result['correct_goals'] else '✗'}\n"
            response += f"⚽ BTTS: {'✓' if result['correct_btts'] else '✗'}\n\n"
            response += f"📈 New accuracy: {acc['overall']:.1f}%"
        else:
            response = f"✅ Recorded: {game_id} → {outcome} ({score})\n"
            response += f"⚠️ No prediction found for this game ID"
        
        bot.reply_to(message, response, parse_mode="Markdown")
        
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)[:100]}")

@bot.message_handler(commands=["pending"])
def handle_pending(message):
    cursor.execute('SELECT game_id, match_text, prediction_outcome, prediction_goals, prediction_btts FROM predictions ORDER BY id DESC LIMIT 10')
    rows = cursor.fetchall()
    
    if not rows:
        bot.reply_to(message, "No pending predictions. Send a screenshot first!")
        return
    
    response = "📋 *Recent Predictions*\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    for row in rows:
        response += f"⚽ {row[1]}\n"
        response += f"🎯 {row[2]} | 📊 {row[3]} | ⚽ {row[4]}\n"
        response += f"🆔 `{row[0]}`\n"
        response += f"📝 Record: `/record {row[0]} outcome score`\n\n"
    
    bot.reply_to(message, response, parse_mode="Markdown")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    msg = bot.reply_to(message, "📷 Analyzing screenshot with full prediction model...")
    
    try:
        file_id = message.photo[-1].file_id
        image_bytes = download_image(file_id)
        matches = extract_teams(image_bytes)
        
        if not matches:
            bot.edit_message_text("❌ Could not read teams. Please send clearer screenshot.", msg.chat.id, msg.message_id)
            return
        
        response = "🎮 *FULL VIRTUAL PREDICTIONS*\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for match in matches:
            game_id = f"{match['home']}_{match['away']}_{int(time.time())}"
            
            outcome = predictor.predict_outcome(match['home'], match['away'])
            goals = predictor.predict_goals(match['home'], match['away'])
            
            predictor.save_prediction(game_id, f"{match['home']} vs {match['away']}", outcome, goals['over_under'], goals['btts'])
            
            outcome_emoji = "🔮" if outcome == '1' else "🤝" if outcome == 'X' else "⚡"
            
            response += f"{outcome_emoji} *{match['home']} vs {match['away']}*\n"
            response += f"┌─────────────────────────\n"
            response += f"│ 🎯 1X2: *{outcome}*\n"
            response += f"│ 📊 O/U 2.5: *{goals['over_under']}*\n"
            response += f"│ ⚽ BTTS: *{goals['btts']}*\n"
            response += f"│ 🎲 Exact: *{goals['exact_score']}*\n"
            response += f"│ 📈 Confidence: {goals['confidence']}\n"
            response += f"└─────────────────────────\n"
            response += f"🆔 `{game_id}`\n\n"
        
        acc = predictor.get_accuracy()
        games_tracked = predictor.get_games_tracked()
        
        response += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += f"📊 *Accuracy:* {acc['overall']:.1f}% ({games_tracked} games)\n"
        response += f"   🎯 {acc['outcome']:.0f}% | 📊 {acc['goals']:.0f}% | ⚽ {acc['btts']:.0f}%\n"
        response += f"🔄 Pattern: {'Active' if predictor.pattern_sequence else 'Learning'}\n\n"
        response += f"📝 *After matches end:* `/record {match['home']}_{match['away']}_... 1 2-1`\n\n"
        response += f"⚠️ Virtual games follow algorithms - recording results improves accuracy"
        
        bot.edit_message_text(response, msg.chat.id, msg.message_id, parse_mode="Markdown")
        
    except Exception as e:
        bot.edit_message_text(f"Error: {str(e)[:100]}", msg.chat.id, msg.message_id)

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(message, "📸 Send a screenshot of BetPawa virtual games.\n\nCommands: /start, /stats, /reset, /record, /pending")

# ============ WEBHOOK ============
@app.route("/" + TELEGRAM_TOKEN, methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode("UTF-8"))
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Full Virtual Predictor Bot - Active ✅", 200

def setup_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"✅ Webhook set: {RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"✅ Full predictor bot active - 1X2, O/U 2.5, BTTS, Exact Score")

if __name__ == "__main__":
    setup_webhook()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
else:
    setup_webhook()
