import os
import json
import base64
import hashlib
import time
import sqlite3
import threading
import requests
from io import BytesIO
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats
from PIL import Image
import telebot
from openai import OpenAI
from flask import Flask, request

# ============ LOAD CONFIGURATION ============
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

# FIXED: OpenAI client with proper initialization
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============ DATABASE SETUP ============
class VirtualGameDB:
    def __init__(self, db_path="virtual_predictor.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                timestamp INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                outcome TEXT,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                predicted_outcome TEXT,
                predicted_score TEXT,
                correct BOOLEAN,
                confidence TEXT,
                method_used TEXT,
                session_id INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_eras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                era_name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                start_game_id INTEGER,
                end_game_id INTEGER,
                games_in_era INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS era_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                era_id INTEGER,
                pattern_hash TEXT,
                sequence_length INTEGER,
                pattern_data TEXT,
                confidence REAL,
                discovered_at INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_strengths (
                team_name TEXT PRIMARY KEY,
                attack_strength REAL,
                defense_strength REAL,
                last_updated TEXT
            )
        ''')
        
        self.conn.commit()
    
    def save_game(self, game_data):
        self.cursor.execute('''
            INSERT OR REPLACE INTO games 
            (game_id, timestamp, home_team, away_team, home_score, away_score, 
             outcome, odds_home, odds_draw, odds_away, predicted_outcome, 
             predicted_score, correct, confidence, method_used, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_data['game_id'], game_data['timestamp'], game_data['home_team'],
            game_data['away_team'], game_data['home_score'], game_data['away_score'],
            game_data['outcome'], game_data['odds_home'], game_data['odds_draw'],
            game_data['odds_away'], game_data.get('predicted_outcome'),
            game_data.get('predicted_score'), game_data.get('correct'),
            game_data.get('confidence'), game_data.get('method_used'),
            game_data.get('session_id', 1)
        ))
        self.conn.commit()
    
    def get_recent_games(self, limit=500):
        self.cursor.execute('SELECT * FROM games ORDER BY timestamp DESC LIMIT ?', (limit,))
        return self.cursor.fetchall()
    
    def get_game_count(self):
        self.cursor.execute('SELECT COUNT(*) FROM games')
        return self.cursor.fetchone()[0]
    
    def get_team_strength(self, team_name):
        self.cursor.execute('SELECT attack_strength, defense_strength FROM team_strengths WHERE team_name = ?', (team_name,))
        result = self.cursor.fetchone()
        if result:
            return result[0], result[1]
        return 1.0, 1.0
    
    def update_team_strength(self, team_name, attack, defense):
        self.cursor.execute('''
            INSERT OR REPLACE INTO team_strengths (team_name, attack_strength, defense_strength, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (team_name, attack, defense, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_active_pattern(self):
        self.cursor.execute('''
            SELECT pattern_data, sequence_length, confidence 
            FROM era_patterns 
            ORDER BY discovered_at DESC LIMIT 1
        ''')
        return self.cursor.fetchone()

# ============ SELF-HEALING PREDICTOR ============
class SelfHealingVirtualPredictor:
    def __init__(self):
        self.db = VirtualGameDB()
        self.current_session_id = 1
        self.active_pattern_sequence = []
        self.cycle_length = None
        self.cycle_position = 0
        self.accuracy_window = []
        self.current_method = 'baseline'
        self.method_accuracy = {'cycle': 0.5, 'hash': 0.5, 'ml': 0.5, 'baseline': 0.5}
        
        self.load_latest_pattern()
        self.init_era()
    
    def init_era(self):
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT id FROM pattern_eras WHERE end_time IS NULL ORDER BY id DESC LIMIT 1')
        era = cursor.fetchone()
        if not era:
            cursor.execute('''
                INSERT INTO pattern_eras (era_name, start_time, start_game_id, games_in_era)
                VALUES (?, ?, ?, ?)
            ''', (f"Era_{int(time.time())}", int(time.time()), self.db.get_game_count(), 0))
            self.db.conn.commit()
    
    def load_latest_pattern(self):
        pattern = self.db.get_active_pattern()
        if pattern:
            self.active_pattern_sequence = json.loads(pattern[0])
            self.cycle_length = pattern[1]
    
    def record_game_result(self, game_data):
        self.db.save_game(game_data)
        
        if 'predicted_outcome' in game_data and 'correct' in game_data:
            self.accuracy_window.append(game_data['correct'])
            if len(self.accuracy_window) > 200:
                self.accuracy_window.pop(0)
            
            if 'method_used' in game_data:
                old_acc = self.method_accuracy.get(game_data['method_used'], 0.5)
                self.method_accuracy[game_data['method_used']] = old_acc * 0.95 + (1 if game_data['correct'] else 0) * 0.05
            
            game_count = self.db.get_game_count()
            if game_count % 25 == 0 and game_count > 0:
                self.find_new_patterns()
            if game_count % 50 == 0 and game_count > 0:
                self.optimize_prediction_method()
            
            self.check_for_reset()
    
    def check_for_reset(self):
        if len(self.accuracy_window) < 30:
            return
        
        recent_acc = sum(self.accuracy_window[-30:]) / 30
        if recent_acc < 0.45 and len(self.accuracy_window) >= 50:
            older_acc = sum(self.accuracy_window[-50:-30]) / 20
            if older_acc - recent_acc > 0.15:
                print(f"⚠️ Reset detected! {older_acc:.1%} → {recent_acc:.1%}")
                self.trigger_reset()
    
    def trigger_reset(self):
        cursor = self.db.conn.cursor()
        cursor.execute('''
            UPDATE pattern_eras 
            SET end_time = ?, games_in_era = (SELECT COUNT(*) FROM games WHERE session_id = ?)
            WHERE end_time IS NULL
        ''', (int(time.time()), self.current_session_id))
        
        cursor.execute('''
            INSERT INTO pattern_eras (era_name, start_time, start_game_id, games_in_era)
            VALUES (?, ?, ?, ?)
        ''', (f"Era_{int(time.time())}", int(time.time()), self.db.get_game_count(), 0))
        self.db.conn.commit()
        
        self.active_pattern_sequence = []
        self.cycle_length = None
        self.cycle_position = 0
        self.current_method = 'baseline'
        self.accuracy_window = self.accuracy_window[-20:] if len(self.accuracy_window) > 20 else []
        for method in self.method_accuracy:
            self.method_accuracy[method] = max(0.4, self.method_accuracy[method] * 0.5)
    
    def find_new_patterns(self):
        recent_games = self.db.get_recent_games(200)
        if len(recent_games) < 50:
            return
        
        outcomes = [g[7] for g in recent_games]
        
        for pattern_len in range(3, min(20, len(outcomes) // 3)):
            for start in range(min(5, len(outcomes) - pattern_len)):
                pattern = tuple(outcomes[start:start + pattern_len])
                occurrences = []
                for i in range(0, len(outcomes) - pattern_len, 1):
                    if tuple(outcomes[i:i + pattern_len]) == pattern:
                        occurrences.append(i)
                
                if len(occurrences) >= 3:
                    confidence = len(occurrences) / (len(outcomes) / pattern_len)
                    if confidence > 0.6:
                        self.save_pattern(list(pattern), pattern_len, confidence)
                        return
    
    def save_pattern(self, pattern, length, confidence):
        pattern_hash = hashlib.md5(str(pattern).encode()).hexdigest()
        cursor = self.db.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO era_patterns 
            (era_id, pattern_hash, sequence_length, pattern_data, confidence, discovered_at)
            VALUES ((SELECT id FROM pattern_eras WHERE end_time IS NULL), ?, ?, ?, ?, ?)
        ''', (pattern_hash, length, json.dumps(pattern), confidence, int(time.time())))
        self.db.conn.commit()
        
        if not self.active_pattern_sequence or confidence > 0.7:
            self.active_pattern_sequence = pattern
            self.cycle_length = length
            self.current_method = 'cycle'
    
    def predict_match(self, game_id, home_team, away_team, timestamp=None):
        if timestamp is None:
            timestamp = int(time.time())
        
        best_method = max(self.method_accuracy, key=self.method_accuracy.get)
        if self.active_pattern_sequence and self.method_accuracy.get('cycle', 0) > 0.6:
            best_method = 'cycle'
        
        if best_method == 'cycle':
            predicted_outcome = self.predict_from_cycle()
        elif best_method == 'hash':
            predicted_outcome = self.predict_from_hash(game_id, timestamp)
        elif best_method == 'ml':
            predicted_outcome = self.predict_from_ml()
        else:
            predicted_outcome = self.predict_from_baseline()
        
        predicted_score = self.predict_score(home_team, away_team, game_id, timestamp)
        confidence = self.calculate_confidence()
        
        return {
            'outcome': predicted_outcome,
            'score': predicted_score,
            'confidence': confidence,
            'method_used': best_method,
            'method_accuracy': self.method_accuracy[best_method],
            'overall_accuracy': self.get_accuracy()
        }
    
    def predict_from_cycle(self):
        if not self.active_pattern_sequence:
            return None
        position = self.cycle_position % self.cycle_length
        self.cycle_position += 1
        return self.active_pattern_sequence[position] if position < len(self.active_pattern_sequence) else None
    
    def predict_from_hash(self, game_id, timestamp):
        time_window = timestamp // 60
        seed = f"{game_id}_{time_window}"
        seed_hash = hashlib.md5(seed.encode()).hexdigest()
        hash_int = int(seed_hash[:8], 16)
        
        outcomes = ['1', 'X', '2']
        dist = self.get_current_distribution()
        
        rand_val = hash_int % 100
        cumulative = 0
        for outcome, weight in zip(outcomes, dist):
            cumulative += weight * 100
            if rand_val < cumulative:
                return outcome
        return '1'
    
    def predict_from_ml(self):
        recent_games = self.db.get_recent_games(10)
        if len(recent_games) < 5:
            return self.predict_from_baseline()
        
        last_5 = tuple([g[7] for g in recent_games[:5]])
        outcomes = [g[7] for g in self.db.get_recent_games(200)]
        
        for i in range(len(outcomes) - 5):
            if tuple(outcomes[i:i+5]) == last_5:
                if i + 5 < len(outcomes):
                    return outcomes[i+5]
        return self.predict_from_baseline()
    
    def predict_from_baseline(self):
        dist = self.get_current_distribution()
        rand_val = np.random.random()
        cumulative = 0
        for outcome, prob in zip(['1', 'X', '2'], dist):
            cumulative += prob
            if rand_val < cumulative:
                return outcome
        return '1'
    
    def predict_score(self, home_team, away_team, game_id, timestamp):
        time_window = timestamp // 60
        seed = f"{game_id}_{time_window}_{home_team}_{away_team}"
        seed_hash = hashlib.md5(seed.encode()).hexdigest()
        
        home_attack, home_defense = self.db.get_team_strength(home_team)
        away_attack, away_defense = self.db.get_team_strength(away_team)
        
        base_goals = 1.4
        home_expected = base_goals * home_attack / max(away_defense, 0.5)
        away_expected = base_goals * away_attack / max(home_defense, 0.5)
        
        np.random.seed(int(seed_hash[:8], 16))
        home_goals = min(np.random.poisson(home_expected), 5)
        away_goals = min(np.random.poisson(away_expected), 5)
        np.random.seed()
        
        return f"{home_goals}-{away_goals}"
    
    def get_current_distribution(self):
        recent_games = self.db.get_recent_games(100)
        if not recent_games:
            return [0.48, 0.24, 0.28]
        
        outcomes = [g[7] for g in recent_games]
        total = len(outcomes)
        return [outcomes.count('1')/total, outcomes.count('X')/total, outcomes.count('2')/total]
    
    def get_accuracy(self):
        if not self.accuracy_window:
            return 0.5
        return sum(self.accuracy_window) / len(self.accuracy_window)
    
    def get_rolling_accuracy(self, games=50):
        if len(self.accuracy_window) < games:
            return self.get_accuracy()
        return sum(self.accuracy_window[-games:]) / games
    
    def calculate_confidence(self):
        acc = self.get_accuracy()
        if acc > 0.75: return "VERY HIGH"
        if acc > 0.65: return "HIGH"
        if acc > 0.55: return "MEDIUM"
        if acc > 0.45: return "LOW"
        return "LEARNING"
    
    def optimize_prediction_method(self):
        best_method = max(self.method_accuracy, key=self.method_accuracy.get)
        if best_method != self.current_method:
            self.current_method = best_method
    
    def get_statistics(self):
        game_count = self.db.get_game_count()
        accuracy = self.get_accuracy() * 100
        rolling = self.get_rolling_accuracy() * 100
        
        return f"""
📊 *VIRTUAL PREDICTOR STATISTICS*
━━━━━━━━━━━━━━━━━━━━━━━━
🎮 Games tracked: {game_count}
📈 Overall accuracy: {accuracy:.1f}%
📊 Rolling accuracy (50 games): {rolling:.1f}%
🔄 Current method: {self.current_method}
🎯 Active pattern: {'YES' if self.cycle_length else 'NO'}
📐 Pattern length: {self.cycle_length if self.cycle_length else 'N/A'}

*Method Performance:*
• Cycle: {self.method_accuracy['cycle']*100:.1f}%
• Hash: {self.method_accuracy['hash']*100:.1f}%
• ML: {self.method_accuracy['ml']*100:.1f}%
• Baseline: {self.method_accuracy['baseline']*100:.1f}%

⚠️ Status: {'HIGH CONFIDENCE' if rolling > 60 else 'LEARNING PATTERNS'}
"""

# ============ VISION EXTRACTION (NO PREDICTIONS) ============
EXTRACTION_PROMPT = """
You are an OCR system. Extract EXACTLY this data from the BetPawa virtual screenshot:

DO NOT predict scores.
DO NOT analyze odds.
DO NOT add any commentary.

Extract only:
1. Team names (3-letter codes like MCI, LIV, CHE, ARS, MUN, TOT, NEW, BHA)
2. Order of matches (1st, 2nd, 3rd, etc.)

Return ONLY valid JSON format:
{
    "matches": [
        {"position": 1, "home": "MCI", "away": "LIV"},
        {"position": 2, "home": "CHE", "away": "ARS"}
    ]
}

If you cannot read something clearly, put null.
"""

def download_and_compress(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
    img_bytes = requests.get(file_url).content
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    if img.width > 1024:
        ratio = 1024 / img.width
        new_size = (1024, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=75, optimize=True)
    return buffer.getvalue()

def extract_teams_from_screenshot(image_bytes):
    """Vision LLM - ONLY extracts team names, NO predictions"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract team names from this BetPawa virtual screenshot. Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        },
    ]
    
    models = ["openai/gpt-4o-mini", "google/gemini-1.5-flash", "anthropic/claude-3-haiku"]
    
    for model in models:
        try:
            print(f"[INFO] Extracting with: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
                if 'matches' in data:
                    print(f"[SUCCESS] Extracted {len(data['matches'])} matches")
                    return data['matches']
        except Exception as e:
            print(f"[ERROR] {model}: {str(e)}")
            continue
    
    return None

# ============ TELEGRAM BOT HANDLERS ============
predictor = SelfHealingVirtualPredictor()

@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "🎮 *Virtual Football Predictor Bot*\n\n"
        "I analyze BetPawa virtual games using pattern recognition.\n\n"
        "📸 Send me a screenshot of virtual fixtures!\n\n"
        "📊 *Commands:*\n"
        "/stats - View model performance\n"
        "/accuracy - Current prediction accuracy\n"
        "/reset - Force model reset\n"
        "/method - Show active prediction method",
        parse_mode="Markdown"
    )

@bot.message_handler(commands=["stats"])
def show_stats(message):
    stats = predictor.get_statistics()
    bot.reply_to(message, stats, parse_mode="Markdown")

@bot.message_handler(commands=["accuracy"])
def show_accuracy(message):
    acc = predictor.get_accuracy() * 100
    rolling = predictor.get_rolling_accuracy() * 100
    bot.reply_to(
        message,
        f"📊 *Accuracy Report*\n\n"
        f"Overall: {acc:.1f}%\n"
        f"Last 50 games: {rolling:.1f}%\n"
        f"Current method: {predictor.current_method}\n\n"
        f"💡 Model improves with more data.",
        parse_mode="Markdown"
    )

@bot.message_handler(commands=["reset"])
def force_reset(message):
    predictor.trigger_reset()
    bot.reply_to(message, "🔄 Model reset. Will relearn patterns from next 50 games.")

@bot.message_handler(commands=["method"])
def show_method(message):
    response = f"🔬 *Current Prediction Method:* {predictor.current_method}\n\n"
    response += "*Method Accuracies:*\n"
    for method, acc in predictor.method_accuracy.items():
        response += f"• {method}: {acc*100:.1f}%\n"
    bot.reply_to(message, response, parse_mode="Markdown")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    processing_msg = bot.reply_to(message, "📷 Extracting teams from screenshot...")
    
    try:
        file_id = message.photo[-1].file_id
        image_bytes = download_and_compress(file_id)
        
        matches = extract_teams_from_screenshot(image_bytes)
        
        if not matches or len(matches) == 0:
            bot.edit_message_text(
                "❌ Could not read team names from screenshot.\n\n"
                "Please ensure:\n"
                "• Screenshot is clear\n"
                "• Team codes (MCI, LIV, etc.) are visible\n"
                "• Send a BetPawa virtual screenshot",
                chat_id=processing_msg.chat.id,
                message_id=processing_msg.message_id
            )
            return
        
        bot.edit_message_text(
            f"🎯 Analyzing {len(matches)} matches with pattern recognition...",
            chat_id=processing_msg.chat.id,
            message_id=processing_msg.message_id
        )
        
        response = "🎮 *VIRTUAL GAME PREDICTIONS*\n"
        response += "🤖 Pattern recognition active\n\n"
        
        for match in matches:
            game_id = f"VIRT_{match['position']}_{int(time.time())}"
            pred = predictor.predict_match(
                game_id=game_id,
                home_team=match['home'],
                away_team=match['away']
            )
            
            conf_emoji = {
                'VERY HIGH': '🔮', 'HIGH': '📈', 
                'MEDIUM': '📊', 'LOW': '⚠️', 'LEARNING': '🔄'
            }.get(pred['confidence'], '📊')
            
            response += f"{conf_emoji} *{match['home']} vs {match['away']}*\n"
            response += f"   Prediction: {pred['outcome']} | Score: {pred['score']}\n"
            response += f"   Confidence: {pred['confidence']}\n"
            response += f"   Method: {pred['method_used']}\n\n"
        
        try:
            best_match = max(zip(matches, [predictor.predict_match(f"VIRT_{m['position']}_{int(time.time())}", m['home'], m['away']) for m in matches]), 
                             key=lambda x: {'VERY HIGH': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'LEARNING': 0}.get(x[1]['confidence'], 0))
            response += f"🔥 *BEST PREDICTION:* {best_match[0]['home']} vs {best_match[0]['away']} → {best_match[1]['outcome']}\n\n"
        except:
            pass
        
        response += predictor.get_statistics()
        response += "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += "⚠️ Virtual games follow algorithms but patterns can change.\n"
        response += "📈 Model improves with more data.\n"
        response += "🎲 Never bet more than you can afford."
        
        if len(response) > 4096:
            response = response[:4000] + "\n...(truncated)"
        
        bot.edit_message_text(
            response,
            chat_id=processing_msg.chat.id,
            message_id=processing_msg.message_id,
            parse_mode="Markdown"
        )
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        bot.edit_message_text(
            f"❌ Error: {str(e)[:100]}\n\nPlease try again.",
            chat_id=processing_msg.chat.id,
            message_id=processing_msg.message_id
        )

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(
        message,
        "📸 Please send a *screenshot* of BetPawa virtual games.\n\n"
        "Commands: /stats, /accuracy, /reset, /method",
        parse_mode="Markdown"
    )

# ============ WEBHOOK SETUP ============
@app.route("/" + TELEGRAM_TOKEN, methods=["POST"])
def webhook():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Virtual Football Predictor Bot - Active ✅", 200

def setup_webhook():
    bot.remove_webhook()
    webhook_url = f"{RENDER_URL}/{TELEGRAM_TOKEN}"
    bot.set_webhook(url=webhook_url)
    print(f"✅ Webhook set: {webhook_url}")
    print(f"✅ Bot active - Pattern recognition online")

if __name__ == "__main__":
    setup_webhook()
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
else:
    setup_webhook()
