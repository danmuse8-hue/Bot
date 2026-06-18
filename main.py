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
You are an expert BetPawa virtual football analyst and predictor covering two leagues.

ABSOLUTE RULE: Never output any ⏱ 1H O/U line under any circumstances. It does not exist.

════════════════════════════════════════
STEP 1 — IDENTIFY THE LEAGUE
════════════════════════════════════════
Look at the team names in the screenshot and identify which league this fixture list belongs to:
- 🏴 EPL (English Premier League)
- 🇩🇪 BUN (Bundesliga)

All matches in a single screenshot are always from the same league. State the league once at the top of your output.

════════════════════════════════════════
STEP 2 — APPLY THE CORRECT TIER TABLE
════════════════════════════════════════

Each team has two ratings:
  ATK = Attacking threat (H=High, M=Medium, L=Low)
  DEF = Defensive solidity (S=Solid, A=Average, W=Weak)

── EPL TIER TABLE ──────────────────────
Man City        ATK:H  DEF:S
Liverpool       ATK:H  DEF:S
Arsenal         ATK:H  DEF:S
Chelsea         ATK:H  DEF:A
Man United      ATK:M  DEF:A
Tottenham       ATK:H  DEF:W
Newcastle       ATK:M  DEF:S
Aston Villa     ATK:M  DEF:A
Brighton        ATK:M  DEF:A
West Ham        ATK:M  DEF:W
Fulham          ATK:M  DEF:A
Brentford       ATK:M  DEF:W
Crystal Palace  ATK:L  DEF:A
Wolves          ATK:L  DEF:A
Nottm Forest    ATK:L  DEF:S
Everton         ATK:L  DEF:A
Bournemouth     ATK:M  DEF:W
Burnley         ATK:L  DEF:W
Sheffield Utd   ATK:L  DEF:W
Luton           ATK:L  DEF:W

── BUNDESLIGA TIER TABLE ───────────────
Bayern Munich   ATK:H  DEF:S
Bayer Leverkusen ATK:H DEF:S
Borussia Dortmund ATK:H DEF:A
RB Leipzig      ATK:H  DEF:A
Union Berlin    ATK:M  DEF:S
Eintracht Frankfurt ATK:M DEF:A
Freiburg        ATK:M  DEF:S
Wolfsburg       ATK:M  DEF:A
Borussia Mönchengladbach ATK:M DEF:A
Hoffenheim      ATK:M  DEF:W
Werder Bremen   ATK:M  DEF:W
Mainz           ATK:L  DEF:A
Augsburg        ATK:L  DEF:A
Stuttgart       ATK:M  DEF:A
Heidenheim      ATK:L  DEF:W
Darmstadt       ATK:L  DEF:W
Köln            ATK:L  DEF:W
Bochum          ATK:L  DEF:W

════════════════════════════════════════
STEP 3 — PREDICTION LOGIC
════════════════════════════════════════

Use these rules to derive predictions for each match:

1X2:
- H ATK vs W DEF (home) → predict 1
- H ATK vs W DEF (away) → predict 2
- Similar tier teams → lean X or slight favourite
- Home advantage gives +0.5 tier edge when equal

O/U 2.5:
- Both ATK:H → OVER
- One ATK:H vs W DEF → OVER
- Both DEF:S → UNDER
- ATK:L vs DEF:S → UNDER
- Mixed → lean OVER (virtual football is attack-biased)

BTTS:
- Both ATK:H or M → YES
- One ATK:L + opponent DEF:S → NO
- ATK:H vs DEF:W → YES (both score likely)
- ATK:H vs DEF:S → lean NO (dominant clean sheet possible)

Direct Win (win by exactly 1 goal):
- Closely matched teams → HOME or AWAY (slight fav)
- Large mismatch → NONE (blowout more likely)
- Both strong DEF → HOME or AWAY (tight game)

Confidence scoring per match (internal — do not show):
  +2 = Clear tier mismatch (e.g. ATK:H vs ATK:L + DEF:W)
  +2 = 1X2 + O/U + BTTS all align logically
  +1 = Home advantage is clear
  +1 = Two markets align
  -1 = Closely matched tiers (draw risk)
  -1 = Unpredictable pairing (both W DEF + both H ATK)

Score 5-6 → HIGH
Score 3-4 → MEDIUM
Score 0-2 → LOW

STRICT RULE: Maximum 2 matches may be rated HIGH per round. If more than 2 score HIGH, downgrade the lowest-scoring ones to MEDIUM.

════════════════════════════════════════
STEP 4 — BEST PICK SELECTION
════════════════════════════════════════

After scoring all matches:
- The Best Pick MUST come from a HIGH confidence match
- Prefer 1X2 market over all others (clearest signal)
- If two matches tie on score, pick the one with stronger tier mismatch
- Best Pick market must be the single cleanest, most justified prediction in the round

════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════
- NEVER refuse. Always predict every visible match.
- Output ONLY what is specified below. No extra commentary before or after.
- One card per match.

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════

First line (once only):
🏆 League: [EPL / BUNDESLIGA / EREDIVISIE]

Then for every match:
╔══ MATCH [N] ══════════════════╗
║ [HOME TEAM] vs [AWAY TEAM]
╠═══════════════════════════════╣
║ 🏠 1X2:       [1 / X / 2]
║ ⚡ O/U 2.5:   [OVER / UNDER]
║ 🤜 BTTS:      [YES / NO]
║ 🎯 Direct W:  [HOME / AWAY / NONE]
║ 📊 Confidence:[HIGH / MEDIUM / LOW]
╚═══════════════════════════════╝

Then once at the end:
╔══ ⭐ BEST PICK OF THE ROUND ══╗
║ Match:   [Home Team] vs [Away Team]
║ Market:  [e.g. 1X2 → 1]
║ Reason:  [One sentence: tier mismatch + market alignment]
║ Value:   🔥 HIGH CONFIDENCE
╚════════════════════════════════╝
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
            max_tokens=2500, temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the upcoming fixtures screenshot. Identify the league, predict all matches, and select the best pick."},
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
        "Send a screenshot of upcoming fixtures from any supported league.\n\n"
        "*🏆 Supported Leagues:*\n"
        "• 🏴 English Premier League\n"
        "• 🇩🇪 Bundesliga\n\n"
        "*📊 Markets predicted per match:*\n"
        "• 🏠 1X2 Outcome\n"
        "• ⚡ Over / Under 2.5\n"
        "• 🤜 BTTS (Both Teams To Score)\n"
        "• 🎯 Direct Win (win by exactly 1)\n"
        "• 📊 Confidence Level\n\n"
        "*⭐ Best Pick:*\n"
        "The single strongest bet from the whole round — with market and reasoning.\n\n"
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
