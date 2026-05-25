import os, json, base64, hashlib, time, threading, sqlite3, requests
from io import BytesIO
from math import exp, factorial
from typing import Optional
from PIL import Image
import telebot
from flask import Flask, request
from openai import OpenAI

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

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
LEAGUE_AVG_HOME  = 1.45
LEAGUE_AVG_AWAY  = 1.10
HOME_ADV         = 1.15
BASE_LR          = 0.05
LR_DECAY         = 0.008
RATING_MIN       = 0.30
RATING_MAX       = 2.50
MIN_GAMES_TRUST  = 30     # don't trust accuracy stats below this
CONF_HIGH        = 0.62
CONF_MED         = 0.50   # below this: skip prediction output for that market

MODE_PREDICT = "predict"
MODE_RESULT  = "result"
user_mode: dict[int, str] = {}

DEFAULT_RATINGS = {
    "MCI": {"att": 1.55, "dfd": 0.65},
    "LIV": {"att": 1.48, "dfd": 0.70},
    "ARS": {"att": 1.38, "dfd": 0.72},
    "CHE": {"att": 1.28, "dfd": 0.80},
    "MUN": {"att": 1.22, "dfd": 0.85},
    "TOT": {"att": 1.18, "dfd": 0.90},
    "NEW": {"att": 1.10, "dfd": 0.92},
    "AST": {"att": 1.05, "dfd": 0.95},
    "BHA": {"att": 1.00, "dfd": 1.00},
    "BRE": {"att": 0.95, "dfd": 1.02},
    "WHU": {"att": 0.90, "dfd": 1.05},
    "CRY": {"att": 0.85, "dfd": 1.08},
    "FUL": {"att": 0.82, "dfd": 1.10},
    "EVE": {"att": 0.78, "dfd": 1.15},
    "WOL": {"att": 0.75, "dfd": 1.18},
    "BOU": {"att": 0.72, "dfd": 1.20},
    "NOT": {"att": 0.70, "dfd": 1.22},
    "LEE": {"att": 0.65, "dfd": 1.25},
    "BUR": {"att": 0.62, "dfd": 1.28},
    "SUN": {"att": 0.58, "dfd": 1.32},
}

# ─── DATABASE ────────────────────────────────────────────────────────────────
class DB:
    def __init__(self, path="predictor.db"):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def ex(self, sql, p=()):
        with self._lock:
            return self._conn.execute(sql, p)

    def commit(self):
        with self._lock:
            self._conn.commit()

    def one(self, sql, p=()):
        with self._lock:
            return self._conn.execute(sql, p).fetchone()

    def all(self, sql, p=()):
        with self._lock:
            return self._conn.execute(sql, p).fetchall()

    def _migrate(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS ratings (
                    team        TEXT PRIMARY KEY,
                    att         REAL NOT NULL,
                    dfd         REAL NOT NULL,
                    games_seen  INTEGER DEFAULT 0,
                    updated     INTEGER
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id     TEXT UNIQUE,
                    home        TEXT,
                    away        TEXT,
                    p_outcome   TEXT,
                    p_ou        TEXT,
                    p_btts      TEXT,
                    p_dw_home   TEXT,
                    p_dw_away   TEXT,
                    xg_home     REAL,
                    xg_away     REAL,
                    ts          INTEGER
                );
                CREATE TABLE IF NOT EXISTS results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id     TEXT UNIQUE,
                    home        TEXT,
                    away        TEXT,
                    hg          INTEGER,
                    ag          INTEGER,
                    outcome     TEXT,
                    ou          TEXT,
                    btts        TEXT,
                    dw_home     TEXT,
                    dw_away     TEXT,
                    c_outcome   INTEGER DEFAULT 0,
                    c_ou        INTEGER DEFAULT 0,
                    c_btts      INTEGER DEFAULT 0,
                    c_dw_home   INTEGER DEFAULT 0,
                    c_dw_away   INTEGER DEFAULT 0,
                    had_pred    INTEGER DEFAULT 0,
                    ts          INTEGER
                );
                CREATE TABLE IF NOT EXISTS h2h (
                    id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    home  TEXT, away TEXT,
                    hg    INTEGER, ag INTEGER,
                    ts    INTEGER
                );
            """)
            self._conn.commit()

db = DB()

def seed_ratings():
    for team, r in DEFAULT_RATINGS.items():
        db.ex("""INSERT OR IGNORE INTO ratings (team,att,dfd,games_seen,updated)
                 VALUES (?,?,?,0,?)""", (team, r["att"], r["dfd"], int(time.time())))
    db.commit()

seed_ratings()

# ─── RATING HELPERS ──────────────────────────────────────────────────────────
def get_rating(team: str) -> dict:
    row = db.one("SELECT att,dfd FROM ratings WHERE team=?", (team,))
    return {"att": row["att"], "dfd": row["dfd"]} if row else {"att": 1.0, "dfd": 1.0}

def update_ratings(home, away, hg, ag):
    hr = get_rating(home)
    ar = get_rating(away)
    xg_h = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADV
    xg_a = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
    h_err = hg - xg_h
    a_err = ag - xg_a
    hrow  = db.one("SELECT games_seen FROM ratings WHERE team=?", (home,))
    arow  = db.one("SELECT games_seen FROM ratings WHERE team=?", (away,))
    lr_h  = BASE_LR / (1 + (hrow["games_seen"] if hrow else 0) * LR_DECAY)
    lr_a  = BASE_LR / (1 + (arow["games_seen"] if arow else 0) * LR_DECAY)
    clamp = lambda v: max(RATING_MIN, min(RATING_MAX, v))
    now   = int(time.time())
    db.ex("""UPDATE ratings SET att=?,dfd=?,games_seen=games_seen+1,updated=?
             WHERE team=?""",
          (clamp(hr["att"] + lr_h * h_err),
           clamp(hr["dfd"] - lr_h * a_err * 0.5), now, home))
    db.ex("""UPDATE ratings SET att=?,dfd=?,games_seen=games_seen+1,updated=?
             WHERE team=?""",
          (clamp(ar["att"] + lr_a * a_err),
           clamp(ar["dfd"] - lr_a * h_err * 0.5), now, away))
    db.commit()

# ─── MATHS ───────────────────────────────────────────────────────────────────
def pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam**k * exp(-lam)) / factorial(k)

def score_grid(xg_h, xg_a, max_g=7):
    """Returns full probability grid of scores."""
    return [[pmf(h, xg_h) * pmf(a, xg_a) for a in range(max_g+1)]
            for h in range(max_g+1)]

def outcome_probs(grid):
    p1 = sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if h > a)
    px = sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if h == a)
    p2 = sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if h < a)
    t  = p1 + px + p2
    return p1/t, px/t, p2/t

def ou_prob(grid, line=2.5):
    return sum(grid[h][a]
               for h in range(len(grid)) for a in range(len(grid[0]))
               if h + a > line)

def btts_p(xg_h, xg_a):
    return (1 - pmf(0, xg_h)) * (1 - pmf(0, xg_a))

def top3_scores(grid):
    scores = []
    for h in range(len(grid)):
        for a in range(len(grid[0])):
            scores.append((grid[h][a], h, a))
    scores.sort(reverse=True)
    return [(h, a, p) for p, h, a in scores[:3]]

def direct_win_prob(grid, side="home"):
    """Probability of winning by exactly 1 goal (direct win market)."""
    if side == "home":
        return sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if h - a == 1)
    else:
        return sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if a - h == 1)

def conf(prob) -> str:
    if prob >= CONF_HIGH: return "🔥HIGH"
    if prob >= CONF_MED:  return "📈MED"
    return "⚠️LOW"

# ─── PREDICTOR ───────────────────────────────────────────────────────────────
class Predictor:
    def xg(self, home, away):
        hr  = get_rating(home)
        ar  = get_rating(away)
        h   = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADV
        a   = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
        h2h = self._h2h(home, away)
        if h2h:
            h = 0.75*h + 0.25*h2h[0]
            a = 0.75*a + 0.25*h2h[1]
        h *= self._form(home)
        a *= self._form(away)
        return round(max(0.1, h), 3), round(max(0.1, a), 3)

    def predict(self, home, away):
        xg_h, xg_a = self.xg(home, away)
        grid        = score_grid(xg_h, xg_a)
        p1, px, p2  = outcome_probs(grid)
        p_over      = ou_prob(grid)
        p_btts      = btts_p(xg_h, xg_a)
        top3        = top3_scores(grid)
        p_dw_h      = direct_win_prob(grid, "home")
        p_dw_a      = direct_win_prob(grid, "away")

        outcome     = max(zip(["1","X","2"],[p1,px,p2]), key=lambda x:x[1])[0]
        outcome_p   = max(p1, px, p2)
        over_under  = "OVER 2.5"  if p_over >= 0.5 else "UNDER 2.5"
        ou_p        = p_over if p_over >= 0.5 else 1 - p_over
        btts        = "YES" if p_btts >= 0.5 else "NO"
        btts_p_val  = p_btts if p_btts >= 0.5 else 1 - p_btts

        return {
            "outcome":    outcome,   "outcome_p":  outcome_p,
            "over_under": over_under,"ou_p":       ou_p,
            "btts":       btts,      "btts_p":     btts_p_val,
            "top3":       top3,
            "xg_h":       xg_h,      "xg_a":       xg_a,
            "p1": p1, "px": px, "p2": p2,
            "dw_home_p":  p_dw_h,
            "dw_away_p":  p_dw_a,
        }

    def save(self, game_id, home, away, pred):
        dw_home = "WIN" if pred["dw_home_p"] >= 0.20 else "NO"
        dw_away = "WIN" if pred["dw_away_p"] >= 0.20 else "NO"
        db.ex("""INSERT OR REPLACE INTO predictions
                 (game_id,home,away,p_outcome,p_ou,p_btts,p_dw_home,p_dw_away,xg_home,xg_away,ts)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
              (game_id, home, away, pred["outcome"], pred["over_under"],
               pred["btts"], dw_home, dw_away,
               pred["xg_h"], pred["xg_a"], int(time.time())))
        db.commit()

    def record(self, game_id, hg, ag, source_home=None, source_away=None):
        outcome  = "1" if hg > ag else ("X" if hg == ag else "2")
        ou       = "OVER 2.5"  if hg + ag > 2 else "UNDER 2.5"
        btts     = "YES" if hg > 0 and ag > 0 else "NO"
        dw_home  = "WIN" if hg - ag == 1 else "NO"
        dw_away  = "WIN" if ag - hg == 1 else "NO"

        pred = db.one("""SELECT home,away,p_outcome,p_ou,p_btts,p_dw_home,p_dw_away
                         FROM predictions WHERE game_id=?""", (game_id,))
        home = source_home or (pred["home"] if pred else None)
        away = source_away or (pred["away"] if pred else None)

        co = cou = cbt = cdh = cda = had = 0
        if pred:
            had = 1
            co  = int(outcome == pred["p_outcome"])
            cou = int(ou      == pred["p_ou"])
            cbt = int(btts    == pred["p_btts"])
            cdh = int(dw_home == pred["p_dw_home"])
            cda = int(dw_away == pred["p_dw_away"])

        db.ex("""INSERT OR REPLACE INTO results
                 (game_id,home,away,hg,ag,outcome,ou,btts,dw_home,dw_away,
                  c_outcome,c_ou,c_btts,c_dw_home,c_dw_away,had_pred,ts)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (game_id, home, away, hg, ag, outcome, ou, btts,
               dw_home, dw_away, co, cou, cbt, cdh, cda, had, int(time.time())))

        if home and away:
            db.ex("INSERT INTO h2h (home,away,hg,ag,ts) VALUES (?,?,?,?,?)",
                  (home, away, hg, ag, int(time.time())))
            update_ratings(home, away, hg, ag)
        db.commit()
        return {"co":co,"cou":cou,"cbt":cbt,"cdh":cdh,"cda":cda,"had":had,
                "outcome":outcome,"ou":ou,"btts":btts}

    def accuracy(self):
        row = db.one("""SELECT COUNT(*) as n,
                               AVG(c_outcome) as oc, AVG(c_ou) as gc,
                               AVG(c_btts) as bc,
                               AVG(c_dw_home+c_dw_away)*0.5 as dc
                        FROM results WHERE had_pred=1""")
        if not row or row["n"] == 0:
            return None
        return {
            "n":       row["n"],
            "outcome": round(row["oc"]*100, 1),
            "ou":      round(row["gc"]*100, 1),
            "btts":    round(row["bc"]*100, 1),
            "dw":      round(row["dc"]*100, 1),
            "overall": round(((row["oc"]+row["gc"]+row["bc"])/3)*100, 1),
            "trusted": row["n"] >= MIN_GAMES_TRUST,
        }

    def _h2h(self, home, away):
        rows = db.all("""SELECT hg,ag FROM h2h WHERE home=? AND away=?
                         ORDER BY ts DESC LIMIT 10""", (home, away))
        if len(rows) < 3: return None
        w = [1.0 - i*0.09 for i in range(len(rows))]
        tw = sum(w)
        return (sum(r["hg"]*wi for r,wi in zip(rows,w))/tw,
                sum(r["ag"]*wi for r,wi in zip(rows,w))/tw)

    def _form(self, team, n=5):
        rows = db.all("""SELECT outcome,home FROM results
                         WHERE (home=? OR away=?) AND outcome IS NOT NULL
                         ORDER BY ts DESC LIMIT ?""", (team, team, n))
        if not rows: return 1.0
        pts = sum(
            1.0 if ((r["home"]==team and r["outcome"]=="1") or
                    (r["home"]!=team and r["outcome"]=="2"))
            else 0.5 if r["outcome"]=="X" else 0.0
            for r in rows
        )
        return 0.85 + (pts / len(rows)) * 0.30


predictor = Predictor()

# ─── VISION ──────────────────────────────────────────────────────────────────
PREDICT_PROMPT = (
    "You are a data extractor for BetPawa virtual football. "
    "From the screenshot extract each UPCOMING match (no scores shown). "
    "Return ONLY valid JSON, no markdown:\n"
    '{"matches":[{"position":1,"home":"MCI","away":"LIV"}]}\n'
    "Use 3-letter codes. Do NOT invent scores."
)
RESULT_PROMPT = (
    "You are a data extractor for BetPawa virtual football. "
    "From the results screenshot extract FINAL SCORES of completed matches. "
    "Return ONLY valid JSON, no markdown:\n"
    '{"results":[{"home":"MCI","away":"LIV","home_goals":2,"away_goals":1}]}\n'
    "Only include matches with a visible final score. Use 3-letter codes."
)

def _resize(img_bytes, max_dim=1024):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()

def _dl_image(file_id):
    info = bot.get_file(file_id)
    return requests.get(
        f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{info.file_path}",
        timeout=15).content

def _vision(img_bytes, prompt):
    b64 = base64.b64encode(_resize(img_bytes)).decode()
    try:
        resp = ai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            max_tokens=700, temperature=0.0,
            messages=[
                {"role":"system","content":prompt},
                {"role":"user","content":[
                    {"type":"text","text":"Extract data. Return JSON only."},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        s, e = raw.find("{"), raw.rfind("}")+1
        return json.loads(raw[s:e]) if s != -1 and e > 0 else None
    except Exception as ex:
        print(f"[VISION] {ex}")
        return None

# ─── FORMATTERS ──────────────────────────────────────────────────────────────
def pct(v): return f"{v:.1f}%" if v is not None else "N/A"

def pred_card(home, away, pred, game_id):
    o_icon = {"1":"🏠","X":"🤝","2":"✈️"}.get(pred["outcome"],"🔮")

    # Only show direct win if probability is meaningful
    dh_p = pred["dw_home_p"]
    da_p = pred["dw_away_p"]
    dw_line = ""
    if dh_p >= 0.20 or da_p >= 0.20:
        if dh_p >= da_p:
            dw_line = f"\n│ 🎯 DirectW: {home} by 1  {conf(dh_p)} ({dh_p:.0%})"
        else:
            dw_line = f"\n│ 🎯 DirectW: {away} by 1  {conf(da_p)} ({da_p:.0%})"

    return (
        f"{o_icon} *{home} v {away}*\n"
        f"├ 1X2   : *{pred['outcome']}*  {conf(pred['outcome_p'])}  "
        f"({pred['p1']:.0%}/{pred['px']:.0%}/{pred['p2']:.0%})\n"
        f"├ O/U   : *{pred['over_under']}*  {conf(pred['ou_p'])}\n"
        f"├ BTTS  : *{pred['btts']}*  {conf(pred['btts_p'])}\n"
        f"├ Scores: *{pred['top3'][0][0]}-{pred['top3'][0][1]}* ({pred['top3'][0][2]:.0%})  "
        f"{pred['top3'][1][0]}-{pred['top3'][1][1]} ({pred['top3'][1][2]:.0%})  "
        f"{pred['top3'][2][0]}-{pred['top3'][2][1]} ({pred['top3'][2][2]:.0%})\n"
        f"├ xG    : {pred['xg_h']}-{pred['xg_a']}"
        f"{dw_line}\n"
        f"└ 🆔`{game_id}`"
    )

def result_card(r, rec):
    had  = rec.get("had", 0)
    line = f"⚽ *{r['home']} {r['home_goals']}-{r['away_goals']} {r['away']}*"
    if had:
        line += (
            f"\n  1X2:{'✅' if rec['co'] else '❌'}"
            f" O/U:{'✅' if rec['cou'] else '❌'}"
            f" BTTS:{'✅' if rec['cbt'] else '❌'}"
            f" DW:{'✅' if rec['cdh'] or rec['cda'] else '❌'}"
        )
    else:
        line += "\n  _(no prior prediction — learned anyway)_"
    return line

# ─── HANDLERS ────────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start","help"])
def h_start(msg):
    bot.reply_to(msg,
        "🎮 *VIRTUAL PREDICTOR v5*\n\n"
        "*Modes*\n"
        "/predict — upcoming screenshot → predictions\n"
        "/result  — results screenshot → bot learns\n\n"
        "*Markets predicted*\n"
        "• 1X2 outcome\n"
        "• Over/Under 2.5\n"
        "• BTTS\n"
        "• Direct Win (win by 1 goal)\n"
        "• Most likely score\n\n"
        "*Commands*\n"
        "/stats   — accuracy report\n"
        "/ratings — live team ratings\n"
        "/history — last 20 results\n"
        "/record `game_id score` — manual entry\n"
        "/reset   — wipe everything\n\n"
        "⚠️ Virtual football only. Not financial advice.",
        parse_mode="Markdown")

@bot.message_handler(commands=["predict"])
def h_predict(msg):
    user_mode[msg.chat.id] = MODE_PREDICT
    bot.reply_to(msg, "📸 *PREDICT mode* — send upcoming matches screenshot.",
                 parse_mode="Markdown")

@bot.message_handler(commands=["result"])
def h_result(msg):
    user_mode[msg.chat.id] = MODE_RESULT
    bot.reply_to(msg, "📊 *RESULT mode* — send results screenshot.",
                 parse_mode="Markdown")

@bot.message_handler(commands=["stats"])
def h_stats(msg):
    acc = predictor.accuracy()
    if not acc:
        bot.reply_to(msg, "No matched results yet. Submit results screenshots first.")
        return
    trust = "" if acc["trusted"] else f"\n⚠️ Only {acc['n']} games — stats not reliable yet (need {MIN_GAMES_TRUST})"
    bot.reply_to(msg,
        f"📊 *ACCURACY* ({acc['n']} matched games)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"1X2      : {pct(acc['outcome'])}\n"
        f"O/U 2.5  : {pct(acc['ou'])}\n"
        f"BTTS     : {pct(acc['btts'])}\n"
        f"DirectWin: {pct(acc['dw'])}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Overall  : {pct(acc['overall'])}"
        f"{trust}",
        parse_mode="Markdown")

@bot.message_handler(commands=["ratings"])
def h_ratings(msg):
    rows = db.all("SELECT team,att,dfd,games_seen FROM ratings ORDER BY att DESC")
    if not rows:
        bot.reply_to(msg, "No ratings yet.")
        return
    lines = ["📐 *Team Ratings*\n`Team  ATT    DEF    G`\n━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        lines.append(f"`{r['team']}  {r['att']:.3f}  {r['dfd']:.3f}  ({r['games_seen']})`")
    bot.reply_to(msg, "\n".join(lines), parse_mode="Markdown")

@bot.message_handler(commands=["history"])
def h_history(msg):
    rows = db.all("""SELECT home,away,hg,ag,c_outcome,c_ou,c_btts,c_dw_home,c_dw_away,had_pred
                     FROM results ORDER BY ts DESC LIMIT 20""")
    if not rows:
        bot.reply_to(msg, "No history yet.")
        return
    lines = ["📋 *Last 20 Results*\n━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        score = f"{r['hg']}-{r['ag']}"
        if r["had_pred"]:
            chk = (f"1X2:{'✅' if r['c_outcome'] else '❌'}"
                   f" OU:{'✅' if r['c_ou'] else '❌'}"
                   f" BT:{'✅' if r['c_btts'] else '❌'}"
                   f" DW:{'✅' if r['c_dw_home'] or r['c_dw_away'] else '❌'}")
        else:
            chk = "_(no pred)_"
        lines.append(f"{r['home']} *{score}* {r['away']}  {chk}")
    bot.reply_to(msg, "\n".join(lines), parse_mode="Markdown")

@bot.message_handler(commands=["record"])
def h_record(msg):
    parts = msg.text.split()
    if len(parts) != 3:
        bot.reply_to(msg,
            "❌ Usage: `/record game_id score`\n_e.g._ `/record MCI_LIV_abc1 2-1`",
            parse_mode="Markdown")
        return
    _, game_id, score = parts
    try:
        hg, ag = map(int, score.split("-"))
        if hg < 0 or ag < 0: raise ValueError
    except ValueError:
        bot.reply_to(msg, "❌ Score format: `2-1`", parse_mode="Markdown")
        return
    rec = predictor.record(game_id, hg, ag)
    acc = predictor.accuracy()
    text = f"✅ *Recorded* `{game_id}` → {score}\n"
    if rec["had"]:
        text += (f"1X2:{'✅' if rec['co'] else '❌'} "
                 f"O/U:{'✅' if rec['cou'] else '❌'} "
                 f"BTTS:{'✅' if rec['cbt'] else '❌'} "
                 f"DW:{'✅' if rec['cdh'] or rec['cda'] else '❌'}\n")
    text += f"📈 Overall: {pct(acc['overall']) if acc else 'Building...'}"
    bot.reply_to(msg, text, parse_mode="Markdown")

@bot.message_handler(commands=["reset"])
def h_reset(msg):
    for t in ("ratings","predictions","results","h2h"):
        db.ex(f"DELETE FROM {t}")
    db.commit()
    seed_ratings()
    bot.reply_to(msg, "🔄 All data cleared. Fresh start.")

@bot.message_handler(content_types=["photo"])
def h_photo(msg):
    if user_mode.get(msg.chat.id, MODE_PREDICT) == MODE_RESULT:
        _do_results(msg)
    else:
        _do_predict(msg)

def _do_predict(msg):
    status = bot.reply_to(msg, "📷 Analysing…")
    try:
        img     = _dl_image(msg.photo[-1].file_id)
        matches = (_vision(img, PREDICT_PROMPT) or {}).get("matches")
        if not matches:
            bot.edit_message_text(
                "❌ No upcoming matches found.\nIf this is a results screen use /result first.",
                status.chat.id, status.message_id)
            return
        lines = ["🎮 *PREDICTIONS*\n━━━━━━━━━━━━━━━━━━\n"]
        for m in matches:
            home = m.get("home","???").upper()
            away = m.get("away","???").upper()
            seed = f"{home}_{away}_{int(time.time()//60)}"
            gid  = f"{home}_{away}_{hashlib.md5(seed.encode()).hexdigest()[:8]}"
            pred = predictor.predict(home, away)
            predictor.save(gid, home, away, pred)
            lines.append(pred_card(home, away, pred, gid))
            lines.append("")
        acc = predictor.accuracy()
        acc_str = (f"{pct(acc['overall'])} ({acc['n']}g)"
                   + ("" if acc["trusted"] else " ⚠️building")
                   ) if acc else "Building…"
        lines += [
            "━━━━━━━━━━━━━━━━━━",
            f"📈 Accuracy: {acc_str}",
            "📊 After matches: /result → send results screenshot",
            "⚠️ Not financial advice.",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as ex:
        bot.edit_message_text(f"⚠️ Error: {str(ex)[:120]}", status.chat.id, status.message_id)

def _do_results(msg):
    status = bot.reply_to(msg, "📊 Reading results…")
    try:
        img     = _dl_image(msg.photo[-1].file_id)
        results = (_vision(img, RESULT_PROMPT) or {}).get("results")
        if not results:
            bot.edit_message_text(
                "❌ No results found. Make sure final scores are visible.",
                status.chat.id, status.message_id)
            return
        lines = [f"📊 *RESULTS* ({len(results)} matches)\n━━━━━━━━━━━━━━━━━━\n"]
        for r in results:
            home = r["home"].upper()
            away = r["away"].upper()
            hg   = int(r["home_goals"])
            ag   = int(r["away_goals"])
            # try to match a saved prediction
            pred_row = db.one("""SELECT game_id FROM predictions
                                 WHERE home=? AND away=?
                                 ORDER BY ts DESC LIMIT 1""", (home, away))
            gid = (pred_row["game_id"] if pred_row
                   else f"{home}_{away}_{hashlib.md5(f'{home}{away}{int(time.time())}'.encode()).hexdigest()[:8]}")
            rec = predictor.record(gid, hg, ag, home, away)
            lines.append(result_card(r, rec))
            lines.append("")
        acc = predictor.accuracy()
        lines += [
            "━━━━━━━━━━━━━━━━━━",
            "📐 Ratings updated.",
            f"📈 Accuracy: {pct(acc['overall']) if acc else 'Building...'} "
            f"({acc['n'] if acc else 0} matched)",
            "\n📸 Next round: /predict",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as ex:
        bot.edit_message_text(f"⚠️ Error: {str(ex)[:120]}", status.chat.id, status.message_id)

@bot.message_handler(func=lambda m: True)
def h_text(msg):
    mode = user_mode.get(msg.chat.id, MODE_PREDICT)
    bot.reply_to(msg,
        f"Mode: *{'📸 PREDICT' if mode == MODE_PREDICT else '📊 RESULT'}*  |  /help",
        parse_mode="Markdown")

# ─── WEBHOOK ─────────────────────────────────────────────────────────────────
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    bot.process_new_updates([telebot.types.Update.de_json(request.get_data(as_text=True))])
    return "OK", 200

@app.route("/", methods=["GET"])
def health():
    return "Virtual Predictor v5 ✅", 200

def setup_webhook():
    bot.remove_webhook()
    time.sleep(0.5)
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[WEBHOOK] {RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    setup_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
