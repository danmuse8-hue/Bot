import os, json, base64, hashlib, time, threading, sqlite3, requests
from io import BytesIO
from math import exp, factorial, log, sqrt
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
LEAGUE_AVG_HOME   = 1.45
LEAGUE_AVG_AWAY   = 1.10
HOME_ADV          = 1.15

# Faster learning: higher base LR, gentler decay, aggressive early boost
BASE_LR           = 0.12       # was 0.05 — 2.4x faster initial learning
LR_DECAY          = 0.004      # was 0.008 — slower decay so it stays aggressive longer
EARLY_BOOST       = 3.0        # multiplier for first 5 games (cold-start boost)
EARLY_GAME_LIMIT  = 5

RATING_MIN        = 0.25
RATING_MAX        = 2.80
MIN_GAMES_TRUST   = 15         # was 30 — trust stats earlier
CONF_HIGH         = 0.60
CONF_MED          = 0.48

# 1H split: virtual games score ~55% of goals in 1st half
HALF_SPLIT        = 0.55

MODE_PREDICT = "predict"
MODE_RESULT  = "result"
user_mode: dict[int, str] = {}

# Premium team tier list for virtual BetPawa
DEFAULT_RATINGS = {
    "MCI": {"att": 1.62, "dfd": 0.60, "h_att": 1.68, "h_dfd": 0.58},
    "LIV": {"att": 1.55, "dfd": 0.64, "h_att": 1.60, "h_dfd": 0.62},
    "ARS": {"att": 1.42, "dfd": 0.68, "h_att": 1.46, "h_dfd": 0.66},
    "CHE": {"att": 1.32, "dfd": 0.76, "h_att": 1.35, "h_dfd": 0.74},
    "MUN": {"att": 1.25, "dfd": 0.82, "h_att": 1.28, "h_dfd": 0.80},
    "TOT": {"att": 1.20, "dfd": 0.88, "h_att": 1.24, "h_dfd": 0.86},
    "NEW": {"att": 1.12, "dfd": 0.90, "h_att": 1.15, "h_dfd": 0.88},
    "AST": {"att": 1.08, "dfd": 0.94, "h_att": 1.10, "h_dfd": 0.92},
    "BHA": {"att": 1.02, "dfd": 0.98, "h_att": 1.04, "h_dfd": 0.96},
    "BRE": {"att": 0.96, "dfd": 1.02, "h_att": 0.98, "h_dfd": 1.00},
    "WHU": {"att": 0.92, "dfd": 1.05, "h_att": 0.94, "h_dfd": 1.03},
    "CRY": {"att": 0.87, "dfd": 1.08, "h_att": 0.89, "h_dfd": 1.06},
    "FUL": {"att": 0.84, "dfd": 1.10, "h_att": 0.86, "h_dfd": 1.08},
    "EVE": {"att": 0.80, "dfd": 1.14, "h_att": 0.82, "h_dfd": 1.12},
    "WOL": {"att": 0.77, "dfd": 1.17, "h_att": 0.79, "h_dfd": 1.15},
    "BOU": {"att": 0.74, "dfd": 1.20, "h_att": 0.76, "h_dfd": 1.18},
    "NOT": {"att": 0.71, "dfd": 1.22, "h_att": 0.73, "h_dfd": 1.20},
    "LEE": {"att": 0.67, "dfd": 1.26, "h_att": 0.69, "h_dfd": 1.24},
    "BUR": {"att": 0.63, "dfd": 1.29, "h_att": 0.65, "h_dfd": 1.27},
    "SUN": {"att": 0.59, "dfd": 1.33, "h_att": 0.61, "h_dfd": 1.31},
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
                    h_att       REAL NOT NULL DEFAULT 1.0,
                    h_dfd       REAL NOT NULL DEFAULT 1.0,
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
                    p_h1ou      TEXT,
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
                    h1ou        TEXT,
                    dw_home     TEXT,
                    dw_away     TEXT,
                    c_outcome   INTEGER DEFAULT 0,
                    c_ou        INTEGER DEFAULT 0,
                    c_btts      INTEGER DEFAULT 0,
                    c_h1ou      INTEGER DEFAULT 0,
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
            # Add new columns if upgrading from older schema
            for col, defval in [
                ("h_att", "1.0"), ("h_dfd", "1.0"),
                ("p_h1ou", "'OVER 0.5'"), ("h1ou", "NULL"),
                ("c_h1ou", "0"),
            ]:
                try:
                    if col in ("h_att", "h_dfd"):
                        self._conn.execute(f"ALTER TABLE ratings ADD COLUMN {col} REAL NOT NULL DEFAULT {defval}")
                    elif col in ("p_h1ou",):
                        self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT")
                    elif col in ("h1ou",):
                        self._conn.execute(f"ALTER TABLE results ADD COLUMN {col} TEXT")
                    elif col in ("c_h1ou",):
                        self._conn.execute(f"ALTER TABLE results ADD COLUMN {col} INTEGER DEFAULT 0")
                except:
                    pass
            self._conn.commit()

db = DB()

def seed_ratings():
    for team, r in DEFAULT_RATINGS.items():
        db.ex("""INSERT OR IGNORE INTO ratings (team,att,dfd,h_att,h_dfd,games_seen,updated)
                 VALUES (?,?,?,?,?,0,?)""",
              (team, r["att"], r["dfd"], r["h_att"], r["h_dfd"], int(time.time())))
    db.commit()

seed_ratings()

# ─── RATING HELPERS ──────────────────────────────────────────────────────────
def get_rating(team: str) -> dict:
    row = db.one("SELECT att,dfd,h_att,h_dfd,games_seen FROM ratings WHERE team=?", (team,))
    if row:
        return {"att": row["att"], "dfd": row["dfd"],
                "h_att": row["h_att"], "h_dfd": row["h_dfd"],
                "games_seen": row["games_seen"]}
    return {"att": 1.0, "dfd": 1.0, "h_att": 1.0, "h_dfd": 1.0, "games_seen": 0}

def update_ratings(home, away, hg, ag):
    hr = get_rating(home)
    ar = get_rating(away)

    xg_h = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADV
    xg_a = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
    h_err = hg - xg_h
    a_err = ag - xg_a

    # 1H proxy errors (estimate ~55% of goals fell in 1st half)
    h1_h_err = (hg * HALF_SPLIT) - (xg_h * HALF_SPLIT)
    h1_a_err = (ag * HALF_SPLIT) - (xg_a * HALF_SPLIT)

    hg_seen = hr["games_seen"]
    ag_seen = ar["games_seen"]

    # Early-game cold-start boost
    h_boost = EARLY_BOOST if hg_seen < EARLY_GAME_LIMIT else 1.0
    a_boost = EARLY_BOOST if ag_seen < EARLY_GAME_LIMIT else 1.0

    lr_h = BASE_LR * h_boost / (1 + hg_seen * LR_DECAY)
    lr_a = BASE_LR * a_boost / (1 + ag_seen * LR_DECAY)

    clamp = lambda v: max(RATING_MIN, min(RATING_MAX, v))
    now   = int(time.time())

    db.ex("""UPDATE ratings SET
             att=?, dfd=?, h_att=?, h_dfd=?,
             games_seen=games_seen+1, updated=?
             WHERE team=?""",
          (clamp(hr["att"] + lr_h * h_err),
           clamp(hr["dfd"] - lr_h * a_err * 0.5),
           clamp(hr["h_att"] + lr_h * h1_h_err),
           clamp(hr["h_dfd"] - lr_h * h1_a_err * 0.5),
           now, home))

    db.ex("""UPDATE ratings SET
             att=?, dfd=?, h_att=?, h_dfd=?,
             games_seen=games_seen+1, updated=?
             WHERE team=?""",
          (clamp(ar["att"] + lr_a * a_err),
           clamp(ar["dfd"] - lr_a * h_err * 0.5),
           clamp(ar["h_att"] + lr_a * h1_a_err),
           clamp(ar["h_dfd"] - lr_a * h1_h_err * 0.5),
           now, away))
    db.commit()

# ─── MATHS ───────────────────────────────────────────────────────────────────
def pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam**k * exp(-lam)) / factorial(k)

def score_grid(xg_h, xg_a, max_g=8):
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

def top_scores(grid, n=5):
    scores = [(grid[h][a], h, a)
              for h in range(len(grid)) for a in range(len(grid[0]))]
    scores.sort(reverse=True)
    return [(h, a, p) for p, h, a in scores[:n]]

def direct_win_prob(grid, side="home"):
    if side == "home":
        return sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if h - a == 1)
    return sum(grid[h][a] for h in range(len(grid)) for a in range(len(grid[0])) if a - h == 1)

def h1_ou_prob(xg_h1, xg_a1, line=0.5):
    """Probability of over X goals in first half."""
    p_zero = pmf(0, xg_h1) * pmf(0, xg_a1)
    return 1 - p_zero

def confidence_label(prob) -> tuple[str, str]:
    """Returns (icon, label)."""
    if prob >= 0.72:  return "🔥", "ELITE"
    if prob >= 0.62:  return "⚡", "HIGH"
    if prob >= 0.50:  return "📈", "MED"
    return "⚠️", "LOW"

def conf_bar(prob, width=8) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)

# ─── PREDICTOR ───────────────────────────────────────────────────────────────
class Predictor:

    def xg(self, home, away):
        hr  = get_rating(home)
        ar  = get_rating(away)
        h   = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADV
        a   = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
        # 1H xG using half-specific ratings
        h1h = LEAGUE_AVG_HOME * hr["h_att"] * ar["h_dfd"] * HOME_ADV * HALF_SPLIT
        h1a = LEAGUE_AVG_AWAY * ar["h_att"] * hr["h_dfd"] * HALF_SPLIT
        # Blend with H2H if available
        h2h = self._h2h(home, away)
        if h2h:
            h   = 0.72*h   + 0.28*h2h[0]
            a   = 0.72*a   + 0.28*h2h[1]
            h1h = 0.72*h1h + 0.28*(h2h[0]*HALF_SPLIT)
            h1a = 0.72*h1a + 0.28*(h2h[1]*HALF_SPLIT)
        # Form adjustment
        hf = self._form(home)
        af = self._form(away)
        h   *= hf;  a   *= af
        h1h *= hf;  h1a *= af
        return (round(max(0.1, h), 3), round(max(0.1, a), 3),
                round(max(0.05, h1h), 3), round(max(0.05, h1a), 3))

    def predict(self, home, away):
        xg_h, xg_a, xg_h1, xg_a1 = self.xg(home, away)
        grid   = score_grid(xg_h, xg_a)
        p1, px, p2 = outcome_probs(grid)
        p_over = ou_prob(grid)
        p_btts = btts_p(xg_h, xg_a)
        top5   = top_scores(grid, 5)
        p_dw_h = direct_win_prob(grid, "home")
        p_dw_a = direct_win_prob(grid, "away")
        p_h1ou = h1_ou_prob(xg_h1, xg_a1, line=0.5)

        outcome    = max(zip(["1","X","2"],[p1,px,p2]), key=lambda x:x[1])[0]
        outcome_p  = max(p1, px, p2)
        over_under = "OVER 2.5"  if p_over >= 0.5 else "UNDER 2.5"
        ou_p       = p_over if p_over >= 0.5 else 1 - p_over
        btts       = "YES" if p_btts >= 0.5 else "NO"
        btts_pv    = p_btts if p_btts >= 0.5 else 1 - p_btts
        h1ou       = "OVER 0.5" if p_h1ou >= 0.5 else "UNDER 0.5"
        h1ou_p     = p_h1ou if p_h1ou >= 0.5 else 1 - p_h1ou

        return {
            "outcome": outcome, "outcome_p": outcome_p,
            "over_under": over_under, "ou_p": ou_p,
            "btts": btts, "btts_p": btts_pv,
            "h1ou": h1ou, "h1ou_p": h1ou_p,
            "top5": top5,
            "xg_h": xg_h, "xg_a": xg_a,
            "xg_h1": xg_h1, "xg_a1": xg_a1,
            "p1": p1, "px": px, "p2": p2,
            "dw_home_p": p_dw_h, "dw_away_p": p_dw_a,
        }

    def save(self, game_id, home, away, pred):
        dw_home = "WIN" if pred["dw_home_p"] >= 0.18 else "NO"
        dw_away = "WIN" if pred["dw_away_p"] >= 0.18 else "NO"
        db.ex("""INSERT OR REPLACE INTO predictions
                 (game_id,home,away,p_outcome,p_ou,p_btts,p_h1ou,p_dw_home,p_dw_away,xg_home,xg_away,ts)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
              (game_id, home, away, pred["outcome"], pred["over_under"],
               pred["btts"], pred["h1ou"], dw_home, dw_away,
               pred["xg_h"], pred["xg_a"], int(time.time())))
        db.commit()

    def record(self, game_id, hg, ag, source_home=None, source_away=None):
        outcome = "1" if hg > ag else ("X" if hg == ag else "2")
        ou      = "OVER 2.5"  if hg + ag > 2 else "UNDER 2.5"
        btts    = "YES" if hg > 0 and ag > 0 else "NO"
        # 1H over proxy: if total goals >= 2 it's very likely there was 1 in first half
        h1ou    = "OVER 0.5"  if hg + ag >= 1 else "UNDER 0.5"
        dw_home = "WIN" if hg - ag == 1 else "NO"
        dw_away = "WIN" if ag - hg == 1 else "NO"

        pred = db.one("""SELECT home,away,p_outcome,p_ou,p_btts,p_h1ou,p_dw_home,p_dw_away
                         FROM predictions WHERE game_id=?""", (game_id,))
        home = source_home or (pred["home"] if pred else None)
        away = source_away or (pred["away"] if pred else None)

        co = cou = cbt = ch1 = cdh = cda = had = 0
        if pred:
            had = 1
            co  = int(outcome == pred["p_outcome"])
            cou = int(ou      == pred["p_ou"])
            cbt = int(btts    == pred["p_btts"])
            ch1 = int(h1ou    == (pred["p_h1ou"] or "OVER 0.5"))
            cdh = int(dw_home == pred["p_dw_home"])
            cda = int(dw_away == pred["p_dw_away"])

        db.ex("""INSERT OR REPLACE INTO results
                 (game_id,home,away,hg,ag,outcome,ou,btts,h1ou,dw_home,dw_away,
                  c_outcome,c_ou,c_btts,c_h1ou,c_dw_home,c_dw_away,had_pred,ts)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (game_id, home, away, hg, ag, outcome, ou, btts, h1ou,
               dw_home, dw_away, co, cou, cbt, ch1, cdh, cda, had, int(time.time())))

        if home and away:
            db.ex("INSERT INTO h2h (home,away,hg,ag,ts) VALUES (?,?,?,?,?)",
                  (home, away, hg, ag, int(time.time())))
            update_ratings(home, away, hg, ag)
        db.commit()
        return {"co":co,"cou":cou,"cbt":cbt,"ch1":ch1,"cdh":cdh,"cda":cda,
                "had":had,"outcome":outcome,"ou":ou,"btts":btts,"h1ou":h1ou}

    def accuracy(self):
        row = db.one("""SELECT COUNT(*) as n,
                               AVG(c_outcome) as oc, AVG(c_ou) as gc,
                               AVG(c_btts)    as bc, AVG(c_h1ou) as hc,
                               AVG(c_dw_home+c_dw_away)*0.5 as dc
                        FROM results WHERE had_pred=1""")
        if not row or row["n"] == 0:
            return None
        return {
            "n":       row["n"],
            "outcome": round(row["oc"]*100, 1),
            "ou":      round(row["gc"]*100, 1),
            "btts":    round(row["bc"]*100, 1),
            "h1ou":    round(row["hc"]*100, 1),
            "dw":      round(row["dc"]*100, 1),
            "overall": round(((row["oc"]+row["gc"]+row["bc"]+row["hc"])/4)*100, 1),
            "trusted": row["n"] >= MIN_GAMES_TRUST,
        }

    def streak(self):
        """Current win/loss streak on outcome market."""
        rows = db.all("""SELECT c_outcome FROM results
                         WHERE had_pred=1 ORDER BY ts DESC LIMIT 10""")
        if not rows: return ""
        streak, val = 0, rows[0]["c_outcome"]
        for r in rows:
            if r["c_outcome"] == val: streak += 1
            else: break
        icon = "🔥" if val == 1 else "❄️"
        label = "W" if val == 1 else "L"
        return f"{icon}{streak}{label}"

    def _h2h(self, home, away):
        rows = db.all("""SELECT hg,ag FROM h2h WHERE home=? AND away=?
                         ORDER BY ts DESC LIMIT 8""", (home, away))
        if len(rows) < 2: return None
        w  = [1.0 - i*0.10 for i in range(len(rows))]
        tw = sum(w)
        return (sum(r["hg"]*wi for r,wi in zip(rows,w))/tw,
                sum(r["ag"]*wi for r,wi in zip(rows,w))/tw)

    def _form(self, team, n=6):
        rows = db.all("""SELECT outcome,home FROM results
                         WHERE (home=? OR away=?) AND outcome IS NOT NULL
                         ORDER BY ts DESC LIMIT ?""", (team, team, n))
        if not rows: return 1.0
        # Exponential recency weighting
        weights = [exp(-0.25*i) for i in range(len(rows))]
        tw = sum(weights)
        pts = sum(
            (1.0 if ((r["home"]==team and r["outcome"]=="1") or
                     (r["home"]!=team and r["outcome"]=="2"))
             else 0.5 if r["outcome"]=="X" else 0.0) * w
            for r, w in zip(rows, weights)
        )
        norm = pts / tw   # 0.0 → 1.0
        return 0.82 + norm * 0.36   # range 0.82–1.18


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

def _vision(img_bytes, prompt):
    b64 = base64.b64encode(_resize(img_bytes)).decode()
    try:
        resp = ai_client.chat.completions.create(
            model="openai/gpt-4o",
            max_tokens=800, temperature=0.0,
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
def pct(v): return f"{v:.1f}%" if v is not None else "—"

TIER = {
    "MCI":"👑","LIV":"🔴","ARS":"🔺","CHE":"💙","MUN":"👹","TOT":"⚪",
    "NEW":"⚫","AST":"🦁","BHA":"🐦","BRE","🐝","WHU":"⚒️","CRY":"🦅",
    "FUL":"⚪","EVE":"💙","WOL":"🐺","BOU":"🍒","NOT":"🌲","LEE":"⚪",
    "BUR":"🔵","SUN":"⚫",
}

def team_icon(code): return TIER.get(code, "⚽")

def outcome_icon(o): return {"1":"🏠","X":"🤝","2":"✈️"}.get(o, "🔮")

def pred_card(home, away, pred, game_id, idx):
    ci, cl  = confidence_label(pred["outcome_p"])
    bar     = conf_bar(pred["outcome_p"])
    o_icon  = outcome_icon(pred["outcome"])
    hi      = team_icon(home)
    ai      = team_icon(away)

    dw_line = ""
    dh_p, da_p = pred["dw_home_p"], pred["dw_away_p"]
    if dh_p >= 0.18 or da_p >= 0.18:
        best_p    = max(dh_p, da_p)
        best_team = home if dh_p >= da_p else away
        dw_ci, _  = confidence_label(best_p)
        dw_line   = f"\n│  🎯 *Direct Win:* {best_team} by 1   {dw_ci} {best_p:.0%}"

    h1ci, _ = confidence_label(pred["h1ou_p"])

    scores_line = "  ".join(
        f"`{h}-{a}`({p:.0%})" for h, a, p in pred["top5"][:3]
    )

    return (
        f"╔══ *MATCH {idx}* ══════════════════╗\n"
        f"║  {hi} *{home}*  vs  *{away}* {ai}\n"
        f"╠══════════════════════════════╣\n"
        f"║  {o_icon} *1X2:*  `{pred['outcome']}`  {ci} *{cl}*\n"
        f"║     {bar}  {pred['outcome_p']:.0%}\n"
        f"║     Split:  1:{pred['p1']:.0%}  X:{pred['px']:.0%}  2:{pred['p2']:.0%}\n"
        f"╠══════════════════════════════╣\n"
        f"║  ⚡ *O/U 2.5:* `{pred['over_under']}`  {conf_bar(pred['ou_p'],6)}  {pred['ou_p']:.0%}\n"
        f"║  🤜 *BTTS:*   `{pred['btts']}`  {conf_bar(pred['btts_p'],6)}  {pred['btts_p']:.0%}\n"
        f"║  ⏱ *1H O/U:* `{pred['h1ou']}`  {h1ci}  {pred['h1ou_p']:.0%}\n"
        f"╠══════════════════════════════╣\n"
        f"║  📐 *xG:* {pred['xg_h']} – {pred['xg_a']}  "
        f"│  *1H xG:* {pred['xg_h1']} – {pred['xg_a1']}\n"
        f"║  🎲 *Scores:* {scores_line}"
        f"{dw_line}\n"
        f"╚══ 🆔`{game_id}`\n"
    )

def result_card(r, rec, idx):
    had  = rec.get("had", 0)
    line = (f"⚽ *{idx}. {r['home']} {r['home_goals']}–{r['away_goals']} {r['away']}*")
    if had:
        checks = (
            f"1X2:{'✅' if rec['co'] else '❌'} "
            f"OU:{'✅' if rec['cou'] else '❌'} "
            f"BTTS:{'✅' if rec['cbt'] else '❌'} "
            f"1H:{'✅' if rec['ch1'] else '❌'} "
            f"DW:{'✅' if rec['cdh'] or rec['cda'] else '❌'}"
        )
        line += f"\n    {checks}"
    else:
        line += "\n    _(no prior prediction — data logged)_"
    return line

def acc_block(acc):
    if not acc: return "📊 Accuracy: _building..._"
    trust_note = "" if acc["trusted"] else f"  ⚠️ _{acc['n']}/{MIN_GAMES_TRUST} games_"
    streak_str = predictor.streak()
    return (
        f"╔══ 📊 *ACCURACY* ({acc['n']} games) {streak_str}  ══╗\n"
        f"║  1X2      {conf_bar(acc['outcome']/100)}  {pct(acc['outcome'])}\n"
        f"║  O/U 2.5  {conf_bar(acc['ou']/100)}  {pct(acc['ou'])}\n"
        f"║  BTTS     {conf_bar(acc['btts']/100)}  {pct(acc['btts'])}\n"
        f"║  1H O/U   {conf_bar(acc['h1ou']/100)}  {pct(acc['h1ou'])}\n"
        f"║  Direct W {conf_bar(acc['dw']/100)}  {pct(acc['dw'])}\n"
        f"╠══════════════════════════════╣\n"
        f"║  Overall  {conf_bar(acc['overall']/100)}  *{pct(acc['overall'])}*{trust_note}\n"
        f"╚══════════════════════════════╝"
    )

# ─── HANDLERS ────────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start","help"])
def h_start(msg):
    bot.reply_to(msg,
        "┌─────────────────────────────┐\n"
        "│  ⚡ *VIRTUAL PREDICTOR v6*  │\n"
        "└─────────────────────────────┘\n\n"
        "*📸 How to use:*\n"
        "1️⃣ `/predict` → send upcoming matches screenshot\n"
        "2️⃣ `/result`  → send results screenshot to learn\n\n"
        "*📊 Markets:*\n"
        "• 1X2 Outcome\n"
        "• Over / Under 2.5\n"
        "• BTTS (Both Teams To Score)\n"
        "• 1st Half Over 0.5 goals\n"
        "• Direct Win (win by exactly 1)\n"
        "• Top 3 most likely scores\n\n"
        "*🛠 Commands:*\n"
        "`/stats`   — full accuracy report\n"
        "`/ratings` — live team strength table\n"
        "`/history` — last 20 results\n"
        "`/record game_id 2-1` — manual entry\n"
        "`/reset`   — wipe all data\n\n"
        "⚠️ _Virtual football only. Not financial advice._",
        parse_mode="Markdown")

@bot.message_handler(commands=["predict"])
def h_predict(msg):
    user_mode[msg.chat.id] = MODE_PREDICT
    bot.reply_to(msg,
        "📸 *PREDICT MODE*\nSend the upcoming matches screenshot now.",
        parse_mode="Markdown")

@bot.message_handler(commands=["result"])
def h_result(msg):
    user_mode[msg.chat.id] = MODE_RESULT
    bot.reply_to(msg,
        "📊 *RESULT MODE*\nSend the results screenshot now.",
        parse_mode="Markdown")

@bot.message_handler(commands=["stats"])
def h_stats(msg):
    acc = predictor.accuracy()
    if not acc:
        bot.reply_to(msg,
            "No matched results yet.\nUse `/result` and send a results screenshot first.",
            parse_mode="Markdown")
        return
    bot.reply_to(msg, acc_block(acc), parse_mode="Markdown")

@bot.message_handler(commands=["ratings"])
def h_ratings(msg):
    rows = db.all("""SELECT team,att,dfd,h_att,games_seen
                     FROM ratings ORDER BY att DESC""")
    if not rows:
        bot.reply_to(msg, "No ratings yet.")
        return
    lines = [
        "╔══ 📐 *TEAM RATINGS* ══════════╗",
        "║ `Team  ATT    DEF    1H-ATT  G`",
        "╠══════════════════════════════╣",
    ]
    for r in rows:
        bar = conf_bar(min(r["att"]/2.8, 1.0), 4)
        lines.append(
            f"║ `{r['team']:<4}` {bar} "
            f"`{r['att']:.3f}` `{r['dfd']:.3f}` `{r['h_att']:.3f}` `({r['games_seen']})`"
        )
    lines.append("╚══════════════════════════════╝")
    bot.reply_to(msg, "\n".join(lines), parse_mode="Markdown")

@bot.message_handler(commands=["history"])
def h_history(msg):
    rows = db.all("""SELECT home,away,hg,ag,c_outcome,c_ou,c_btts,c_h1ou,
                            c_dw_home,c_dw_away,had_pred
                     FROM results ORDER BY ts DESC LIMIT 20""")
    if not rows:
        bot.reply_to(msg, "No history yet.")
        return
    lines = ["╔══ 📋 *LAST 20 RESULTS* ══════╗"]
    for i, r in enumerate(rows, 1):
        score = f"{r['hg']}-{r['ag']}"
        if r["had_pred"]:
            chk = (f"1X2:{'✅' if r['c_outcome'] else '❌'}"
                   f" OU:{'✅' if r['c_ou'] else '❌'}"
                   f" BT:{'✅' if r['c_btts'] else '❌'}"
                   f" 1H:{'✅' if r['c_h1ou'] else '❌'}"
                   f" DW:{'✅' if r['c_dw_home'] or r['c_dw_away'] else '❌'}")
        else:
            chk = "_(no pred)_"
        lines.append(f"║ {i:>2}. {r['home']} *{score}* {r['away']}  {chk}")
    lines.append("╚══════════════════════════════╝")
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
        bot.reply_to(msg, "❌ Score format must be like `2-1`", parse_mode="Markdown")
        return
    rec = predictor.record(game_id, hg, ag)
    acc = predictor.accuracy()
    text = f"✅ *Recorded* `{game_id}` → `{score}`\n"
    if rec["had"]:
        text += (
            f"1X2:{'✅' if rec['co'] else '❌'} "
            f"OU:{'✅' if rec['cou'] else '❌'} "
            f"BTTS:{'✅' if rec['cbt'] else '❌'} "
            f"1H:{'✅' if rec['ch1'] else '❌'} "
            f"DW:{'✅' if rec['cdh'] or rec['cda'] else '❌'}\n"
        )
    text += f"\n{acc_block(acc)}"
    bot.reply_to(msg, text, parse_mode="Markdown")

@bot.message_handler(commands=["reset"])
def h_reset(msg):
    for t in ("ratings","predictions","results","h2h"):
        db.ex(f"DELETE FROM {t}")
    db.commit()
    seed_ratings()
    bot.reply_to(msg, "🔄 All data cleared. Ratings reset to defaults.")

@bot.message_handler(content_types=["photo"])
def h_photo(msg):
    if user_mode.get(msg.chat.id, MODE_PREDICT) == MODE_RESULT:
        _do_results(msg)
    else:
        _do_predict(msg)

def _do_predict(msg):
    status = bot.reply_to(msg, "⚡ Analysing screenshot…")
    try:
        img     = _dl_image(msg.photo[-1].file_id)
        matches = (_vision(img, PREDICT_PROMPT) or {}).get("matches")
        if not matches:
            bot.edit_message_text(
                "❌ No upcoming matches found.\nIf this is a results screen, use /result first.",
                status.chat.id, status.message_id)
            return
        acc  = predictor.accuracy()
        header = (
            "┌─────────────────────────────┐\n"
            "│  ⚡ *PREDICTIONS*            │\n"
            f"│  {len(matches)} matches  •  "
            f"Overall: {pct(acc['overall']) if acc else 'Building...'}\n"
            "└─────────────────────────────┘\n"
        )
        lines = [header]
        for i, m in enumerate(matches, 1):
            home = m.get("home","???").upper()
            away = m.get("away","???").upper()
            seed = f"{home}_{away}_{int(time.time()//60)}"
            gid  = f"{home}_{away}_{hashlib.md5(seed.encode()).hexdigest()[:8]}"
            pred = predictor.predict(home, away)
            predictor.save(gid, home, away, pred)
            lines.append(pred_card(home, away, pred, gid, i))
        lines += [
            "─────────────────────────────",
            "📊 After matches: /result → send results screenshot",
            "⚠️ _Not financial advice._",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as ex:
        print(f"[PREDICT ERROR] {ex}")
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
        lines = [
            "┌─────────────────────────────┐\n"
            f"│  📊 *RESULTS*  {len(results)} matches     │\n"
            "└─────────────────────────────┘\n"
        ]
        for i, r in enumerate(results, 1):
            home = r["home"].upper()
            away = r["away"].upper()
            hg   = int(r["home_goals"])
            ag   = int(r["away_goals"])
            pred_row = db.one("""SELECT game_id FROM predictions
                                 WHERE home=? AND away=?
                                 ORDER BY ts DESC LIMIT 1""", (home, away))
            gid = (pred_row["game_id"] if pred_row
                   else f"{home}_{away}_{hashlib.md5(f'{home}{away}{int(time.time())}'.encode()).hexdigest()[:8]}")
            rec = predictor.record(gid, hg, ag, home, away)
            lines.append(result_card(r, rec, i))
        acc = predictor.accuracy()
        lines += [
            "\n─────────────────────────────",
            "📐 Ratings updated with fast-learn engine.",
            acc_block(acc),
            "\n📸 Next round: /predict",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as ex:
        print(f"[RESULT ERROR] {ex}")
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
    return "Virtual Predictor v6 ✅", 200

def setup_webhook():
    bot.remove_webhook()
    time.sleep(0.5)
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[WEBHOOK] {RENDER_URL}/{TELEGRAM_TOKEN}")

if __name__ == "__main__":
    setup_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
