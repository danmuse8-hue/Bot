import os
import json
import base64
import hashlib
import time
import threading
import sqlite3
import requests
from io import BytesIO
from math import exp, factorial
from typing import Optional

from PIL import Image
import telebot
from flask import Flask, request
from openai import OpenAI

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RENDER_URL         = os.getenv("RENDER_URL")

for _name, _val in [("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
                    ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
                    ("RENDER_URL", RENDER_URL)]:
    if not _val:
        raise RuntimeError(f"Missing env var: {_name}")

ai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
bot        = telebot.TeleBot(TELEGRAM_TOKEN)
app        = Flask(__name__)

user_mode: dict[int, str] = {}
MODE_PREDICT = "predict"
MODE_RESULT  = "result"

ADMIN_CHAT_IDS: list[int] = []


class Database:
    def __init__(self, path: str = "virtual_predictor.db"):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params)

    def commit(self):
        with self._lock:
            self._conn.commit()

    def fetchone(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _migrate(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS games (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id          TEXT UNIQUE,
                    timestamp        INTEGER,
                    home_team        TEXT,
                    away_team        TEXT,
                    actual_outcome   TEXT,
                    actual_home_g    INTEGER,
                    actual_away_g    INTEGER,
                    pred_outcome     TEXT,
                    pred_over_under  TEXT,
                    pred_btts        TEXT,
                    pred_exact       TEXT,
                    correct_outcome  INTEGER DEFAULT 0,
                    correct_goals    INTEGER DEFAULT 0,
                    correct_btts     INTEGER DEFAULT 0,
                    correct_exact    INTEGER DEFAULT 0,
                    source           TEXT DEFAULT 'manual'
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id          TEXT,
                    home_team        TEXT,
                    away_team        TEXT,
                    pred_outcome     TEXT,
                    pred_over_under  TEXT,
                    pred_btts        TEXT,
                    pred_exact       TEXT,
                    xg_home          REAL,
                    xg_away          REAL,
                    timestamp        INTEGER
                );
                CREATE TABLE IF NOT EXISTS h2h (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team        TEXT,
                    away_team        TEXT,
                    home_goals       INTEGER,
                    away_goals       INTEGER,
                    outcome          TEXT,
                    timestamp        INTEGER
                );
                CREATE TABLE IF NOT EXISTS team_ratings (
                    team             TEXT PRIMARY KEY,
                    att              REAL NOT NULL,
                    dfd              REAL NOT NULL,
                    games_seen       INTEGER DEFAULT 0,
                    last_updated     INTEGER
                );
                CREATE TABLE IF NOT EXISTS result_history (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team        TEXT,
                    away_team        TEXT,
                    home_goals       INTEGER,
                    away_goals       INTEGER,
                    outcome          TEXT,
                    over_under       TEXT,
                    matched_pred     INTEGER DEFAULT 0,
                    timestamp        INTEGER,
                    source           TEXT DEFAULT 'screenshot'
                );
                CREATE TABLE IF NOT EXISTS drift_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        INTEGER,
                    event            TEXT,
                    recent_acc       REAL,
                    overall_acc      REAL,
                    action           TEXT
                );
            """)
            self._conn.commit()


db = Database()

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

LEAGUE_AVG_HOME   = 1.45
LEAGUE_AVG_AWAY   = 1.10
HOME_ADVANTAGE    = 1.15
BASE_LR           = 0.04
LR_BOOST          = 3.0
RATING_MIN        = 0.30
RATING_MAX        = 2.50
PATTERN_CHECK_EVERY = 5
DRIFT_CHECK_EVERY   = 5
DRIFT_WARN_DROP     = 0.15
DRIFT_RESET_FLOOR   = 0.35
DRIFT_BOOST_GAMES   = 20
PATTERN_MIN_REPEAT  = 2
PATTERN_MAX_LEN     = 8

_lr_boost_active = False
_lr_boost_games  = 0
_lr_boost_lock   = threading.Lock()
_pattern_outcome: list[str] = []
_pattern_ou:      list[str] = []
_pattern_lock    = threading.Lock()


def _seed_ratings():
    for team, r in DEFAULT_RATINGS.items():
        db.execute("""
            INSERT OR IGNORE INTO team_ratings (team, att, dfd, games_seen, last_updated)
            VALUES (?,?,?,0,?)
        """, (team, r["att"], r["dfd"], int(time.time())))
    db.commit()


_seed_ratings()


def get_rating(team: str) -> dict:
    row = db.fetchone("SELECT att, dfd FROM team_ratings WHERE team=?", (team,))
    return {"att": row["att"], "dfd": row["dfd"]} if row else {"att": 1.0, "dfd": 1.0}


def _current_lr(games_seen: int) -> float:
    base = BASE_LR / (1 + games_seen * 0.01)
    with _lr_boost_lock:
        if _lr_boost_active:
            return base * LR_BOOST
    return base


def _consume_boost():
    global _lr_boost_active, _lr_boost_games
    with _lr_boost_lock:
        if _lr_boost_active:
            _lr_boost_games -= 1
            if _lr_boost_games <= 0:
                _lr_boost_active = False


def update_ratings(home: str, away: str, home_goals: int, away_goals: int):
    hr = get_rating(home)
    ar = get_rating(away)
    xg_h  = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADVANTAGE
    xg_a  = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
    h_err = home_goals - xg_h
    a_err = away_goals - xg_a
    hg_row = db.fetchone("SELECT games_seen FROM team_ratings WHERE team=?", (home,))
    ag_row = db.fetchone("SELECT games_seen FROM team_ratings WHERE team=?", (away,))
    lr_h   = _current_lr(hg_row["games_seen"] if hg_row else 0)
    lr_a   = _current_lr(ag_row["games_seen"] if ag_row else 0)
    clamp  = lambda v: max(RATING_MIN, min(RATING_MAX, v))
    now    = int(time.time())
    db.execute("""
        UPDATE team_ratings SET att=?, dfd=?, games_seen=games_seen+1, last_updated=?
        WHERE team=?
    """, (clamp(hr["att"] + lr_h * h_err), clamp(hr["dfd"] - lr_h * a_err * 0.5), now, home))
    db.execute("""
        UPDATE team_ratings SET att=?, dfd=?, games_seen=games_seen+1, last_updated=?
        WHERE team=?
    """, (clamp(ar["att"] + lr_a * a_err), clamp(ar["dfd"] - lr_a * h_err * 0.5), now, away))
    db.commit()
    _consume_boost()


def soft_reset_ratings():
    for team, defaults in DEFAULT_RATINGS.items():
        row = db.fetchone("SELECT att, dfd FROM team_ratings WHERE team=?", (team,))
        if row:
            db.execute("""
                UPDATE team_ratings SET att=?, dfd=?, last_updated=? WHERE team=?
            """, ((row["att"] + defaults["att"]) / 2,
                  (row["dfd"] + defaults["dfd"]) / 2,
                  int(time.time()), team))
    db.commit()


def _find_repeating(sequence: list, max_len: int, min_repeats: int) -> list:
    n = len(sequence)
    for length in range(2, min(max_len + 1, n // min_repeats + 1)):
        candidate = sequence[:length]
        repeats   = sum(
            1 for start in range(0, n - length + 1, length)
            if sequence[start:start + length] == candidate
        )
        if repeats >= min_repeats:
            return candidate
    return []


def run_pattern_check(notify_chat_id: Optional[int] = None):
    global _pattern_outcome, _pattern_ou
    rows = db.fetchall("""
        SELECT actual_outcome,
               CASE WHEN actual_home_g + actual_away_g > 2 THEN 'OVER' ELSE 'UNDER' END as ou
        FROM games WHERE actual_outcome IS NOT NULL
        ORDER BY timestamp DESC LIMIT 40
    """)
    if len(rows) < 10:
        return
    outcomes    = [r["actual_outcome"] for r in rows]
    ous         = [r["ou"] for r in rows]
    new_pat_o   = _find_repeating(outcomes, PATTERN_MAX_LEN, PATTERN_MIN_REPEAT)
    new_pat_ou  = _find_repeating(ous, PATTERN_MAX_LEN, PATTERN_MIN_REPEAT)
    with _pattern_lock:
        changed = (new_pat_o != _pattern_outcome) or (new_pat_ou != _pattern_ou)
        _pattern_outcome = new_pat_o
        _pattern_ou      = new_pat_ou
    if changed and notify_chat_id:
        msg = "🔍 *Pattern update:*\n"
        msg += f"  Outcomes  : `{'→'.join(new_pat_o)}`\n"   if new_pat_o  else "  Outcomes  : none yet\n"
        msg += f"  Over/Under: `{'→'.join(new_pat_ou)}`\n"  if new_pat_ou else "  Over/Under: none yet\n"
        _notify(notify_chat_id, msg)


def get_pattern_outcome_hint(pos: int) -> Optional[str]:
    with _pattern_lock:
        return _pattern_outcome[pos % len(_pattern_outcome)] if _pattern_outcome else None


def get_pattern_ou_hint(pos: int) -> Optional[str]:
    with _pattern_lock:
        return _pattern_ou[pos % len(_pattern_ou)] if _pattern_ou else None


def run_drift_check(notify_chat_id: Optional[int] = None) -> Optional[str]:
    global _lr_boost_active, _lr_boost_games
    overall_row = db.fetchone("""
        SELECT COUNT(*) as n, AVG(correct_outcome) as acc FROM games
        WHERE actual_outcome IS NOT NULL AND pred_outcome IS NOT NULL
    """)
    if not overall_row or overall_row["n"] < 10:
        return None
    overall_acc = overall_row["acc"]
    recent_row  = db.fetchone("""
        SELECT AVG(correct_outcome) as acc FROM (
            SELECT correct_outcome FROM games
            WHERE actual_outcome IS NOT NULL AND pred_outcome IS NOT NULL
            ORDER BY timestamp DESC LIMIT 10
        )
    """)
    if not recent_row or recent_row["acc"] is None:
        return None
    recent_acc = recent_row["acc"]
    drop       = overall_acc - recent_acc

    if recent_acc < DRIFT_RESET_FLOOR and overall_row["n"] >= 20:
        soft_reset_ratings()
        with _pattern_lock:
            _pattern_outcome.clear()
            _pattern_ou.clear()
        with _lr_boost_lock:
            _lr_boost_active = True
            _lr_boost_games  = DRIFT_BOOST_GAMES
        db.execute("""
            INSERT INTO drift_log (timestamp, event, recent_acc, overall_acc, action)
            VALUES (?,?,?,?,?)
        """, (int(time.time()), "COLLAPSE", recent_acc, overall_acc, "soft_reset+boost"))
        db.commit()
        if notify_chat_id:
            _notify(notify_chat_id,
                "🚨 *RNG SHIFT DETECTED — RESET*\n"
                f"Recent accuracy: {recent_acc*100:.1f}%\n"
                "Ratings soft-reset. Learning rate boosted for next "
                f"{DRIFT_BOOST_GAMES} games. Patterns cleared.")
        return "reset"

    if drop >= DRIFT_WARN_DROP:
        with _lr_boost_lock:
            _lr_boost_active = True
            _lr_boost_games  = DRIFT_BOOST_GAMES
        db.execute("""
            INSERT INTO drift_log (timestamp, event, recent_acc, overall_acc, action)
            VALUES (?,?,?,?,?)
        """, (int(time.time()), "DROP", recent_acc, overall_acc, "lr_boost"))
        db.commit()
        if notify_chat_id:
            _notify(notify_chat_id,
                "⚠️ *Possible RNG shift detected*\n"
                f"Overall: {overall_acc*100:.1f}%  |  Last 10: {recent_acc*100:.1f}%\n"
                f"Drop: {drop*100:.1f}pp\n"
                f"Learning rate boosted for {DRIFT_BOOST_GAMES} games.")
        return "warn"

    return "ok"


def _notify(chat_id: int, text: str):
    try:
        bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception as e:
        print(f"[NOTIFY ERROR] {e}")


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k * exp(-lam)) / factorial(k)


def match_outcome_probs(xg_h: float, xg_a: float, max_g: int = 7):
    p1 = px = p2 = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson_pmf(h, xg_h) * poisson_pmf(a, xg_a)
            if   h > a:  p1 += p
            elif h == a: px += p
            else:        p2 += p
    t = p1 + px + p2
    return p1/t, px/t, p2/t


def over_under_prob(xg_h: float, xg_a: float, line: float = 2.5, max_g: int = 14):
    return sum(
        poisson_pmf(h, xg_h) * poisson_pmf(a, xg_a)
        for h in range(max_g + 1)
        for a in range(max_g + 1)
        if h + a > line
    )


def btts_prob(xg_h: float, xg_a: float):
    return (1 - poisson_pmf(0, xg_h)) * (1 - poisson_pmf(0, xg_a))


def most_likely_score(xg_h: float, xg_a: float, max_g: int = 6):
    best_p, bh, ba = -1.0, 1, 1
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson_pmf(h, xg_h) * poisson_pmf(a, xg_a)
            if p > best_p:
                best_p, bh, ba = p, h, a
    return bh, ba


def confidence_label(prob: float) -> str:
    if prob >= 0.65: return "🔥 HIGH"
    if prob >= 0.50: return "📈 MED"
    return "⚠️ LOW"


class Predictor:
    _pred_counter = 0
    _pred_lock    = threading.Lock()

    def expected_goals(self, home: str, away: str) -> tuple:
        hr   = get_rating(home)
        ar   = get_rating(away)
        xg_h = LEAGUE_AVG_HOME * hr["att"] * ar["dfd"] * HOME_ADVANTAGE
        xg_a = LEAGUE_AVG_AWAY * ar["att"] * hr["dfd"]
        h2h  = self._h2h_avg(home, away)
        if h2h:
            xg_h = 0.75 * xg_h + 0.25 * h2h[0]
            xg_a = 0.75 * xg_a + 0.25 * h2h[1]
        xg_h *= self._form_factor(home)
        xg_a *= self._form_factor(away)
        return round(max(0.1, xg_h), 3), round(max(0.1, xg_a), 3)

    def predict(self, home: str, away: str) -> dict:
        xg_h, xg_a = self.expected_goals(home, away)
        p1, px, p2 = match_outcome_probs(xg_h, xg_a)
        p_over      = over_under_prob(xg_h, xg_a)
        p_btts      = btts_prob(xg_h, xg_a)
        sh, sa      = most_likely_score(xg_h, xg_a)
        with self._pred_lock:
            pos = self._pred_counter
            self._pred_counter += 1
        o_hint  = get_pattern_outcome_hint(pos)
        ou_hint = get_pattern_ou_hint(pos)
        if o_hint == "1":   p1 = p1 * 0.70 + 0.30
        elif o_hint == "X": px = px * 0.70 + 0.30
        elif o_hint == "2": p2 = p2 * 0.70 + 0.30
        total = p1 + px + p2
        p1, px, p2 = p1/total, px/total, p2/total
        if ou_hint == "OVER":  p_over = min(0.99, p_over * 0.70 + 0.30)
        elif ou_hint == "UNDER": p_over = max(0.01, p_over * 0.70)
        outcome = max(zip(["1","X","2"], [p1,px,p2]), key=lambda x: x[1])[0]
        return {
            "outcome":       outcome,
            "outcome_prob":  max(p1, px, p2),
            "over_under":    "OVER 2.5" if p_over >= 0.5 else "UNDER 2.5",
            "ou_prob":       p_over if p_over >= 0.5 else 1 - p_over,
            "btts":          "YES" if p_btts >= 0.5 else "NO",
            "btts_prob":     p_btts if p_btts >= 0.5 else 1 - p_btts,
            "exact":         f"{sh}-{sa}",
            "xg_home":       xg_h,
            "xg_away":       xg_a,
            "probs":         {"1": round(p1,3), "X": round(px,3), "2": round(p2,3)},
            "pattern_used":  bool(o_hint or ou_hint),
        }

    def save_prediction(self, game_id: str, home: str, away: str, pred: dict):
        db.execute("""
            INSERT OR REPLACE INTO predictions
            (game_id, home_team, away_team, pred_outcome, pred_over_under,
             pred_btts, pred_exact, xg_home, xg_away, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (game_id, home, away, pred["outcome"], pred["over_under"],
              pred["btts"], pred["exact"], pred["xg_home"], pred["xg_away"], int(time.time())))
        db.commit()

    def record_result(self, game_id: str, outcome: str, score: str,
                      source: str = "manual") -> Optional[dict]:
        try:
            hg, ag = map(int, score.split("-"))
            if hg < 0 or ag < 0:
                raise ValueError
        except ValueError:
            return None
        actual_ou   = "OVER 2.5"  if hg + ag > 2 else "UNDER 2.5"
        actual_btts = "YES"       if hg > 0 and ag > 0 else "NO"
        pred = db.fetchone(
            "SELECT pred_outcome, pred_over_under, pred_btts, pred_exact, home_team, away_team "
            "FROM predictions WHERE game_id=?", (game_id,))
        co = cg = cb = ce = 0
        home_team = away_team = None
        if pred:
            co        = int(outcome     == pred["pred_outcome"])
            cg        = int(actual_ou   == pred["pred_over_under"])
            cb        = int(actual_btts == pred["pred_btts"])
            ce        = int(score       == pred["pred_exact"])
            home_team = pred["home_team"]
            away_team = pred["away_team"]
        else:
            parts = game_id.split("_")
            if len(parts) >= 2:
                home_team, away_team = parts[0], parts[1]
        db.execute("""
            INSERT OR REPLACE INTO games
            (game_id, timestamp, home_team, away_team,
             actual_outcome, actual_home_g, actual_away_g,
             pred_outcome, pred_over_under, pred_btts, pred_exact,
             correct_outcome, correct_goals, correct_btts, correct_exact, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (game_id, int(time.time()), home_team, away_team,
              outcome, hg, ag,
              pred["pred_outcome"]    if pred else None,
              pred["pred_over_under"] if pred else None,
              pred["pred_btts"]       if pred else None,
              pred["pred_exact"]      if pred else None,
              co, cg, cb, ce, source))
        if home_team and away_team:
            db.execute("""
                INSERT INTO h2h (home_team, away_team, home_goals, away_goals, outcome, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (home_team, away_team, hg, ag, outcome, int(time.time())))
            update_ratings(home_team, away_team, hg, ag)
        db.commit()
        return {"correct_outcome": co, "correct_goals": cg,
                "correct_btts": cb,   "correct_exact": ce,
                "had_prediction": pred is not None}

    def record_results_bulk(self, results: list,
                            notify_chat_id: Optional[int] = None) -> list:
        out = []
        for r in results:
            home    = r["home"]
            away    = r["away"]
            hg      = int(r["home_goals"])
            ag      = int(r["away_goals"])
            score   = f"{hg}-{ag}"
            outcome = "1" if hg > ag else ("X" if hg == ag else "2")
            ou      = "OVER" if hg + ag > 2 else "UNDER"
            pred_row = db.fetchone("""
                SELECT game_id FROM predictions
                WHERE home_team=? AND away_team=?
                ORDER BY timestamp DESC LIMIT 1
            """, (home, away))
            game_id = pred_row["game_id"] if pred_row else \
                f"{home}_{away}_{hashlib.md5(f'{home}{away}{int(time.time())}'.encode()).hexdigest()[:8]}"
            rec = self.record_result(game_id, outcome, score, source="screenshot")
            db.execute("""
                INSERT INTO result_history
                (home_team, away_team, home_goals, away_goals, outcome, over_under,
                 matched_pred, timestamp, source)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (home, away, hg, ag, outcome, ou,
                  1 if (rec and rec["had_prediction"]) else 0,
                  int(time.time()), "screenshot"))
            db.commit()
            out.append({"home": home, "away": away, "score": score,
                        "outcome": outcome, **(rec if rec else {})})
        total = db.fetchone(
            "SELECT COUNT(*) as n FROM games WHERE actual_outcome IS NOT NULL")["n"]
        if total % PATTERN_CHECK_EVERY == 0:
            run_pattern_check(notify_chat_id)
        if total % DRIFT_CHECK_EVERY == 0:
            run_drift_check(notify_chat_id)
        return out

    def get_accuracy(self) -> dict:
        row = db.fetchone("""
            SELECT COUNT(*) as n, AVG(correct_outcome) as oc,
                   AVG(correct_goals) as gc, AVG(correct_btts) as bc,
                   AVG(correct_exact) as ec
            FROM games WHERE actual_outcome IS NOT NULL AND pred_outcome IS NOT NULL
        """)
        if not row or row["n"] == 0:
            return {"n":0,"outcome":None,"goals":None,"btts":None,"exact":None,"overall":None}
        return {
            "n":       row["n"],
            "outcome": round(row["oc"] * 100, 1),
            "goals":   round(row["gc"] * 100, 1),
            "btts":    round(row["bc"] * 100, 1),
            "exact":   round(row["ec"] * 100, 1),
            "overall": round(((row["oc"] + row["gc"] + row["bc"]) / 3) * 100, 1),
        }

    def _h2h_avg(self, home: str, away: str) -> Optional[tuple]:
        rows = db.fetchall("""
            SELECT home_goals, away_goals FROM h2h
            WHERE home_team=? AND away_team=?
            ORDER BY timestamp DESC LIMIT 10
        """, (home, away))
        if len(rows) < 3:
            return None
        weights = [1.0 - i * 0.09 for i in range(len(rows))]
        total_w = sum(weights)
        return (
            sum(r["home_goals"] * w for r, w in zip(rows, weights)) / total_w,
            sum(r["away_goals"] * w for r, w in zip(rows, weights)) / total_w,
        )

    def _form_factor(self, team: str, n: int = 5) -> float:
        rows = db.fetchall("""
            SELECT actual_outcome, home_team FROM games
            WHERE (home_team=? OR away_team=?) AND actual_outcome IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (team, team, n))
        if not rows:
            return 1.0
        pts = sum(
            1.0 if ((r["home_team"] == team and r["actual_outcome"] == "1") or
                    (r["home_team"] != team and r["actual_outcome"] == "2"))
            else 0.5 if r["actual_outcome"] == "X" else 0.0
            for r in rows
        )
        return 0.85 + (pts / len(rows)) * 0.30


predictor = Predictor()

PREDICT_PROMPT = (
    "You are a data extractor for BetPawa virtual football. "
    "From the screenshot extract each UPCOMING match (no scores shown). "
    "Return ONLY valid JSON, no markdown:\n"
    '{"matches": [{"position":1,"home":"MCI","away":"LIV"}]}\n'
    "Use 3-letter codes. Do NOT invent scores."
)

RESULT_PROMPT = (
    "You are a data extractor for BetPawa virtual football. "
    "From the results screenshot extract the FINAL SCORES of completed matches. "
    "Return ONLY valid JSON, no markdown:\n"
    '{"results": [{"home":"MCI","away":"LIV","home_goals":2,"away_goals":1}]}\n'
    "Only include matches with a visible final score. Use 3-letter codes."
)


def _resize(image_bytes: bytes, max_dim: int = 1024) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def download_image(file_id: str) -> bytes:
    info = bot.get_file(file_id)
    return requests.get(
        f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{info.file_path}",
        timeout=15).content


def _call_vision(image_bytes: bytes, system_prompt: str) -> Optional[dict]:
    b64 = base64.b64encode(_resize(image_bytes)).decode()
    try:
        resp = ai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            max_tokens=700,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",      "text": "Extract data. Return JSON only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"[VISION ERROR] {e}")
        return None


def extract_teams(image_bytes: bytes) -> Optional[list]:
    return (_call_vision(image_bytes, PREDICT_PROMPT) or {}).get("matches")


def extract_results(image_bytes: bytes) -> Optional[list]:
    return (_call_vision(image_bytes, RESULT_PROMPT) or {}).get("results")


def fmt_pct(val: Optional[float]) -> str:
    return f"{val:.1f}%" if val is not None else "N/A"


def build_prediction_card(home: str, away: str, pred: dict, game_id: str) -> str:
    o_emoji  = {"1": "🏠", "X": "🤝", "2": "✈️"}.get(pred["outcome"], "🔮")
    pat_note = "  _(+pattern)_" if pred.get("pattern_used") else ""
    return "\n".join([
        f"{o_emoji} *{home} vs {away}*",
        f"┌────────────────────────",
        f"│ 🎯 1X2  : *{pred['outcome']}*  [{confidence_label(pred['outcome_prob'])}]{pat_note}",
        f"│   1:{pred['probs']['1']:.0%}  X:{pred['probs']['X']:.0%}  2:{pred['probs']['2']:.0%}",
        f"│ 📊 O/U  : *{pred['over_under']}*  [{confidence_label(pred['ou_prob'])}]",
        f"│ ⚽ BTTS : *{pred['btts']}*  [{confidence_label(pred['btts_prob'])}]",
        f"│ 🎲 Score: *{pred['exact']}*",
        f"│ 📐 xG   : {pred['xg_home']} – {pred['xg_away']}",
        f"└────────────────────────",
        f"🆔 `{game_id}`",
    ])


def build_result_card(r: dict) -> str:
    had  = r.get("had_prediction", False)
    line = f"⚽ *{r['home']} {r['score']} {r['away']}*"
    if had:
        line += (
            f"\n│ 1X2:{'✅' if r.get('correct_outcome') else '❌'}  "
            f"O/U:{'✅' if r.get('correct_goals') else '❌'}  "
            f"BTTS:{'✅' if r.get('correct_btts') else '❌'}  "
            f"Exact:{'✅' if r.get('correct_exact') else '❌'}"
        )
    else:
        line += "\n│ _(saved for learning — no prior prediction)_"
    return line


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(message, (
        "🎮 *VIRTUAL PREDICTOR BOT v4*\n\n"
        "*Modes:*\n"
        "📸 /predict — upcoming matches screenshot → predictions\n"
        "📊 /result  — results screenshot → bot learns\n"
        "/mode — check current mode\n\n"
        "*Info:*\n"
        "/stats   — accuracy report\n"
        "/ratings — live team ratings\n"
        "/pattern — detected patterns\n"
        "/drift   — RNG drift log\n"
        "/history — last 20 results\n"
        "/pending — unrecorded predictions\n\n"
        "*Manual entry:*\n"
        "/record `game_id outcome score`\n"
        "/reset  — wipe all data\n\n"
        "⚙️ Poisson xG + H2H + form + patterns + drift detection"
    ), parse_mode="Markdown")


@bot.message_handler(commands=["mode"])
def handle_mode(message):
    mode = user_mode.get(message.chat.id, MODE_PREDICT)
    bot.reply_to(message,
        f"Mode: *{'📸 PREDICT' if mode == MODE_PREDICT else '📊 RESULT'}*\n"
        "Use /predict or /result to switch.", parse_mode="Markdown")


@bot.message_handler(commands=["predict"])
def handle_set_predict(message):
    user_mode[message.chat.id] = MODE_PREDICT
    bot.reply_to(message, "📸 *PREDICT mode* — send upcoming matches screenshot.",
                 parse_mode="Markdown")


@bot.message_handler(commands=["result"])
def handle_set_result(message):
    user_mode[message.chat.id] = MODE_RESULT
    bot.reply_to(message, "📊 *RESULT mode* — send results screenshot to teach the bot.",
                 parse_mode="Markdown")


@bot.message_handler(commands=["stats"])
def handle_stats(message):
    acc = predictor.get_accuracy()
    if acc["n"] == 0:
        bot.reply_to(message, "No matched results yet. Submit results screenshots first.")
        return
    with _lr_boost_lock:
        boost_str = f"⚡ Boosted ({_lr_boost_games}g left)" if _lr_boost_active else "Normal"
    bot.reply_to(message, (
        f"📊 *ACCURACY REPORT* ({acc['n']} games)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 1X2 Outcome : {fmt_pct(acc['outcome'])}\n"
        f"📊 O/U 2.5     : {fmt_pct(acc['goals'])}\n"
        f"⚽ BTTS        : {fmt_pct(acc['btts'])}\n"
        f"🎲 Exact Score : {fmt_pct(acc['exact'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Overall: {fmt_pct(acc['overall'])}\n"
        f"🔧 Adapt: {boost_str}"
    ), parse_mode="Markdown")


@bot.message_handler(commands=["pattern"])
def handle_pattern(message):
    with _pattern_lock:
        po  = list(_pattern_outcome)
        pou = list(_pattern_ou)
    lines = ["🔍 *Detected Patterns*\n━━━━━━━━━━━━━━━━━━━━━━━━"]
    lines.append(f"Outcomes  : `{'→'.join(po)}`"  if po  else "Outcomes  : none yet")
    lines.append(f"Over/Under: `{'→'.join(pou)}`" if pou else "Over/Under: none yet")
    lines.append(f"\n_(Checked every {PATTERN_CHECK_EVERY} games)_")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["drift"])
def handle_drift(message):
    rows = db.fetchall(
        "SELECT timestamp, event, recent_acc, overall_acc, action "
        "FROM drift_log ORDER BY id DESC LIMIT 10"
    )
    if not rows:
        bot.reply_to(message, "No drift events recorded yet.")
        return
    lines = ["📉 *Drift Log*\n━━━━━━━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        ts = time.strftime("%m-%d %H:%M", time.localtime(r["timestamp"]))
        lines.append(
            f"`{ts}` *{r['event']}*\n"
            f"  Recent:{r['recent_acc']*100:.1f}%  Overall:{r['overall_acc']*100:.1f}%  → {r['action']}"
        )
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["ratings"])
def handle_ratings(message):
    rows = db.fetchall(
        "SELECT team, att, dfd, games_seen FROM team_ratings ORDER BY att DESC")
    if not rows:
        bot.reply_to(message, "No ratings yet.")
        return
    lines = ["📐 *Live Team Ratings*\n`Team  att    dfd    (g)`\n━━━━━━━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        lines.append(f"`{r['team']}  {r['att']:.3f}  {r['dfd']:.3f}  ({r['games_seen']}g)`")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["history"])
def handle_history(message):
    rows = db.fetchall("""
        SELECT home_team, away_team, actual_home_g, actual_away_g,
               correct_outcome, correct_goals, correct_btts, source
        FROM games WHERE actual_outcome IS NOT NULL
        ORDER BY timestamp DESC LIMIT 20
    """)
    if not rows:
        bot.reply_to(message, "No history yet.")
        return
    lines = ["📋 *Last 20 Results*\n━━━━━━━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        score = f"{r['actual_home_g']}-{r['actual_away_g']}"
        src   = "📸" if r["source"] == "screenshot" else "✏️"
        chk   = (
            f"1X2:{'✅' if r['correct_outcome'] else '❌'} "
            f"O/U:{'✅' if r['correct_goals'] else '❌'} "
            f"BTTS:{'✅' if r['correct_btts'] else '❌'}"
            if r["correct_outcome"] is not None else "_(no pred)_"
        )
        lines.append(f"{src} {r['home_team']} *{score}* {r['away_team']}  {chk}")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["pending"])
def handle_pending(message):
    rows = db.fetchall(
        "SELECT game_id, home_team, away_team, pred_outcome, pred_over_under, pred_btts "
        "FROM predictions ORDER BY id DESC LIMIT 10")
    if not rows:
        bot.reply_to(message, "No pending predictions.")
        return
    lines = ["📋 *Unrecorded Predictions*\n━━━━━━━━━━━━━━━━━━━━━━━━"]
    for r in rows:
        lines.append(
            f"\n⚽ {r['home_team']} vs {r['away_team']}\n"
            f"🎯 {r['pred_outcome']}  📊 {r['pred_over_under']}  ⚽ {r['pred_btts']}\n"
            f"🆔 `{r['game_id']}`\n"
            f"📝 `/record {r['game_id']} outcome score`"
        )
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["record"])
def handle_record(message):
    parts = message.text.split()
    if len(parts) != 4:
        bot.reply_to(message,
            "❌ Usage: `/record game_id outcome score`\n_e.g._ `/record MCI_LIV_abc1 1 2-1`",
            parse_mode="Markdown")
        return
    _, game_id, raw_outcome, score = parts
    outcome = raw_outcome.upper()
    if outcome not in ("1", "X", "2"):
        bot.reply_to(message, "❌ Outcome must be `1`, `X`, or `2`.", parse_mode="Markdown")
        return
    try:
        hg, ag = map(int, score.split("-"))
        if hg < 0 or ag < 0:
            raise ValueError
    except ValueError:
        bot.reply_to(message, "❌ Score must be `2-1` format.", parse_mode="Markdown")
        return
    result = predictor.record_result(game_id, outcome, score, source="manual")
    if not result:
        bot.reply_to(message, "❌ Could not parse score.", parse_mode="Markdown")
        return
    total = db.fetchone(
        "SELECT COUNT(*) as n FROM games WHERE actual_outcome IS NOT NULL")["n"]
    if total % DRIFT_CHECK_EVERY == 0:
        run_drift_check(message.chat.id)
    if total % PATTERN_CHECK_EVERY == 0:
        run_pattern_check(message.chat.id)
    acc  = predictor.get_accuracy()
    text = f"✅ *Recorded* `{game_id}`\nResult: *{outcome}* ({score})\n\n"
    if result["had_prediction"]:
        text += (
            f"🎯 1X2   : {'✅' if result['correct_outcome'] else '❌'}\n"
            f"📊 O/U   : {'✅' if result['correct_goals'] else '❌'}\n"
            f"⚽ BTTS  : {'✅' if result['correct_btts'] else '❌'}\n"
            f"🎲 Exact : {'✅' if result['correct_exact'] else '❌'}\n\n"
        )
    text += f"📐 Ratings updated.\n📈 Accuracy: {fmt_pct(acc['overall'])} ({acc['n']} games)"
    bot.reply_to(message, text, parse_mode="Markdown")


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    for t in ("games","h2h","predictions","result_history","team_ratings","drift_log"):
        db.execute(f"DELETE FROM {t}")
    db.commit()
    _seed_ratings()
    with _pattern_lock:
        _pattern_outcome.clear()
        _pattern_ou.clear()
    global _lr_boost_active, _lr_boost_games
    with _lr_boost_lock:
        _lr_boost_active = False
        _lr_boost_games  = 0
    bot.reply_to(message, "🔄 All data cleared. Ratings, patterns and drift log reset.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    if user_mode.get(message.chat.id, MODE_PREDICT) == MODE_RESULT:
        _handle_result_photo(message)
    else:
        _handle_predict_photo(message)


def _handle_predict_photo(message):
    status = bot.reply_to(message, "📷 Analysing upcoming matches…")
    try:
        img     = download_image(message.photo[-1].file_id)
        matches = extract_teams(img)
        if not matches:
            bot.edit_message_text(
                "❌ No upcoming matches found.\n"
                "If this is a results screenshot, switch with /result first.",
                status.chat.id, status.message_id)
            return
        lines = ["🎮 *VIRTUAL PREDICTIONS*\n━━━━━━━━━━━━━━━━━━━━━━━━\n"]
        for m in matches:
            home = m.get("home", "???").upper()
            away = m.get("away", "???").upper()
            seed = f"{home}_{away}_{int(time.time() // 60)}"
            gid  = f"{home}_{away}_{hashlib.md5(seed.encode()).hexdigest()[:8]}"
            pred = predictor.predict(home, away)
            predictor.save_prediction(gid, home, away, pred)
            lines.append(build_prediction_card(home, away, pred, gid))
            lines.append("")
        acc = predictor.get_accuracy()
        acc_str = f"{fmt_pct(acc['overall'])} ({acc['n']}g)" if acc["n"] > 0 else "Building…"
        with _pattern_lock:
            pat_active = bool(_pattern_outcome or _pattern_ou)
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📈 Accuracy: {acc_str}",
            f"🔍 Pattern: {'Active' if pat_active else 'Learning'}",
            "📊 After matches: /result then send results screenshot.",
            "⚠️ Not financial advice.",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as e:
        bot.edit_message_text(f"⚠️ Error: {str(e)[:120]}", status.chat.id, status.message_id)


def _handle_result_photo(message):
    status = bot.reply_to(message, "📊 Reading results screenshot…")
    try:
        img     = download_image(message.photo[-1].file_id)
        results = extract_results(img)
        if not results:
            bot.edit_message_text(
                "❌ No results found. Make sure final scores are visible.",
                status.chat.id, status.message_id)
            return
        recorded = predictor.record_results_bulk(results, notify_chat_id=message.chat.id)
        acc      = predictor.get_accuracy()
        lines    = [f"📊 *RESULTS LEARNED* ({len(recorded)} matches)\n━━━━━━━━━━━━━━━━━━━━━━━━\n"]
        for r in recorded:
            lines.append(build_result_card(r))
            lines.append("")
        with _lr_boost_lock:
            adapt_str = (f"⚡ Adapting fast ({_lr_boost_games}g boost left)"
                         if _lr_boost_active else "Normal learning")
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            "📐 Team ratings updated.",
            f"🔧 {adapt_str}",
            f"📈 Accuracy: {fmt_pct(acc['overall'])} ({acc['n']} matched games)",
            "\n📸 Ready for next round: /predict",
        ]
        bot.edit_message_text("\n".join(lines), status.chat.id, status.message_id,
                              parse_mode="Markdown")
    except Exception as e:
        bot.edit_message_text(f"⚠️ Error: {str(e)[:120]}", status.chat.id, status.message_id)


@bot.message_handler(func=lambda m: True)
def handle_text(message):
    mode = user_mode.get(message.chat.id, MODE_PREDICT)
    bot.reply_to(message,
        f"Mode: *{'📸 PREDICT' if mode == MODE_PREDICT else '📊 RESULT'}*  |  /help for commands.",
        parse_mode="Markdown")


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data(as_text=True))
    bot.process_new_updates([update])
    return "OK", 200


@app.route("/", methods=["GET"])
def health():
    return "Virtual Predictor Bot v4 — Active ✅", 200


def setup_webhook():
    bot.remove_webhook()
    time.sleep(0.5)
    bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    print(f"[WEBHOOK] → {RENDER_URL}/{TELEGRAM_TOKEN}")


if __name__ == "__main__":
    setup_webhook()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
