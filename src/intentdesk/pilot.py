"""A bounded, local card-support pilot; synthetic cases never prove business value."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import time
from uuid import uuid4

import numpy as np

from intentdesk.common import sha256_file
from intentdesk.reviews import ReviewConflict


WORKFLOW_VERSION = "card_support_v1"
CATEGORIES = {
    "report_lost_card": {"name": "Lost card", "team": "Card security", "definition": "The customer says their card is lost or stolen."},
    "report_fraud": {"name": "Suspected fraud", "team": "Fraud review", "definition": "The customer reports a transaction they did not authorize."},
    "damaged_card": {"name": "Damaged card", "team": "Card replacements", "definition": "The physical card is broken, worn or damaged."},
    "card_declined": {"name": "Declined card", "team": "Payment support", "definition": "A card payment was rejected or declined."},
    "pin_change": {"name": "Change PIN", "team": "Card access", "definition": "The customer wants to change the card PIN."},
    "replacement_card_duration": {"name": "Replacement delivery", "team": "Card delivery", "definition": "The customer asks when a replacement card will arrive."},
    "oos": {"name": "Outside this desk", "team": "General review", "definition": "The request does not fit any of the six card-support categories."},
    "uncertain": {"name": "Needs clarification", "team": "General review", "definition": "There is too little information to choose a category confidently."},
}

# Authored for this demonstration, not customer messages or independently adjudicated labels.
# Four variants per stratum; a participant sees two, once each, in opposite conditions.
CASE_TEXTS = {
    "report_lost_card": [
        "My wallet disappeared on the train and my bank card was inside it.",
        "I cannot find my debit card anywhere and want to report it missing.",
        "Someone took my purse with my payment card. I need to report the theft.",
        "I left my credit card somewhere yesterday and have not recovered it.",
    ],
    "report_fraud": [
        "There is a card purchase on my statement that I never authorized.",
        "A stranger appears to have used my card for an online payment.",
        "I see a card transaction I did not make and want to report suspected fraud.",
        "My statement shows an unfamiliar shop charging my card without my permission.",
    ],
    "damaged_card": [
        "My payment card snapped in half and I need help with the broken card.",
        "The chip on my card is physically damaged and no reader can use it.",
        "My debit card is cracked along the middle after being bent.",
        "The plastic on my bank card is peeling and the card is damaged.",
    ],
    "card_declined": [
        "The shop rejected my card payment even though I have money in my account.",
        "My card was declined when I tried to pay for groceries today.",
        "An online checkout keeps rejecting the payment from my debit card.",
        "The restaurant terminal said my credit card payment was declined.",
    ],
    "pin_change": [
        "I want to replace my current card PIN with a different number.",
        "How can I change the four-digit PIN for my debit card?",
        "I would like to update my bank card PIN to a new one.",
        "Where do I change the PIN number used for card payments?",
    ],
    "replacement_card_duration": [
        "I ordered a replacement card last week. How long does delivery normally take?",
        "When should the replacement for my old bank card arrive in the post?",
        "Can you tell me the delivery time for the replacement debit card I requested?",
        "I am waiting for a replacement payment card and want an estimated arrival date.",
    ],
    "oos": [
        "Please recommend a toothbrush with soft bristles.",
        "I want to change the date of my flight reservation.",
        "Can you reset the password for my email account?",
        "Please tell me tomorrow's weather forecast for my city.",
    ],
}
CASES = [{"id": f"{label}-{i+1}", "text": text, "label": label}
         for label, texts in CASE_TEXTS.items() for i, text in enumerate(texts)]
PROTOCOL = {
    "version": 1, "workflow": WORKFLOW_VERSION, "data": "assistant_authored_demonstration_v1",
    "data_is_real_customer_traffic": False, "independent_label_adjudication": False,
    "cases_per_participant": 14, "conditions": ["manual", "assisted"],
    "allocation": "Two distinct cases per label; opposite conditions. Adjacent reviewer slots counterbalance identical case pairs. Random mixed order; no repeated case within reviewer.",
    "seed": 20261010, "time_limit_seconds": 300,
    "timer": "Server case exposure to server decision receipt; includes network/render time. Inference measured separately, before exposure.",
    "timing_exclusions": "Explicit interruption, clock reversal, or more than 300 seconds. Decisions remain in accuracy counts.",
    "primary_outcomes": ["strict_category_accuracy", "mean_decision_seconds"],
    "other_outcomes": ["uncertain_decisions", "model_suggestion_overrides", "inference_seconds", "timing_exclusions"],
    "uncertainty": "95% percentile bootstrap of reviewer-level assisted-minus-manual differences; 2000 resamples, seed 20261010. Exploratory, not a powered business trial.",
    "eligibility": "Only completed human sessions. One session per reviewer code and binding. Smoke sessions excluded. Timing pairs need at least four timed decisions in each condition.",
    "success_claim": "No business-effect claim from synthetic cases, automated sessions or this small exploratory comparison.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_cases(cases):
    ids, texts = set(), set()
    counts = {label: 0 for label in CASE_TEXTS}
    for case in cases:
        if case["id"] in ids or case["label"] not in counts or not 1 <= len(case["text"]) <= 4000:
            raise ValueError("Invalid or repeated pilot case.")
        normalized = " ".join(case["text"].casefold().split())
        if normalized in texts:
            raise ValueError("Duplicate pilot case text.")
        ids.add(case["id"]); texts.add(normalized); counts[case["label"]] += 1
    if min(counts.values()) < 2:
        raise ValueError("Each pilot category needs at least two distinct cases.")


def assignment(slot):
    rng = random.Random(PROTOCOL["seed"] + slot // 2)
    trials = []
    for label in CASE_TEXTS:
        pair = rng.sample([c for c in CASES if c["label"] == label], 2)
        for i, case in enumerate(pair):
            condition = "assisted" if (i + slot % 2) % 2 else "manual"
            trials.append({**case, "condition": condition})
    random.Random(PROTOCOL["seed"] + slot).shuffle(trials)
    return trials


def suggest(analysis):
    """Project the original top choice onto this desk; never cherry-pick a lower label."""
    status = analysis.get("scope_assessment", {}).get("status")
    candidates = analysis.get("candidates", [])
    if status == "likely_out_of_scope":
        label, reason = "oos", "The scope detector flags this request as likely unsupported."
    elif status != "supported_candidate" or not candidates:
        label, reason = "uncertain", "The model has insufficient support for a category. Review the request."
    elif candidates[0]["intent"] not in CATEGORIES or candidates[0]["intent"] in ("oos", "uncertain"):
        label, reason = "oos", "The model's first category is outside this card-support desk."
    else:
        label, reason = candidates[0]["intent"], "A category looks plausible. Confirm or correct it before preparing a handoff."
    return {"label": label, "name": CATEGORIES[label]["name"], "reason": reason,
            "model_version": analysis.get("policy_version"),
            "scope_version": analysis.get("scope_assessment", {}).get("version"),
            "action_executed": False}


def handoff(label):
    return {"category": CATEGORIES[label]["name"], "team": CATEGORIES[label]["team"],
            "status": "Prepared locally; no external ticket sent", "action_executed": False}


class PilotStore:
    def __init__(self, path: Path, root: Path, router):
        validate_cases(CASES)
        self.path, self.router = path, router
        path.parent.mkdir(parents=True, exist_ok=True)
        files = ["artifacts/policy_latest.json", "artifacts/scope_latest.json", "uv.lock",
                 "src/intentdesk/app.py", "src/intentdesk/web/pilot.html", "src/intentdesk/web/pilot.js",
                 "src/intentdesk/web/style.css", "src/intentdesk/web/pilot.css"]
        binding = {"protocol": PROTOCOL, "workflow": CATEGORIES, "cases_sha256": digest(CASES),
                   "module_sha256": sha256_file(Path(__file__)),
                   "policy_version": router.metadata["run_id"],
                   "files": {f: sha256_file(root/f) for f in files if (root/f).is_file()}}
        self.run_id = digest(binding)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pilot_runs(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, binding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pilot_sessions(
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES pilot_runs(id),
                    reviewer TEXT NOT NULL, kind TEXT NOT NULL, slot INTEGER NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(run_id,reviewer)
                );
                CREATE TABLE IF NOT EXISTS pilot_trials(
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES pilot_sessions(id),
                    position INTEGER NOT NULL, case_id TEXT NOT NULL, text TEXT NOT NULL,
                    expected_label TEXT NOT NULL, condition TEXT NOT NULL,
                    shown_at REAL, suggestion TEXT, inference_seconds REAL NOT NULL DEFAULT 0,
                    label TEXT, decided_at TEXT, elapsed_seconds REAL, timing_exclusion TEXT,
                    UNIQUE(session_id,position)
                );
            """)
            db.execute("INSERT OR IGNORE INTO pilot_runs VALUES(?,?,?)",
                       (self.run_id, self.utc(), canonical(binding)))

    @staticmethod
    def utc():
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def start(self, reviewer, kind="human"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", reviewer) or kind not in ("human", "smoke"):
            raise ValueError("Use a reviewer code of 3–40 letters, numbers, underscores or hyphens.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT id,kind FROM pilot_sessions WHERE run_id=? AND reviewer=?",
                             (self.run_id, reviewer)).fetchone()
            if old:
                if old["kind"] != kind:
                    raise ReviewConflict("This reviewer code already belongs to a different session type.")
                return self.view(old["id"])
            # Smoke runs do not influence the human counterbalancing slots.
            slot = db.execute("SELECT count(*) FROM pilot_sessions WHERE run_id=? AND kind=?",
                              (self.run_id, kind)).fetchone()[0]
            sid = str(uuid4())
            db.execute("INSERT INTO pilot_sessions VALUES(?,?,?,?,?,?)",
                       (sid, self.run_id, reviewer, kind, slot, self.utc(),))
            for i, case in enumerate(assignment(slot)):
                db.execute("INSERT INTO pilot_trials(id,session_id,position,case_id,text,expected_label,condition) VALUES(?,?,?,?,?,?,?)",
                           (str(uuid4()), sid, i, case["id"], case["text"], case["label"], case["condition"]))
        return self.view(sid)

    def rows(self, db, sid):
        session = db.execute("SELECT * FROM pilot_sessions WHERE id=?", (sid,)).fetchone()
        if not session:
            raise KeyError(sid)
        if session["run_id"] != self.run_id:
            raise ReviewConflict("The pilot binding changed. Keep the previous records and start with a new reviewer code.")
        trials = db.execute("SELECT * FROM pilot_trials WHERE session_id=? ORDER BY position", (sid,)).fetchall()
        return session, trials

    def view(self, sid):
        with self.connect() as db:
            session, trials = self.rows(db, sid)
        done = [r for r in trials if r["label"] is not None]
        current = next((r for r in trials if r["label"] is None), None)
        result = {"id": sid, "reviewer": session["reviewer"], "kind": session["kind"],
                  "run_id": self.run_id, "completed": len(done), "total": len(trials),
                  "finished": current is None, "trial": None,
                  "last_handoff": handoff(done[-1]["label"]) if done else None,
                  "automatic_actions": False, "demonstration_data": True}
        if current is not None and current["shown_at"] is not None:
            result["trial"] = {k: current[k] for k in ("id", "text", "condition", "position")}
            result["trial"]["suggestion"] = json.loads(current["suggestion"]) if current["suggestion"] else None
        if current is None:
            result["summary"] = session_metrics(trials)
        return result

    def next(self, sid):
        with self.connect() as db:
            _, trials = self.rows(db, sid)
            current = next((r for r in trials if r["label"] is None), None)
        if current is None or current["shown_at"] is not None:
            return self.view(sid)
        advisory, elapsed = None, 0.
        if current["condition"] == "assisted":
            started = time.perf_counter()
            advisory = suggest(self.router.route(current["text"]))
            elapsed = time.perf_counter() - started
        with self.connect() as db:
            db.execute("UPDATE pilot_trials SET shown_at=?,suggestion=?,inference_seconds=? WHERE id=? AND shown_at IS NULL AND label IS NULL",
                       (time.time(), canonical(advisory) if advisory else None, elapsed, current["id"]))
        return self.view(sid)

    def decide(self, sid, trial_id, label, interrupted=False):
        if label not in CATEGORIES:
            raise ValueError("Choose a card-support category or clarification.")
        with self.connect() as db:
            _, trials = self.rows(db, sid)
            current = next((r for r in trials if r["label"] is None), None)
            if current is None or current["id"] != trial_id or current["shown_at"] is None:
                raise ReviewConflict("This ticket changed or was not opened. Reload the session.")
            seconds = time.time() - current["shown_at"]
            exclusion = ("interrupted" if interrupted else "clock_reversal" if seconds < 0
                         else "over_300_seconds" if seconds > PROTOCOL["time_limit_seconds"] else None)
            changed = db.execute("UPDATE pilot_trials SET label=?,decided_at=?,elapsed_seconds=?,timing_exclusion=? WHERE id=? AND label IS NULL",
                                 (label, self.utc(), max(0., seconds), exclusion, trial_id)).rowcount
            if changed != 1:
                raise ReviewConflict("This ticket was already saved. Reload the session.")
        return self.view(sid)

    def report(self):
        with self.connect() as db:
            sessions = db.execute("SELECT * FROM pilot_sessions WHERE run_id=?", (self.run_id,)).fetchall()
            binding = json.loads(db.execute("SELECT binding FROM pilot_runs WHERE id=?", (self.run_id,)).fetchone()[0])
            all_trials = {s["id"]: db.execute("SELECT * FROM pilot_trials WHERE session_id=? ORDER BY position", (s["id"],)).fetchall()
                          for s in sessions}
        complete = [s for s in sessions if all(r["label"] is not None for r in all_trials[s["id"]])]
        human = [s for s in complete if s["kind"] == "human"]
        smoke = [s for s in sessions if s["kind"] == "smoke"]
        summaries = [session_metrics(all_trials[s["id"]]) for s in human]
        accuracy_diffs = [m["assisted"]["accuracy"]-m["manual"]["accuracy"] for m in summaries]
        time_diffs = [m["assisted"]["mean_decision_seconds"]-m["manual"]["mean_decision_seconds"]
                      for m in summaries if min(m[c]["timed_count"] for c in ("manual", "assisted")) >= 4]
        return {"run_id": self.run_id, "binding": binding, "generated_at": self.utc(),
                "completed_human_sessions": len(human), "incomplete_human_sessions": len(sessions)-len(human)-len(smoke),
                "smoke_sessions_excluded": len(smoke), "conditions": session_metrics([r for s in human for r in all_trials[s["id"]]]),
                "paired_reviewer_accuracy_difference": paired_summary(accuracy_diffs),
                "paired_reviewer_seconds_difference": paired_summary(time_diffs),
                "business_value_established": False,
                "interpretation": "Demonstration only. Synthetic cases and exploratory human comparisons cannot establish customer savings or production reliability."}


def session_metrics(rows):
    result = {}
    for condition in ("manual", "assisted"):
        group = [r for r in rows if r["condition"] == condition and r["label"] is not None]
        timed = [r["elapsed_seconds"] for r in group if r["timing_exclusion"] is None]
        correct = sum(r["label"] == r["expected_label"] for r in group)
        result[condition] = {"count": len(group), "correct": correct,
                             "accuracy": correct/len(group) if group else None,
                             "uncertain": sum(r["label"] == "uncertain" for r in group),
                             "timed_count": len(timed), "timing_exclusions": len(group)-len(timed),
                             "mean_decision_seconds": float(np.mean(timed)) if timed else None,
                             "median_decision_seconds": float(np.median(timed)) if timed else None,
                             "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in group])) if group else None,
                             "suggestion_overrides": sum(r["suggestion"] is not None and json.loads(r["suggestion"])["label"] != r["label"] for r in group)}
    return result


def paired_summary(values):
    if not values:
        return {"reviewers": 0, "mean_assisted_minus_manual": None, "interval_95": None}
    mean = float(np.mean(values))
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(PROTOCOL["seed"])
        # Reviewer resampling preserves dependence among each person's tickets.
        boot = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(2000)]
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {"reviewers": len(values), "mean_assisted_minus_manual": mean, "interval_95": interval,
            "qualification": "Exploratory reviewer bootstrap; few reviewers and shared synthetic tickets limit inference."}
