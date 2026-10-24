"""Turns the form inputs into a win probability and the numbers shown on the page."""
import math
import pandas as pd

def r(x, places):
    """Round half up, the same way the browser does, so the page and the server always agree."""
    k = 10 ** places
    return math.floor(x * k + 0.5) / k

def parse_overs(text):
    """Cricket notation: '14.3' means 14 overs and 3 balls. Returns balls bowled."""
    s = str(text).strip()
    whole, _, part = s.partition(".")
    if not whole.isdigit() or (part and (not part.isdigit() or len(part) != 1)):
        raise ValueError("Overs must look like 14 or 14.3 (14 overs and 3 balls).")
    overs, balls = int(whole), int(part or 0)
    if balls > 5:
        raise ValueError("Balls after the decimal point must be 0 to 5. Example: 14.3 means 14 overs and 3 balls.")
    total = overs * 6 + balls
    if total > 120:
        raise ValueError("A chase is at most 20 overs.")
    return total

def _int(value, name, low, high):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if v != int(v) or not low <= v <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}.")
    return int(v)

def situation(d, probability_fn):
    """d: dict with batting_team, bowling_team, venue, score, wickets, overs, target.
    probability_fn(features_dict) -> chance (0 to 1) that the batting team wins a live chase."""
    bat, bowl = d.get("batting_team"), d.get("bowling_team")
    score = _int(d.get("score"), "Current score", 0, 500)
    wickets = _int(d.get("wickets"), "Wickets lost", 0, 10)
    target = _int(d.get("target"), "Target score", 1, 500)
    bowled = parse_overs(d.get("overs"))
    if bat == bowl:
        raise ValueError("Choose two different teams.")
    runs_left, balls_left = target - score, 120 - bowled
    crr = r(score * 6 / bowled, 2) if bowled else 0.0
    rrr = r(runs_left * 6 / balls_left, 2) if balls_left > 0 and runs_left > 0 else 0.0
    out = {"batting_team": bat, "bowling_team": bowl, "runs_remaining": max(runs_left, 0), "balls_remaining": balls_left,
           "wickets_in_hand": 10 - wickets, "current_run_rate": crr, "required_run_rate": rrr}
    if runs_left <= 0:
        out.update(status="won", probability=100.0, message=f"{bat} have already reached the target of {target}.")
    elif wickets >= 10:
        out.update(status="lost", probability=0.0, message=f"{bat} are all out, so the chase is over.")
    elif balls_left <= 0:
        out.update(status="lost", probability=0.0, message=f"The overs are finished and {bat} still need {runs_left} runs, so the chase is over.")
    else:
        feats = {"batting_team": bat, "bowling_team": bowl, "venue": d.get("venue"), "runs_left": runs_left, "balls_left": balls_left,
                 "wickets_left": 10 - wickets, "target": target, "crr": crr, "rrr": min(rrr, 36)}
        p = probability_fn(feats)
        w = 10 - wickets
        out.update(status="live", probability=r(100 * p, 1),
                   message=f"{bat} need {runs_left} runs from {balls_left} legal balls, with {w} wicket{'s' if w != 1 else ''} in hand. "
                           f"The required run rate is {rrr:.2f}.")
    return out

def model_probability(bundle):
    pipe = bundle["pipeline"]
    def fn(f):
        return float(pipe.predict_proba(pd.DataFrame([f]))[0][1])
    return fn
