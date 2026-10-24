"""Trains the chase win-probability model from ball-by-ball IPL data.
Run from the project folder:  python model/train_model.py

For every ball of a second innings (a chase) we describe the situation (runs left, balls left,
wickets in hand, run rates, teams, venue) and learn whether the batting team went on to win."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Clean names ----------------------------------------------------------------
RENAME = {"Delhi Daredevils": "Delhi Capitals", "Deccan Chargers": "Sunrisers Hyderabad"}
TEAMS = ["Chennai Super Kings", "Delhi Capitals", "Kings XI Punjab", "Kolkata Knight Riders",
         "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Sunrisers Hyderabad"]
VENUES = {"Eden Gardens": "Eden Gardens", "Wankhede Stadium": "Wankhede Stadium",
          "M Chinnaswamy Stadium": "M Chinnaswamy Stadium", "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
          "Feroz Shah Kotla": "Feroz Shah Kotla", "Feroz Shah Kotla Ground": "Feroz Shah Kotla",
          "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
          "Rajiv Gandhi Intl. Cricket Stadium": "Rajiv Gandhi International Stadium",
          "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium", "M. A. Chidambaram Stadium": "MA Chidambaram Stadium",
          "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
          "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association Stadium",
          "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association Stadium",
          "IS Bindra Stadium": "Punjab Cricket Association Stadium"}
VENUE_LIST = sorted(set(VENUES.values())) + ["Other venue"]
NUM = ["runs_left", "balls_left", "wickets_left", "target", "crr", "rrr"]
CAT = ["batting_team", "bowling_team", "venue"]

def chase_table():
    d = pd.read_csv("data/deliveries.csv")
    m = pd.read_csv("data/matches.csv")
    m = m.dropna(subset=["winner", "venue"])
    m = m[m.dl_applied == 0].copy()                                   # rain-adjusted targets are not comparable
    m["winner"] = m.winner.replace(RENAME)
    m["venue"] = m.venue.map(VENUES).fillna("Other venue")
    d = d[d.is_super_over == 0].copy()
    for c in ["batting_team", "bowling_team"]:
        d[c] = d[c].replace(RENAME)

    # Target = first-innings total + 1
    target = d[d.inning == 1].groupby("match_id").total_runs.sum() + 1
    c = d[d.inning == 2].copy()
    c["target"] = c.match_id.map(target)
    c = c.merge(m[["id", "venue", "winner"]], left_on="match_id", right_on="id")
    c = c[c.batting_team.isin(TEAMS) & c.bowling_team.isin(TEAMS)]

    # State after every delivery
    c = c.sort_values(["match_id", "over", "ball"])
    legal = ((c.wide_runs == 0) & (c.noball_runs == 0)).astype(int)
    out = (c.player_dismissed.notna() & (c.dismissal_kind != "retired hurt")).astype(int)
    g = c.groupby("match_id")
    c["score"] = g.total_runs.cumsum()
    c["balls_bowled"] = legal.groupby(c.match_id).cumsum()
    c["wickets"] = out.groupby(c.match_id).cumsum()

    # Also add the very first state of each chase (nothing has happened yet)
    first = c.groupby("match_id").first().reset_index()
    start = first.assign(score=0, balls_bowled=0, wickets=0)
    c = pd.concat([start, c], ignore_index=True)

    c["runs_left"] = c.target - c.score
    c["balls_left"] = 120 - c.balls_bowled
    c["wickets_left"] = 10 - c.wickets
    c = c[(c.runs_left > 0) & (c.balls_left > 0) & (c.wickets_left > 0)]     # the match is still live
    c["crr"] = np.where(c.balls_bowled > 0, c.score * 6 / c.balls_bowled.clip(lower=1), 0.0)
    c["rrr"] = (c.runs_left * 6 / c.balls_left).clip(upper=36)
    c["won"] = (c.winner == c.batting_team).astype(int)
    return c[["match_id"] + CAT + NUM + ["won"]].reset_index(drop=True)

def make_pipeline(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(categories=[TEAMS, TEAMS, VENUE_LIST], handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])

def scores(pipe, X, y):
    p = pipe.predict_proba(X)[:, 1]
    return {"accuracy": round(accuracy_score(y, p >= .5), 4), "log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4)}

if __name__ == "__main__":
    t = chase_table()
    print(f"{t.match_id.nunique()} chases, {len(t)} match situations (one per ball)")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(t, groups=t.match_id))
    tr, te = t.iloc[train_idx], t.iloc[test_idx]          # split by MATCH, so no match is in both sets
    X_tr, y_tr, X_te, y_te = tr[CAT + NUM], tr.won, te[CAT + NUM], te.won

    candidates = {"Logistic Regression": make_pipeline(LogisticRegression(max_iter=3000)),
                  "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=30,
                                                                         n_jobs=-1, random_state=42))}
    metrics = {"matches": int(t.match_id.nunique()), "situations": int(len(t)),
               "test_matches": int(te.match_id.nunique()), "models": {}}
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        metrics["models"][name] = scores(pipe, X_te, y_te)
        print(name, metrics["models"][name])

    # Choose the deployed model: Logistic Regression unless Random Forest has clearly lower log loss.
    lr, rf = metrics["models"]["Logistic Regression"], metrics["models"]["Random Forest"]
    chosen = "Random Forest" if rf["log_loss"] < lr["log_loss"] - 0.02 else "Logistic Regression"
    print("Chosen:", chosen)

    # 5-fold cross-validation by match for the chosen model
    cv = []
    for a, b in GroupKFold(5).split(t, groups=t.match_id):
        p = make_pipeline(LogisticRegression(max_iter=3000) if chosen == "Logistic Regression" else
                          RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=30, n_jobs=-1, random_state=42))
        p.fit(t.iloc[a][CAT + NUM], t.iloc[a].won)
        cv.append(accuracy_score(t.iloc[b].won, p.predict(t.iloc[b][CAT + NUM])))
    metrics["chosen"], metrics["cv_accuracy"] = chosen, round(float(np.mean(cv)), 4)

    # Accuracy by stage of the chase, and a calibration check
    pipe = candidates[chosen]
    p = pipe.predict_proba(X_te)[:, 1]
    over = (120 - te.balls_left) / 6
    metrics["by_stage"] = {k: round(float(accuracy_score(y_te[m], p[m] >= .5)), 4) for k, m in
                           {"overs 0-6": over < 6, "overs 6-15": (over >= 6) & (over < 15), "overs 15-20": over >= 15}.items()}
    bins = pd.cut(p, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    cal = pd.DataFrame({"b": bins, "p": p, "y": y_te.values}).groupby("b", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    metrics["calibration"] = [{"predicted": round(r.predicted, 3), "actual": round(r.actual, 3), "n": int(r.n)} for r in cal.itertuples()]
    print("By stage:", metrics["by_stage"]); print("CV accuracy:", metrics["cv_accuracy"]); print(cal.round(3).to_string())

    # Final model: refit on all matches
    final = make_pipeline(LogisticRegression(max_iter=3000) if chosen == "Logistic Regression" else
                          RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=30, n_jobs=-1, random_state=42))
    final.fit(t[CAT + NUM], t.won)
    joblib.dump({"pipeline": final, "teams": TEAMS, "venues": VENUE_LIST, "chosen": chosen}, "model/win_model.joblib")
    json.dump(metrics, open("model/metrics.json", "w"), indent=2)

    # Export for the browser (offline copy of the same logistic model)
    if chosen == "Logistic Regression":
        sc, lr_m = final.named_steps["pre"].named_transformers_["num"], final.named_steps["model"]
        coef = lr_m.coef_[0]
        params = {"num": NUM, "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
                  "num_coef": coef[:len(NUM)].round(6).tolist(),
                  "bat": dict(zip(TEAMS, coef[len(NUM):len(NUM) + 8].round(6).tolist())),
                  "bowl": dict(zip(TEAMS, coef[len(NUM) + 8:len(NUM) + 16].round(6).tolist())),
                  "venue": dict(zip(VENUE_LIST, coef[len(NUM) + 16:].round(6).tolist())),
                  "intercept": round(float(lr_m.intercept_[0]), 6)}
        js = {"teams": TEAMS, "venues": VENUE_LIST, "params": params,
              "metrics": {"accuracy": metrics["models"][chosen]["accuracy"], "cv": metrics["cv_accuracy"],
                          "log_loss": metrics["models"][chosen]["log_loss"], "matches": metrics["matches"], "situations": metrics["situations"]}}
        open("static/js/model.js", "w").write("// Generated by model/train_model.py: an offline copy of the trained model.\nconst MODEL=" + json.dumps(js, separators=(",", ":")) + ";\n")
    print("Saved model/win_model.joblib, model/metrics.json, static/js/model.js")
