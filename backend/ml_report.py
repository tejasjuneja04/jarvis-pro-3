"""
Words, charts and files for the machine-learning engine (ml.py), plus the handler data_assistant calls.

    detect(low, ds, session)            -> "ml_train" | "ml_predict" | "ml_cluster" | "ml_anomaly" | "ml_stats" | None
    handle(intent, session, ds, text)   -> Reply
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import analytics as A
import charts as C
import config
import drivers as D
import ml
import reports as R
from models import Reply, _file_payload

T = config.THEME
GOLD = T["gold"]
PAL = T["palette"]

_TRAIN = re.compile(
    r"\b(train|build|fit|create|make|develop|run)\b.*\b(model|classifier|regressor|regression|machine learning|ml)\b"
    r"|\b(machine learning|automl|auto ml|random forest|gradient boosting|xgboost|classif(y|ier)|"
    r"predictive model|prediction model)\b"
    r"|\bpredict\b\s+(the\s+)?\w+", re.I)
_CLUSTER = re.compile(r"\b(cluster\w*|k-?means|natural groups?|group (my|the|these) (customers|rows|data|records)|"
                      r"unsupervised|find groups|hidden groups|segments? (by|using) (ml|machine learning))\b", re.I)
_ANOMALY = re.compile(r"\b(isolation forest|anomal\w*|unusual (rows|records|entries|transactions|orders)|"
                      r"weird (rows|records|data)|fraud)\b", re.I)
_STATS = re.compile(r"\b(significan\w*|differ\w*|related to|linked to|associated with|p-?value|t-?test|anova|chi-?square|chi2|hypothesis|statistically|"
                    r"is there (a )?(real |significant )?(difference|relationship|link|association)|"
                    r"does \w[\w ]* (differ|depend))\b", re.I)
_WHATIF = re.compile(r"\b(what if|what would|what will|if (the|a|an|i|we)|suppose|for a|for an|given|set)\b", re.I)
_TIME = re.compile(r"\b(next (week|month|quarter|year|day)|future|forecast|projection|over time)\b", re.I)


def detect(low: str, ds: A.Dataset, session) -> str | None:
    cols = ds.columns_in_text(low)
    if _ANOMALY.search(low):
        return "ml_anomaly"
    if _CLUSTER.search(low):
        return "ml_cluster"
    if _STATS.search(low) and len(cols) >= 2:
        return "ml_stats"
    if getattr(session, "ml", None) and _WHATIF.search(low) and re.search(r"\b(predict|chance|probab\w*|likely|"
                                                                          r"what (would|will)|what if|expect)\b", low):
        return "ml_predict"
    if _TRAIN.search(low) and not _TIME.search(low):
        if re.search(r"\bpredict\b", low) and not cols and not re.search(r"model|classif|regress|machine|ml\b", low):
            return None
        return "ml_train"
    return None


def handle(intent: str, session, ds: A.Dataset, text: str) -> Reply:
    try:
        return {"ml_train": _train, "ml_predict": _predict, "ml_cluster": _cluster,
                "ml_anomaly": _anomaly, "ml_stats": _stats}[intent](session, ds, text)
    except ml.MLError as e:
        return Reply(str(e), speech=str(e))
    except ImportError:
        msg = "Machine learning needs scikit-learn. Run: pip install scikit-learn scipy"
        return Reply(msg, speech="I need scikit-learn installed for that. The command is on screen.")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _target(ds, text):
    low = text.lower()
    cols = ds.columns_in_text(low)
    m = re.search(r"\b(?:predict|predicting|classify|forecast|estimate|model for|model of)\s+(?:the\s+)?(.+)$", low)
    if m:
        c = ds.find_column(re.split(r"\b(using|from|based|with|by|given|and|for)\b", m.group(1))[0].strip())
        if c:
            return c
    for pat in (r"\b(?:target|outcome|label)\b\s*(?:is|=|:)?\s*(.+)$",):
        m = re.search(pat, low)
        if m:
            c = ds.find_column(m.group(1).strip())
            if c:
                return c
    targets = D.detect_targets(ds)
    for c in cols:
        if c in targets:
            return c
    if len(targets) == 1:
        return targets[0]
    if cols:
        return cols[0]
    return ds.primary_metric if ds.primary_metric else None


def _train(session, ds, text):
    target = _target(ds, text)
    if not target:
        cands = D.detect_targets(ds) + [c for c in ds.num_cols[:4]]
        hint = f" For example: 'predict {cands[0]}'." if cands else ""
        return Reply("Which column should the model predict?" + hint)
    res = ml.train(ds, target)
    session.ml = {"result": res, "dataset": ds.name, "target": target}
    charts = [c for c in (C._safe(_leaderboard_chart, res), C._safe(_importance_chart, res),
                          C._safe(_fit_chart, res)) if c]
    files = []
    try:
        path = _save_predictions(ds, res)
        f = _file_payload(path, "text/csv")
        files = [f] if f else []
    except Exception:
        files = []
    return Reply(train_report(res), speech=_train_speech(res), charts=charts, files=files,
                 suggestions=_followups(res))


def train_report(res):
    task, target, m = res["task"], res["target"], res["metrics"]
    lines = [R._head(f"MACHINE LEARNING · predicting {target}"),
             f"Task: {'predict a number' if task == 'regression' else 'classify ' + ', '.join(map(str, res['classes'][:4])) + ('…' if len(res['classes']) > 4 else '')}"
             f"   Rows: {res['rows']:,}   Columns used: {len(set(ml._origin(c) for c in res['features']))}",
             "", "Model comparison (3-fold cross-validation on 80% of the rows)"]
    rows = []
    for b in res["leaderboard"]:
        mm = b["metrics"]
        if task == "regression":
            rows.append([b["model"], f"{b['cv']:.3f}", f"{mm['r2']:.3f}", f"{mm['mae']:,.2f}"])
        else:
            rows.append([b["model"], f"{b['cv']:.3f}", f"{mm['accuracy']:.1%}", f"{mm.get('auc', mm['f1']):.3f}"])
    head = ["Model", "CV score", "Test R²", "Test MAE"] if task == "regression" else \
        ["Model", "CV F1", "Test accuracy", "AUC" if "auc" in m else "Test F1"]
    lines.append(R.table(rows, head, "lrrr"))
    lines += ["", f"Winner: {res['best']['model']}"]
    if task == "regression":
        lines.append(f"On rows it never saw, it explains {m['r2']:.0%} of the variation in {target} and is off by "
                     f"{m['mae']:,.2f} on average.")
    else:
        extra = f" and ranks a random positive above a random negative {m['auc']:.0%} of the time" if "auc" in m else ""
        base = res.get("baseline")
        lift = f" (guessing the common answer gets {base['metrics']['accuracy']:.1%})" if base else ""
        lines.append(f"On rows it never saw, it is right {m['accuracy']:.1%} of the time{lift}{extra}.")
    lines += ["", "What drives it most"]
    for c, v in res["importance"][:6]:
        lines.append(f"  {c:<24} {'█' * max(1, int(round(v * 24)))} {v:.0%}")
    if res["effects"]:
        lines += ["", "In plain terms"]
        for e in res["effects"][:4]:
            lines.append("  • " + _effect_sentence(e, res))
        lines.append("  (These are patterns the model found, not proof of cause.)")
    if res["dropped"]:
        lines += ["", "Not used: " + ", ".join(f"{c} ({why})" for c, why in res["dropped"][:6])]
    for n in res["notes"]:
        lines.append("⚠ " + n)
    lines += ["", "Ask me: 'what if tenure is 3 and contract is Month' to use this model."]
    return "\n".join(lines)


def _effect_sentence(e, res):
    unit = "%" if (res["task"] == "classification" and res["positive"] is not None) else ""
    what = f"the chance of {res['positive']}" if unit else f"predicted {res['target']}"
    def amt(d):
        return f"{abs(d) * 100:.0f} points" if unit else f"{abs(d):,.2f}"
    if e["kind"] == "numeric":
        way = "raises" if e["delta"] > 0 else "lowers"
        return f"Going from low to high {e['column']} ({ml._fmt(e['low'])} → {ml._fmt(e['high'])}) {way} {what} by about {amt(e['delta'])}."
    return f"{e['column']} = {e['top']} gives the highest {what}; {e['bottom']} the lowest ({amt(e['delta'])} apart)."


def _train_speech(res):
    m = res["metrics"]
    top = res["importance"][0][0] if res["importance"] else "one column"
    if res["task"] == "regression":
        return (f"Model trained. The best one, {res['best']['model']}, explains {m['r2']:.0%} of the variation in "
                f"{res['target']}. The biggest driver is {top}.")
    return (f"Model trained. The best one, {res['best']['model']}, is right {m['accuracy']:.0%} of the time on unseen "
            f"rows. The biggest driver is {top}.")


def _followups(res):
    out = []
    if res["importance"]:
        out.append(f"What if {res['importance'][0][0]} changes?")
    out += ["Find natural groups", "Find unusual rows"]
    return out[:3]


def _save_predictions(ds, res):
    X = res["X"]
    pred = res["model"].predict(X)
    out = ds.df.loc[X.index].copy()
    out[f"predicted_{res['target']}"] = pred
    if res["task"] == "classification" and res["positive"] is not None:
        classes = list(res["model"].named_steps["model"].classes_)
        out[f"chance_of_{res['positive']}"] = res["model"].predict_proba(X)[:, classes.index(res["positive"])].round(4)
    path = config.OUTPUT_DIR / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', str(ds.name).rsplit('.', 1)[0])}_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# what-if
# ---------------------------------------------------------------------------

def parse_assignments(ds, res, text):
    """Pull 'column value' pairs out of a sentence, for the columns the model actually uses."""
    values = {}
    low = text.lower()
    X = res["X"]
    names = sorted({ml._origin(c) for c in X.columns} | set(X.columns), key=lambda c: -len(str(c)))
    for c in names:
        if c not in X.columns:
            continue
        spaced = re.escape(str(c).lower().replace("_", " "))
        pat = re.compile(rf"(?<![\w]){spaced.replace(chr(92) + ' ', r'[ _]')}(?![\w])\s*(?:=|:|is|to|of|at|was|are)?\s*"
                         r"(-?\d[\d,]*\.?\d*|[a-z][\w\-]*(?: [a-z][\w\-]*)?)")
        m = pat.search(low.replace("_", " ")) or pat.search(low)
        if not m:
            continue
        raw = m.group(1).strip()
        if pd.api.types.is_numeric_dtype(X[c]):
            try:
                values[c] = float(raw.replace(",", ""))
            except ValueError:
                continue
        else:
            levels = {str(v).lower(): v for v in X[c].dropna().unique()}
            hit = levels.get(raw) or levels.get(raw.split()[0])
            if hit is None:
                hit = next((v for k, v in levels.items() if k in low), None)
            if hit is not None:
                values[c] = hit
    for c in X.columns:                                   # a level named on its own: "for a Month contract"
        if c in values or pd.api.types.is_numeric_dtype(X[c]):
            continue
        for lv in X[c].dropna().unique():
            if re.search(rf"(?<![\w]){re.escape(str(lv).lower())}(?![\w])", low) and len(str(lv)) > 2:
                values[c] = lv
                break
    return values


def _predict(session, ds, text):
    state = session.ml
    if not state:
        return Reply("I haven't trained a model yet. Say 'predict <column>' first, for example 'predict Churn'.")
    res = state["result"]
    values = parse_assignments(ds, res, text)
    if not values:
        cols = ", ".join(sorted({ml._origin(c) for c in res["features"]})[:8])
        return Reply(f"Tell me the values to try, for example: 'what if {res['importance'][0][0]} is …'. "
                     f"The model uses: {cols}.")
    out = ml.predict(res, values)
    given = ", ".join(f"{k} = {ml._fmt(v)}" for k, v in out["used"].items())
    lines = [R._head(f"WHAT-IF · {res['target']}"), f"If: {given}", "(everything else typical: the median or most common value)", ""]
    if out["task"] == "regression":
        diff = out["value"] - out["baseline"]
        lines += [f"Predicted {res['target']}: {out['value']:,.2f}  (typical row: {out['baseline']:,.2f}, "
                  f"{'+' if diff >= 0 else '-'}{abs(diff):,.2f})",
                  f"Usual error band: about ±{out['margin']:,.2f}"]
        speech = f"With those values I predict {res['target']} of about {out['value']:,.0f}."
    else:
        for lab, p in out["probabilities"]:
            lines.append(f"  {lab:<20} {'█' * int(round(p * 24))} {p:.0%}")
        if "event_prob" in out:
            lines.append(f"\nChance of {out['event']}: {out['event_prob']:.0%} (typical row: {out['baseline_prob']:.0%})")
            speech = f"With those values the chance of {out['event']} is {out['event_prob']:.0%}."
        else:
            speech = f"I predict {out['label']}, with {out['probabilities'][0][1]:.0%} confidence."
    if out["ignored"]:
        lines.append("Ignored (not in the model): " + ", ".join(out["ignored"]))
    return Reply("\n".join(lines), speech=speech)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def _cluster(session, ds, text):
    m = re.search(r"\b(\d+)\s*(clusters?|groups?|segments?)\b", text.lower())
    res = ml.cluster(ds, k=int(m.group(1)) if m else None)
    quality = "clear" if res["silhouette"] > 0.5 else "moderate" if res["silhouette"] > 0.25 else "loose"
    lines = [R._head("NATURAL GROUPS · K-means"),
             f"I found {res['k']} groups in {res['rows']:,} rows ({quality} separation, silhouette {res['silhouette']:.2f}).",
             f"Columns compared: {', '.join(res['features'][:8])}", ""]
    for g in res["groups"]:
        lines.append(f"▌{g['name']}   {g['size']:,} rows ({g['share']:.0%})")
        for c, mean, z in g["traits"]:
            lines.append(f"    {c}: average {ml._fmt(mean)} ({'above' if z > 0 else 'below'} typical by {abs(z):.1f}σ)")
        for c, v, share in g["categories"][:2]:
            lines.append(f"    mostly {c} = {v} ({share:.0%})")
        lines.append("")
    chart = C._safe(_cluster_chart, res)
    files = []
    try:
        out = ds.df.loc[res["index"]].copy()
        names = {g["id"]: g["name"] for g in res["groups"]}
        out["cluster"] = [names[i] for i in res["labels"]]
        path = config.OUTPUT_DIR / "clusters.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        f = _file_payload(path, "text/csv")
        files = [f] if f else []
    except Exception:
        files = []
    big = res["groups"][0]
    return Reply("\n".join(lines).rstrip(), charts=[chart] if chart else [], files=files,
                 speech=f"I found {res['k']} natural groups. The largest is {big['name']}, with {big['share']:.0%} of the rows.",
                 suggestions=["Find unusual rows", "Show correlations"])


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------

def _anomaly(session, ds, text):
    res = ml.anomalies(ds)
    lines = [R._head("UNUSUAL ROWS · Isolation Forest"),
             f"{res['flagged']:,} of {res['total']:,} rows ({res['flagged'] / res['total']:.1%}) look unusual when all columns are "
             "considered together. The 10 strangest:", ""]
    for i, r in enumerate(res["rows"], 1):
        lines.append(f"{i:>2}. row {r['index']}: " + "; ".join(r["why"]))
    chart = C._safe(_anomaly_chart, res)
    files = []
    try:
        out = ds.df.loc[res["index"]].copy()
        out["anomaly_score"] = res["score"].round(4)
        out["flagged"] = res["flag"]
        path = config.OUTPUT_DIR / "anomalies.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.sort_values("anomaly_score", ascending=False).to_csv(path, index=False)
        f = _file_payload(path, "text/csv")
        files = [f] if f else []
    except Exception:
        files = []
    return Reply("\n".join(lines), charts=[chart] if chart else [], files=files,
                 speech=f"{res['flagged']} rows look unusual. The top ones are listed on screen.")


# ---------------------------------------------------------------------------
# significance tests
# ---------------------------------------------------------------------------

def _stats(session, ds, text):
    cols = ds.columns_in_text(text.lower())
    if len(cols) < 2:
        return Reply("Name two columns to test, for example: 'is Sales different by Region'.")
    r = ml.test(ds, cols[0], cols[1])
    lines = [R._head(f"SIGNIFICANCE · {r['a']} vs {r['b']}"), f"Test: {r['test']}   Rows used: {r['n']:,}", "",
             r["verdict"]]
    if r.get("means"):
        lines += ["", "Averages"] + [f"  {k:<22} {ml._fmt(v)}" for k, v in sorted(r["means"].items(), key=lambda t: -t[1])[:8]]
    lines += ["", "p below 0.05 means a gap this big would be unlikely by chance alone. It says nothing about cause."]
    return Reply("\n".join(lines), speech=r["verdict"].split(" (p")[0])


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

def _leaderboard_chart(res):
    fig, ax = C._figure(7.5, 3.4)
    rows = sorted(res["leaderboard"], key=lambda b: b["cv"])
    ax.barh([b["model"].split(" (")[0] for b in rows], [max(b["cv"], 0) for b in rows],
            color=[GOLD if b is res["best"] else PAL[1] for b in rows])
    for i, b in enumerate(rows):
        ax.text(max(b["cv"], 0) + 0.01, i, f"{b['cv']:.2f}", va="center", color=T["muted"], fontsize=8)
    C._style(ax, "Which model learned best", "R²" if res["task"] == "regression" else "F1 (cross-validated)", grid_axis="x")
    return C._pack(fig, "ml_models", "Model comparison", "Higher is better. Gold is the winner.")


def _importance_chart(res):
    top = res["importance"][:8][::-1]
    fig, ax = C._figure(7.5, 3.6)
    ax.barh([c for c, _ in top], [v for _, v in top], color=GOLD)
    for i, (_, v) in enumerate(top):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", color=T["muted"], fontsize=8)
    C._style(ax, f"What moves {res['target']}", "share of the model's accuracy that depends on the column", grid_axis="x")
    return C._pack(fig, "ml_importance", "Feature importance", "Measured by shuffling each column and watching accuracy drop.")


def _fit_chart(res):
    t = res["test"]
    if res["task"] == "regression":
        fig, ax = C._figure(6.2, 4.6)
        ax.scatter(t["y"], t["pred"], s=14, alpha=0.55, color=PAL[0], edgecolors="none")
        lo, hi = float(min(t["y"].min(), t["pred"].min())), float(max(t["y"].max(), t["pred"].max()))
        ax.plot([lo, hi], [lo, hi], color=GOLD, lw=1.4)
        C._style(ax, "Predicted vs actual (unseen rows)", "actual", "predicted", grid_axis="both")
        return C._pack(fig, "ml_fit", "Prediction accuracy", "Points on the gold line are perfect predictions.")
    cm, classes = res["confusion"], res["classes"]
    fig, ax = C._figure(5.6, 4.6)
    ax.imshow(cm, cmap="YlOrBr")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="#111", fontsize=10, fontweight="bold")
    C._style(ax, "Right and wrong calls (unseen rows)", "predicted", "actual", grid_axis=None)
    return C._pack(fig, "ml_confusion", "Confusion matrix", "The diagonal is what it got right.")


def _cluster_chart(res):
    fig, ax = C._figure(7.2, 4.8)
    names = {g["id"]: g["name"] for g in res["groups"]}
    for g in range(res["k"]):
        mask = res["labels"] == g
        ax.scatter(res["xy"][mask, 0], res["xy"][mask, 1], s=12, alpha=0.6, color=PAL[g % len(PAL)],
                   label=names.get(g, f"Group {g + 1}")[:28], edgecolors="none")
    ax.legend(fontsize=7, frameon=False, labelcolor=T["muted"], loc="best")
    C._style(ax, "Groups on a 2-D map", "main direction of variation", "second direction", grid_axis="both")
    return C._pack(fig, "ml_clusters", "Natural groups", "Rows that sit close together behave alike.")


def _anomaly_chart(res):
    fig, ax = C._figure(7.2, 3.8)
    s = res["score"]
    ax.hist(s, bins=40, color=PAL[1], alpha=0.9)
    ax.axvline(np.min(s[res["flag"]]) if res["flag"].any() else np.max(s), color=GOLD, lw=1.6)
    C._style(ax, "How unusual is each row", "anomaly score (higher = stranger)", "rows")
    return C._pack(fig, "ml_anomaly", "Anomaly scores", "Rows right of the gold line are flagged.")
