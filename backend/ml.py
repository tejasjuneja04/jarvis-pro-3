"""
JARVIS machine-learning engine (scikit-learn). No plotting and no text formatting in here: just the maths.
ml_report.py turns these results into words, charts and files.

    train(ds, target)        -> auto classification / regression, model comparison, importance, leakage checks
    predict(model, values)   -> what-if prediction from a trained model
    cluster(ds, k=None)      -> K-means with automatic k, a profile and a name for every group
    anomalies(ds, top=10)    -> Isolation Forest, with the reason each row looks unusual
    test(ds, a, b)           -> the right significance test for the two columns, in plain terms

Everything works on whatever `analytics.Dataset` hands over (kinds: numeric, categorical, discrete, datetime,
year, id, constant, sparse, text). Designed for Python 3.9+ and scikit-learn 1.0+.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

MAX_ROWS = 20000          # more rows than this are sampled for speed
MAX_CATEGORIES = 40       # a categorical column with more levels is dropped from the features
MAX_CLASSES = 30
SEED = 42

warnings.filterwarnings("ignore")


class MLError(Exception):
    """Something the user can fix (their message is shown as is)."""


# ---------------------------------------------------------------------------
# turning a dataset into model inputs
# ---------------------------------------------------------------------------

def _one_hot():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                                   # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def feature_frame(ds, exclude=()):
    """Return (X, dropped) where X holds numeric columns (float) and categorical columns (object)."""
    df = ds.df
    cols = {}
    dropped = []
    for c in df.columns:
        if c in exclude:
            continue
        kind = ds.kinds.get(c)
        s = df[c]
        if kind in ("id", "constant", "text"):
            dropped.append((c, kind))
            continue
        if kind == "datetime":
            d = pd.to_datetime(s, errors="coerce")
            if d.notna().sum() < max(5, len(d) * 0.5):
                dropped.append((c, "dates I could not read"))
                continue
            cols[f"{c}__year"] = d.dt.year.astype(float)
            cols[f"{c}__month"] = d.dt.month.astype(float)
            cols[f"{c}__weekday"] = d.dt.dayofweek.astype(float)
            continue
        if kind in ("numeric", "discrete", "year", "sparse") and pd.api.types.is_numeric_dtype(s):
            cols[c] = s.astype(float)
            continue
        if s.nunique(dropna=True) <= MAX_CATEGORIES:
            cols[c] = s.astype(object).where(s.notna(), np.nan)
            continue
        dropped.append((c, "too many different values"))
    X = pd.DataFrame(cols, index=df.index)
    return X, dropped


def _split_types(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def _preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num, cat = _split_types(X)
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", _one_hot())]), cat))
    return ColumnTransformer(parts, remainder="drop")


def _task_for(ds, target):
    s = ds.df[target]
    kind = ds.kinds.get(target)
    if kind == "numeric" and pd.api.types.is_numeric_dtype(s):
        return "regression"
    if kind in ("categorical", "discrete", "year") or s.nunique() <= 10 or not pd.api.types.is_numeric_dtype(s):
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _models(task, imbalance):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge

    cw = "balanced" if imbalance else None
    if task == "classification":
        return [
            ("Baseline (guess the common answer)", DummyClassifier(strategy="most_frequent")),
            ("Logistic regression", LogisticRegression(max_iter=1000, class_weight=cw)),
            ("Random forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=cw,
                                                     n_jobs=-1, random_state=SEED)),
            ("Gradient boosting", HistGradientBoostingClassifier(max_iter=150, class_weight=cw, random_state=SEED)
             if _hgb_has_class_weight() else HistGradientBoostingClassifier(max_iter=150, random_state=SEED)),
        ]
    return [
        ("Baseline (always the average)", DummyRegressor(strategy="mean")),
        ("Ridge regression", Ridge(alpha=1.0)),
        ("Random forest", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
        ("Gradient boosting", HistGradientBoostingRegressor(max_iter=150, random_state=SEED)),
    ]


def _hgb_has_class_weight():
    import inspect
    from sklearn.ensemble import HistGradientBoostingClassifier
    return "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters


def _metrics(task, y_true, y_pred, proba=None, classes=None):
    from sklearn import metrics as M
    if task == "regression":
        rmse = float(np.sqrt(M.mean_squared_error(y_true, y_pred)))
        return {"r2": float(M.r2_score(y_true, y_pred)), "mae": float(M.mean_absolute_error(y_true, y_pred)),
                "rmse": rmse}
    out = {"accuracy": float(M.accuracy_score(y_true, y_pred)),
           "f1": float(M.f1_score(y_true, y_pred, average="macro")),
           "balanced_accuracy": float(M.balanced_accuracy_score(y_true, y_pred))}
    if proba is not None and classes is not None and len(classes) == 2:
        try:
            out["auc"] = float(M.roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1]))
        except Exception:
            pass
    return out


def train(ds, target, budget=30.0):
    """Train, compare and explain models that predict `target`. Raises MLError with a friendly message."""
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline

    if target not in ds.df.columns:
        raise MLError(f"I can't find a column called {target}.")
    t0 = time.time()
    notes = []
    task = _task_for(ds, target)

    y_all = ds.df[target]
    keep = y_all.notna()
    if keep.sum() < 30:
        raise MLError(f"Only {int(keep.sum())} rows have a value for {target}. I need at least 30 to learn anything honest.")

    X, dropped = feature_frame(ds, exclude=(target,))
    X, y = X[keep], y_all[keep]

    # features that are the answer in disguise
    leaked = []
    if task == "regression":
        for c in list(X.columns):
            if pd.api.types.is_numeric_dtype(X[c]) and X[c].notna().sum() > 10:
                r = X[c].corr(y.astype(float), method="spearman")
                if r == r and abs(r) > 0.995:
                    leaked.append(c)
    else:
        for c in list(X.columns):
            if X[c].nunique() > 1 and len(set(zip(X[c].fillna("∅").astype(str), y.astype(str)))) == X[c].fillna("∅").nunique() \
                    and X[c].fillna("∅").nunique() <= MAX_CLASSES and y.nunique() > 1 and X[c].fillna("∅").nunique() == y.nunique() \
                    and not pd.api.types.is_numeric_dtype(X[c]):
                leaked.append(c)                         # a text column that maps 1:1 onto the target
    if leaked:
        X = X.drop(columns=leaked)
        notes.append("I left out " + ", ".join(leaked) + " because it is almost a copy of " + target
                     + " (that would be cheating, not predicting).")
    if X.shape[1] == 0:
        raise MLError("There are no usable columns left to learn from.")

    if len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
        notes.append(f"Big dataset: I trained on a random {MAX_ROWS:,} rows.")

    classes = None
    if task == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        if len(counts) < 2:
            raise MLError(f"{target} has only one value, so there is nothing to predict.")
        rare = counts[counts < 6].index
        if len(rare):
            mask = ~y.isin(rare)
            X, y = X[mask], y[mask]
            notes.append(f"Ignored {len(rare)} very rare value(s) of {target} (fewer than 6 rows each).")
            counts = y.value_counts()
        if len(counts) > MAX_CLASSES:
            raise MLError(f"{target} has {len(counts)} different values. That is too many to classify; try a smaller group.")
        if len(counts) < 2:
            raise MLError(f"Not enough variety in {target} to train on.")
        classes = sorted(counts.index.tolist())
        imbalance = counts.min() / counts.max() < 0.5
        positive = counts.index[-1] if len(classes) == 2 else None
    else:
        y = y.astype(float)
        imbalance = False
        positive = None

    stratify = y if task == "classification" else None
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scoring = "f1_macro" if task == "classification" else "r2"
    folds = 3 if len(Xtr) > 60 else 2
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED) if task == "classification" \
        else KFold(folds, shuffle=True, random_state=SEED)

    board = []
    fitted = {}
    for name, est in _models(task, imbalance):
        if board and time.time() - t0 > budget:
            notes.append("I stopped trying more models to stay within the time budget.")
            break
        pipe = Pipeline([("prep", _preprocessor(Xtr)), ("model", est)])
        try:
            sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
            pipe.fit(Xtr, ytr)
        except Exception as e:                                      # one bad model must not kill the run
            notes.append(f"{name} failed ({type(e).__name__}) and was skipped.")
            continue
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte) if (task == "classification" and hasattr(pipe, "predict_proba")) else None
        m = _metrics(task, yte, pred, proba, classes)
        board.append({"model": name, "cv": float(np.mean(sc)), "cv_std": float(np.std(sc)), "metrics": m})
        fitted[name] = pipe

    real = [b for b in board if not b["model"].startswith("Baseline")]
    if not real:
        raise MLError("None of the models could be trained on this data.")
    best = max(real, key=lambda b: b["cv"])
    base = next((b for b in board if b["model"].startswith("Baseline")), None)
    pipe = fitted[best["model"]]
    pred = pipe.predict(Xte)

    # which columns matter
    try:
        imp = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=SEED, scoring=scoring, n_jobs=1)
        raw = {c: max(float(v), 0.0) for c, v in zip(Xte.columns, imp.importances_mean)}
    except Exception:
        raw = {}
    total = sum(raw.values()) or 1.0
    importance = sorted(((_origin(c), v / total) for c, v in raw.items()), key=lambda t: -t[1])
    merged = {}
    for c, v in importance:                                         # month/year/weekday of one date -> one column
        merged[c] = merged.get(c, 0.0) + v
    importance = sorted(merged.items(), key=lambda t: -t[1])

    quality = best["metrics"].get("r2" if task == "regression" else "f1", 0.0)
    if quality > 0.985 and importance and importance[0][1] > 0.8:
        notes.append(f"Suspiciously perfect: {importance[0][0]} alone explains almost everything. Check that it is "
                     f"not something you only know after {target} happens.")

    # final model on all the data, for predictions
    final = Pipeline([("prep", _preprocessor(X)), ("model", clone(pipe.named_steps["model"]))])
    final.fit(X, y)

    result = {
        "target": target, "task": task, "rows": int(len(X)), "features": list(X.columns),
        "dropped": dropped, "notes": notes, "leaderboard": sorted(board, key=lambda b: -b["cv"]),
        "best": best, "baseline": base, "importance": importance, "classes": classes, "positive": positive,
        "metrics": best["metrics"], "model": final, "X": X, "y": y,
        "test": {"y": yte.to_numpy(), "pred": np.asarray(pred), "X": Xte},
        "seconds": round(time.time() - t0, 1),
    }
    if task == "classification":
        from sklearn.metrics import confusion_matrix
        result["confusion"] = confusion_matrix(yte, pred, labels=classes)
    result["effects"] = [e for e in (effect(result, c) for c, _ in importance[:4]) if e]
    return result


def _origin(col):
    return col.split("__")[0] if "__" in str(col) else col


# ---------------------------------------------------------------------------
# explaining the model
# ---------------------------------------------------------------------------

def _score(result, X):
    """A single number per row: the predicted value, or the chance of the event for yes/no targets."""
    model = result["model"]
    if result["task"] == "regression":
        return model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if result["positive"] is not None:
        return proba[:, classes.index(result["positive"])]
    return proba.max(axis=1)


def effect(result, col):
    """What happens to the prediction when `col` is moved from low to high (or from one group to another)."""
    X = result["X"]
    cols = [c for c in X.columns if _origin(c) == col]
    if not cols:
        return None
    sample = X.sample(min(len(X), 800), random_state=SEED)
    c = cols[0]
    try:
        if pd.api.types.is_numeric_dtype(X[c]) and len(cols) == 1:
            lo, hi = X[c].quantile(0.1), X[c].quantile(0.9)
            if lo == hi:
                return None
            a, b = sample.copy(), sample.copy()
            a[c], b[c] = lo, hi
            d = float(np.mean(_score(result, b)) - np.mean(_score(result, a)))
            return {"column": col, "kind": "numeric", "low": float(lo), "high": float(hi), "delta": d}
        if len(cols) > 1:
            return None
        levels = X[c].value_counts().head(6).index.tolist()
        scores = {}
        for lv in levels:
            s = sample.copy()
            s[c] = lv
            scores[str(lv)] = float(np.mean(_score(result, s)))
        if len(scores) < 2:
            return None
        top = max(scores, key=scores.get)
        bot = min(scores, key=scores.get)
        return {"column": col, "kind": "category", "top": top, "bottom": bot,
                "top_value": scores[top], "bottom_value": scores[bot], "delta": scores[top] - scores[bot]}
    except Exception:
        return None


def baseline_row(result):
    X = result["X"]
    row = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            row[c] = float(X[c].median())
        else:
            m = X[c].mode()
            row[c] = m.iloc[0] if len(m) else np.nan
    return row


def predict(result, values):
    """`values` maps column name -> value. Everything not given is set to a typical value."""
    X = result["X"]
    row = baseline_row(result)
    used, ignored = {}, []
    for k, v in values.items():
        if k in row:
            if pd.api.types.is_numeric_dtype(X[k]):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    ignored.append(k)
                    continue
            row[k] = v
            used[k] = v
        else:
            ignored.append(k)
    frame = pd.DataFrame([row], columns=X.columns)
    base = pd.DataFrame([baseline_row(result)], columns=X.columns)
    model = result["model"]
    out = {"used": used, "ignored": ignored, "task": result["task"]}
    if result["task"] == "regression":
        out["value"] = float(model.predict(frame)[0])
        out["baseline"] = float(model.predict(base)[0])
        resid = result["test"]["y"] - result["test"]["pred"]
        out["margin"] = float(np.percentile(np.abs(resid), 80))
    else:
        proba = model.predict_proba(frame)[0]
        classes = [str(c) for c in model.named_steps["model"].classes_]
        order = np.argsort(-proba)
        out["label"] = classes[order[0]]
        out["probabilities"] = [(classes[i], float(proba[i])) for i in order[:4]]
        bp = model.predict_proba(base)[0]
        if result["positive"] is not None:
            i = classes.index(str(result["positive"]))
            out["event"] = str(result["positive"])
            out["event_prob"] = float(proba[i])
            out["baseline_prob"] = float(bp[i])
    return out


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def _cluster_matrix(ds, exclude=()):
    X, _ = feature_frame(ds, exclude=exclude)
    X = X[[c for c in X.columns if "__" not in str(c)]]          # date parts are calendar noise, not behaviour
    num, cat = _split_types(X)
    cat = [c for c in cat if X[c].nunique() <= 8]
    X = X[num + cat]
    num = [c for c in num if X[c].nunique() > 1]
    if len(num) + len(cat) < 2:
        raise MLError("I need at least two usable columns to look for groups.")
    return X[num + cat], num, cat


def cluster(ds, k=None, max_k=8):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X, num, cat = _cluster_matrix(ds)
    if len(X) < 20:
        raise MLError("There are too few rows to find meaningful groups.")
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=SEED)
    prep = _preprocessor(X)
    M = prep.fit_transform(X)
    M = np.asarray(M, dtype=float)

    scores = {}
    if k is None:
        sub = M if len(M) <= 3000 else M[np.random.RandomState(SEED).choice(len(M), 3000, replace=False)]
        for kk in range(2, min(max_k, len(sub) - 1) + 1):
            labels = KMeans(kk, n_init=5, random_state=SEED).fit_predict(sub)
            if len(set(labels)) > 1:
                scores[kk] = float(silhouette_score(sub, labels))
        k = max(scores, key=scores.get) if scores else 3
    k = int(max(2, min(k, 12)))
    km = KMeans(k, n_init=10, random_state=SEED).fit(M)
    labels = km.labels_
    sil = scores.get(k)
    if sil is None:
        try:
            sub = np.random.RandomState(SEED).choice(len(M), min(3000, len(M)), replace=False)
            sil = float(silhouette_score(M[sub], labels[sub]))
        except Exception:
            sil = 0.0

    xy = PCA(n_components=2, random_state=SEED).fit_transform(M)
    groups = []
    overall_mean = X[num].mean() if num else pd.Series(dtype=float)
    overall_std = X[num].std().replace(0, 1) if num else pd.Series(dtype=float)
    for g in range(k):
        mask = labels == g
        part = X[mask]
        z = ((part[num].mean() - overall_mean) / overall_std) if num else pd.Series(dtype=float)
        traits = []
        for c in z.abs().sort_values(ascending=False).index[:3]:
            if abs(z[c]) >= 0.35:
                traits.append((c, float(part[c].mean()), float(z[c])))
        cats = []
        for c in cat:
            vc = part[c].value_counts(normalize=True)
            if len(vc) and vc.iloc[0] >= 0.6:
                base = X[c].value_counts(normalize=True).get(vc.index[0], 0)
                if vc.iloc[0] - base > 0.15:
                    cats.append((c, str(vc.index[0]), float(vc.iloc[0])))
        groups.append({"id": g, "size": int(mask.sum()), "share": float(mask.mean()), "traits": traits,
                       "categories": cats, "name": _name_group(traits, cats, g)})
    groups.sort(key=lambda d: -d["size"])
    return {"k": k, "silhouette": sil, "groups": groups, "labels": labels, "index": X.index, "xy": xy,
            "tried": scores, "features": num + cat, "rows": len(X)}


def _name_group(traits, cats, g):
    bits = []
    for c, _, z in traits[:2]:
        bits.append(("High " if z > 0 else "Low ") + str(c).replace("_", " "))
    if not bits and cats:
        bits.append(f"{cats[0][0]} = {cats[0][1]}")
    return " · ".join(bits) if bits else f"Group {g + 1}"


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------

def anomalies(ds, top=10, contamination=0.03):
    from sklearn.ensemble import IsolationForest

    X, num, cat = _cluster_matrix(ds)
    if len(X) < 30:
        raise MLError("I need at least 30 rows to say what is unusual.")
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=SEED)
    M = np.asarray(_preprocessor(X).fit_transform(X), dtype=float)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1).fit(M)
    score = -forest.score_samples(M)
    flagged = forest.predict(M) == -1
    order = np.argsort(-score)[:top]
    mean, std = (X[num].mean(), X[num].std().replace(0, 1)) if num else (None, None)
    rows = []
    for i in order:
        idx = X.index[i]
        why = []
        if num:
            z = (X.iloc[i][num] - mean) / std
            for c in z.abs().sort_values(ascending=False).index[:2]:
                if z[c] == z[c] and abs(z[c]) >= 1.5:
                    why.append(f"{c} = {_fmt(X.iloc[i][c])} ({'+' if z[c] > 0 else '-'}{abs(z[c]):.1f}σ)")
        for c in cat:
            share = (X[c] == X.iloc[i][c]).mean()
            if share < 0.03:
                why.append(f"{c} = {X.iloc[i][c]} (only {share:.1%} of rows)")
        rows.append({"index": idx, "score": float(score[i]), "why": why[:3] or ["unusual combination of values"]})
    return {"rows": rows, "flagged": int(flagged.sum()), "total": int(len(X)), "score": score,
            "index": X.index, "flag": flagged, "features": num + cat}


def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v == int(v) and 1900 <= v <= 2100:
        return str(int(v))
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# hypothesis tests
# ---------------------------------------------------------------------------

def test(ds, a, b):
    """Choose and run the right test for columns a and b. Returns plain facts plus a verdict sentence."""
    from scipy import stats

    df = ds.df[[a, b]].dropna()
    if len(df) < 8:
        raise MLError("There are too few complete rows to test that.")
    na = pd.api.types.is_numeric_dtype(df[a]) and ds.kinds.get(a) in ("numeric", "discrete", "year", "sparse") \
        and df[a].nunique() > 6
    nb = pd.api.types.is_numeric_dtype(df[b]) and ds.kinds.get(b) in ("numeric", "discrete", "year", "sparse") \
        and df[b].nunique() > 6

    if na and nb:
        r, p = stats.pearsonr(df[a].astype(float), df[b].astype(float))
        rho, p2 = stats.spearmanr(df[a], df[b])
        strength = _strength(abs(r))
        return {"kind": "correlation", "a": a, "b": b, "n": len(df), "stat": float(r), "p": float(p),
                "rho": float(rho), "effect": float(r), "test": "Pearson correlation",
                "verdict": _verdict(p, f"a {strength} {'positive' if r > 0 else 'negative'} relationship (r = {r:.2f})",
                                    f"no reliable linear relationship between {a} and {b}")}

    if na != nb:
        num, grp = (a, b) if na else (b, a)
        groups = [g[num].astype(float).values for _, g in df.groupby(grp) if len(g) >= 3]
        names = [str(k) for k, g in df.groupby(grp) if len(g) >= 3]
        if len(groups) < 2:
            raise MLError(f"{grp} doesn't have two groups with enough rows to compare.")
        means = {n: float(np.mean(g)) for n, g in zip(names, groups)}
        if len(groups) == 2:
            t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            sd = np.sqrt((np.var(groups[0], ddof=1) + np.var(groups[1], ddof=1)) / 2) or 1.0
            d = (np.mean(groups[0]) - np.mean(groups[1])) / sd
            hi = max(means, key=means.get)
            return {"kind": "ttest", "a": num, "b": grp, "n": len(df), "stat": float(t), "p": float(p),
                    "effect": float(d), "means": means, "test": "Welch t-test",
                    "verdict": _verdict(p, f"{hi} has a higher {num} (effect size {abs(d):.2f}, {_d_word(abs(d))})",
                                        f"the {num} of {names[0]} and {names[1]} is not reliably different")}
        f, p = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (np.mean(g) - df[num].astype(float).mean()) ** 2 for g in groups)
        ss_total = ((df[num].astype(float) - df[num].astype(float).mean()) ** 2).sum() or 1.0
        eta = float(ss_between / ss_total)
        hi, lo = max(means, key=means.get), min(means, key=means.get)
        return {"kind": "anova", "a": num, "b": grp, "n": len(df), "stat": float(f), "p": float(p),
                "effect": eta, "means": means, "test": "one-way ANOVA",
                "verdict": _verdict(p, f"{num} differs between {grp} groups: highest {hi}, lowest {lo} "
                                       f"({eta:.0%} of the variation is explained by the group)",
                                    f"{num} doesn't differ reliably between {grp} groups")}

    table = pd.crosstab(df[a], df[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise MLError("One of those columns has only one value, so there is nothing to compare.")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))
    return {"kind": "chi2", "a": a, "b": b, "n": int(table.values.sum()), "stat": float(chi2), "p": float(p),
            "effect": v, "test": "chi-square test of independence", "table": table,
            "verdict": _verdict(p, f"{a} and {b} are linked, a {_strength(v)} association (Cramér's V = {v:.2f})",
                                f"{a} and {b} look independent")}


def _verdict(p, yes, no):
    if p < 0.01:
        return f"Strong evidence: {yes}. (p = {_p(p)})"
    if p < 0.05:
        return f"Some evidence: {yes}. (p = {_p(p)})"
    return f"No convincing evidence: {no}. (p = {_p(p)})"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _strength(x):
    return "very weak" if x < 0.1 else "weak" if x < 0.3 else "moderate" if x < 0.5 else "strong" if x < 0.8 else "very strong"


def _d_word(d):
    return "small" if d < 0.5 else "medium" if d < 0.8 else "large"
