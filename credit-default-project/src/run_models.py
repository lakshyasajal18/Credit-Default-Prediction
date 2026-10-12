"""
Credit card default prediction: logistic regression baseline, lasso, gradient boosting.

Dataset: UCI "Default of Credit Card Clients" (30,000 Taiwanese credit card clients,
April to September 2005, target = default payment next month).

Usage:
    python run_models.py --data path/to/default_of_credit_card_clients.xls --out ../results

Everything reported in the README comes from results/metrics.json, which this script writes.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "default"
# Excluded on purpose: protected attributes in credit decisioning (see README).
DROP_PROTECTED = ["SEX", "MARRIAGE"]


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p, header=1)  # UCI file has a title row above the header
    else:
        df = pd.read_csv(p)
        if "ID" not in df.columns and df.shape[1] > 20:  # csv copies sometimes keep the title row
            df = pd.read_csv(p, header=1)
    df = df.rename(columns={"default payment next month": TARGET,
                            "default.payment.next.month": TARGET,  # Kaggle copy
                            "PAY_0": "PAY_1"})
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Undocumented category codes in the raw data are merged into "other".
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df = df.drop(columns=[c for c in DROP_PROTECTED if c in df.columns])
    df = pd.get_dummies(df, columns=["EDUCATION"], drop_first=True, dtype=float)
    # Simple engineered features from repayment history and utilization.
    pay_cols = [f"PAY_{i}" for i in (1, 2, 3, 4, 5, 6)]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    paid_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["max_delay"] = df[pay_cols].max(axis=1)
    df["months_delayed"] = (df[pay_cols] > 0).sum(axis=1)
    df["avg_utilization"] = (df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]).clip(-1, 5)
    df["avg_payment_ratio"] = (df[paid_cols].sum(axis=1) /
                               df[bill_cols].clip(lower=1).sum(axis=1)).clip(0, 5)
    return df


def best_threshold(y, p, cost_fn, cost_fp):
    """Pick the probability cutoff that minimizes total cost on the given data."""
    grid = np.linspace(0.02, 0.9, 89)
    costs = []
    for t in grid:
        pred = p >= t
        fn = np.sum((~pred) & (y == 1))
        fp = np.sum(pred & (y == 0))
        costs.append(cost_fn * fn + cost_fp * fp)
    costs = np.array(costs)
    return grid, costs, float(grid[costs.argmin()])


def main(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = clean(load(data))
    y = df.pop(TARGET).astype(int)
    X = df
    results = {"n_rows": int(len(df)), "n_features": int(X.shape[1]),
               "default_rate": float(y.mean()), "features": list(X.columns)}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    results["n_train"], results["n_test"] = int(len(Xtr)), int(len(Xte))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # 1) Statsmodels logistic regression: interpretable baseline with standardized inputs.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=X.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=X.columns, index=Xte.index)
    logit = sm.Logit(ytr, sm.add_constant(Xtr_s)).fit(disp=0, maxiter=200)
    p_logit = logit.predict(sm.add_constant(Xte_s, has_constant="add"))
    coefs = pd.DataFrame({"coef": logit.params, "p_value": logit.pvalues}).drop("const")
    coefs["odds_ratio_per_sd"] = np.exp(coefs["coef"])
    coefs.sort_values("p_value").to_csv(out / "logit_coefficients.csv")

    # 2) Lasso (L1) logistic regression, penalty chosen by 5 fold CV on the training set.
    lasso = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(penalty="l1", solver="saga", Cs=12, cv=cv,
                                                   scoring="roc_auc", max_iter=5000,
                                                   random_state=SEED, n_jobs=-1))])
    lasso.fit(Xtr, ytr)
    p_lasso = lasso.predict_proba(Xte)[:, 1]
    lasso_coef = lasso.named_steps["clf"].coef_.ravel()
    results["lasso_nonzero_features"] = int((lasso_coef != 0).sum())
    results["lasso_chosen_C"] = float(lasso.named_steps["clf"].C_[0])

    # 3) Gradient boosting, small grid searched with 5 fold CV on the training set.
    gb = GridSearchCV(
        HistGradientBoostingClassifier(random_state=SEED, early_stopping=True),
        {"learning_rate": [0.03, 0.1], "max_depth": [3, 5], "max_leaf_nodes": [15, 31],
         "l2_regularization": [0.0, 1.0]},
        scoring="roc_auc", cv=cv, n_jobs=-1)
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]
    results["gb_best_params"] = gb.best_params_
    results["cv_auc_train"] = {"gradient_boosting": float(gb.best_score_),
                               "lasso": float(lasso.named_steps["clf"].scores_[1].mean(axis=0).max())}

    # Held out test evaluation.
    models = {"logistic_statsmodels": np.asarray(p_logit), "lasso_logistic": p_lasso,
              "gradient_boosting": p_gb}
    base_rate = float(ytr.mean())
    results["test"] = {}
    for name, p in models.items():
        results["test"][name] = {
            "roc_auc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
        }
    results["test"]["_baseline_brier_predict_base_rate"] = float(
        brier_score_loss(yte, np.full(len(yte), base_rate)))

    # Bootstrap CI on test AUC (1,000 resamples) so differences between models are not over-read.
    rng = np.random.default_rng(SEED)
    yv = yte.values
    for name, p in models.items():
        aucs = []
        for _ in range(1000):
            idx = rng.integers(0, len(yv), len(yv))
            if yv[idx].min() == yv[idx].max():
                continue
            aucs.append(roc_auc_score(yv[idx], p[idx]))
        results["test"][name]["roc_auc_ci95"] = [float(np.percentile(aucs, 2.5)),
                                                 float(np.percentile(aucs, 97.5))]

    # Threshold analysis. The cost ratio is an ASSUMPTION, not a fact about any lender:
    # a missed defaulter costs `ratio` times as much as a wrongly flagged good customer.
    # Threshold is chosen on training predictions (out of fold) and applied to the test set.
    results["threshold_analysis"] = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio in (2, 5):
        oof = np.zeros(len(ytr))
        for tr_i, va_i in cv.split(Xtr, ytr):
            m = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                               **gb.best_params_)
            m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
            oof[va_i] = m.predict_proba(Xtr.iloc[va_i])[:, 1]
        _, _, t = best_threshold(ytr.values, oof, ratio, 1)
        pred = p_gb >= t
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        cost_model = ratio * fn + fp
        cost_none = ratio * int(yte.sum())     # approve everyone: every defaulter is missed
        cost_all = int((yte == 0).sum())       # decline everyone: every good customer is lost
        results["threshold_analysis"][f"cost_ratio_{ratio}_to_1"] = {
            "threshold": t, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "recall": float(tp / (tp + fn)), "precision": float(tp / max(tp + fp, 1)),
            "cost_model": float(cost_model), "cost_approve_all": float(cost_none),
            "cost_decline_all": float(cost_all)}
        grid, costs, _ = best_threshold(yte.values, p_gb, ratio, 1)
        ax.plot(grid, costs / len(yte), label=f"cost ratio {ratio}:1")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost per customer (units of one false positive)")
    ax.set_title("Threshold vs. cost (test set, gradient boosting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_cost.png", dpi=150)
    plt.close(fig)

    # Plots: ROC and calibration.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in models.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results['test'][name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC on held out test set"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in models.items():
        frac, mean_p = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration on held out test set"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150); plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
