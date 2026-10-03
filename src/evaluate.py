"""OOF 予測からの指標集計と作図。

単発の 80/20 分割ではなく nested CV の out-of-fold 予測を評価対象にする。
報告値は fold ごとの指標の平均±標準偏差（`metrics_summary`）で、
図は repeat ごとに pooled した OOF から描く。
"""

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_TOP_N_PERCENTAGES = (0.05, 0.10, 0.15, 0.20)


def precision_at_top_n(y_true, y_proba, n: int) -> float:
    """予測確率が高い順に n 件取ったときの陽性の割合。"""
    if n <= 0:
        return float("nan")
    order = np.argsort(np.asarray(y_proba))[::-1][:n]
    return float(np.asarray(y_true)[order].mean())


def compute_metrics(
    y_true,
    y_proba,
    y_pred=None,
    top_n_percentages: Sequence[float] = DEFAULT_TOP_N_PERCENTAGES,
    threshold: float = None,
) -> Dict[str, float]:
    """1 つの予測セット（1 fold ぶん、または pooled OOF）の指標を計算する。

    閾値依存の指標（accuracy / precision / recall / f1）は `y_pred` から計算する。
    `y_pred` を省くときは `threshold` を明示すること。以前は 0.5 を暗黙の
    デフォルトにしていたが、config の閾値と食い違っても気づけないのでやめた。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_pred is None:
        if threshold is None:
            raise ValueError("y_pred か threshold のどちらかを指定してください")
        y_pred = (y_proba >= threshold).astype(int)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        # 確率の値そのものの当たり具合。AP-HP のチューニング指標に対応する
        # （Optuna は neg_brier_score を最大化 = この値を最小化している）
        "brier_score": brier_score_loss(y_true, y_proba),
    }

    n_total = len(y_true)
    for p in top_n_percentages:
        n_items = int(n_total * p)
        # 件数は fold ごとに変わるので、キー名には割合だけを入れる（集計時に揃わなくなるため）
        metrics[f"precision_at_top_{int(p * 100)}%"] = precision_at_top_n(
            y_true, y_proba, n_items
        )
        metrics[f"n_at_top_{int(p * 100)}%"] = n_items
    return metrics


def metrics_per_fold(
    oof: pd.DataFrame, top_n_percentages: Sequence[float] = DEFAULT_TOP_N_PERCENTAGES
) -> pd.DataFrame:
    """外側 fold ごとの指標（n_splits × n_repeats 行）。"""
    rows: List[Dict[str, float]] = []
    for (repeat, fold), g in oof.groupby(["repeat", "fold"]):
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n_test": len(g),
                "n_positive": int(g["y_true"].sum()),
                **compute_metrics(g["y_true"], g["proba"], g["pred"], top_n_percentages),
            }
        )
    return pd.DataFrame(rows)


def metrics_per_repeat(
    oof: pd.DataFrame, top_n_percentages: Sequence[float] = DEFAULT_TOP_N_PERCENTAGES
) -> pd.DataFrame:
    """repeat ごとに全 fold を pooled して計算した指標（全学習行が 1 回ずつ入る）。"""
    rows: List[Dict[str, float]] = []
    for repeat, g in oof.groupby("repeat"):
        rows.append(
            {
                "repeat": repeat,
                "n_rows": len(g),
                **compute_metrics(g["y_true"], g["proba"], g["pred"], top_n_percentages),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(
    folds: pd.DataFrame, repeats: pd.DataFrame
) -> pd.DataFrame:
    """報告用のまとめ表: fold 平均±std と repeat pooled 平均±std を 1 枚にする。"""
    skip = {"repeat", "fold", "n_test", "n_positive", "n_rows"}
    metric_cols = [
        c for c in folds.columns if c not in skip and not c.startswith("n_at_top_")
    ]
    rows = []
    for c in metric_cols:
        rows.append(
            {
                "metric": c,
                "fold_mean": folds[c].mean(),
                "fold_std": folds[c].std(ddof=1),
                "fold_min": folds[c].min(),
                "fold_max": folds[c].max(),
                "repeat_pooled_mean": repeats[c].mean() if c in repeats else np.nan,
                "repeat_pooled_std": repeats[c].std(ddof=1) if c in repeats else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("metric")


DEFAULT_SWEEP_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def threshold_sweep(
    oof: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS
) -> pd.DataFrame:
    """閾値を振ったときの precision / recall / f1 / 予測陽性数を repeat 平均±std で返す。

    config の `prediction.threshold` を決め直すための表。repeat ごとに全 fold を
    pooled して（＝全学習行が 1 回ずつ入る）計算し、repeat をまたいで集計する。

    Returns:
        columns=[threshold, precision_mean, precision_std, recall_mean, recall_std,
                 f1_mean, f1_std, n_pred_positive_mean, n_pred_positive_std]
    """
    rows: List[Dict[str, float]] = []
    for t in thresholds:
        per_repeat = []
        for _, g in oof.groupby("repeat"):
            y_true = g["y_true"].to_numpy()
            y_pred = (g["proba"].to_numpy() >= t).astype(int)
            per_repeat.append(
                {
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, zero_division=0),
                    "n_pred_positive": int(y_pred.sum()),
                }
            )
        d = pd.DataFrame(per_repeat)
        row = {"threshold": float(t)}
        for c in d.columns:
            row[f"{c}_mean"] = d[c].mean()
            row[f"{c}_std"] = d[c].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 図 ----------


def plot_confusion_matrix_oof(oof: pd.DataFrame) -> plt.Figure:
    """全 repeat を合算した混同行列（件数は 学習行数 × n_repeats）。"""
    cm = confusion_matrix(oof["y_true"], oof["pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    n_repeats = oof["repeat"].nunique()
    ax.set_title(f"Confusion Matrix (OOF, {n_repeats} repeats pooled)")
    fig.tight_layout()
    return fig


def plot_roc_oof(oof: pd.DataFrame) -> plt.Figure:
    """repeat ごとの ROC を重ね描きし、平均 AUC を凡例に出す。"""
    fig, ax = plt.subplots()
    aucs = []
    for repeat, g in oof.groupby("repeat"):
        fpr, tpr, _ = roc_curve(g["y_true"], g["proba"])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, alpha=0.5, lw=1, label=f"repeat {repeat} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (OOF)  mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_pr_curve_oof(oof: pd.DataFrame, threshold: float = None) -> plt.Figure:
    """repeat ごとの PR 曲線を重ね描きし、採用閾値の動作点を打つ。ROC と対にする。"""
    fig, ax = plt.subplots()
    aps = []
    for repeat, g in oof.groupby("repeat"):
        y_true = g["y_true"].to_numpy()
        y_proba = g["proba"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        aps.append(ap)
        ax.plot(recall, precision, alpha=0.5, lw=1, label=f"repeat {repeat} (AP={ap:.3f})")
        if threshold is not None:
            y_pred = (y_proba >= threshold).astype(int)
            ax.plot(
                recall_score(y_true, y_pred, zero_division=0),
                precision_score(y_true, y_pred, zero_division=0),
                marker="o",
                color="black",
                markersize=4,
            )
    base = float(oof["y_true"].mean())
    ax.axhline(base, linestyle="--", color="gray", label=f"positive rate = {base:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    title = f"Precision-Recall (OOF)  mean AP = {np.mean(aps):.3f} ± {np.std(aps, ddof=1):.3f}"
    if threshold is not None:
        title += f"\n(black dots: threshold = {threshold})"
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize="small")
    fig.tight_layout()
    return fig


def plot_calibration_oof(
    oof: pd.DataFrame, n_bins: int = 10, note: str = None
) -> plt.Figure:
    """信頼性図（reliability diagram）。Brier score の中身を目で見るための図。

    Args:
        note: 図に添える注記。NP-Pyro では負例に UK（未報告）が混ざるので、
            calibration の解釈が AP-HP とは違うことをここに書く
    """
    fig, ax = plt.subplots()
    for repeat, g in oof.groupby("repeat"):
        y_true = g["y_true"].to_numpy()
        y_proba = g["proba"].to_numpy()
        # 空のビンが出ると点が落ちるので strategy="uniform"（既定）のまま素直に描く
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_proba)
        ax.plot(
            prob_pred,
            prob_true,
            marker=".",
            alpha=0.6,
            lw=1,
            label=f"repeat {repeat} (Brier={brier:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    title = f"Calibration (OOF, {n_bins} bins)"
    if note:
        title += f"\n{note}"
    ax.set_title(title, fontsize="medium")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    return fig


def plot_feature_importance_cv(importances: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """fold をまたいだ特徴量重要度の平均±標準偏差。"""
    values = importances.drop(columns=["repeat", "fold"])
    mean = values.mean().sort_values(ascending=False)[:top_n]
    std = values.std(ddof=1)[mean.index]

    fig, ax = plt.subplots()
    ax.bar(range(len(mean)), mean.to_numpy(), yerr=std.to_numpy(), capsize=3)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels([str(i) for i in mean.index], rotation=90)
    ax.set_title(f"Top {len(mean)} Feature Importances (mean ± std over {len(values)} folds)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    return fig


def plot_feature_importance_final(model, feature_names, top_n: int = 20) -> plt.Figure:
    """最終モデル単体の特徴量重要度。"""
    fig, ax = plt.subplots()
    if not hasattr(model, "feature_importances_"):
        ax.text(0.5, 0.5, "No feature importance available", ha="center")
        return fig

    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][: min(top_n, len(importances))]
    ax.bar(range(len(order)), importances[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(feature_names[i]) for i in order], rotation=90)
    ax.set_title(f"Top {len(order)} Feature Importances (final model)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    return fig


def plot_optimization_history(study) -> plt.Figure:
    fig, ax = plt.subplots()
    values = [t.value for t in study.trials if t.value is not None]
    ax.plot(range(1, len(values) + 1), values, marker=".", lw=0.5, label="trial")
    ax.plot(
        range(1, len(values) + 1),
        np.maximum.accumulate(values),
        color="tab:red",
        label="best so far",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.set_title("Optimization History (final model)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig
