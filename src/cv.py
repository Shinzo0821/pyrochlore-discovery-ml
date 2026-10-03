"""nested cross-validation。

外側 RepeatedStratifiedKFold(n_splits × n_repeats) の各 fold で
内側 Optuna（さらに内側 CV）を回してハイパーパラメータを決め、
外側テスト fold を予測して out-of-fold 確率を集める。

内側 Optuna は外側の学習 fold だけを見るので、外側テスト fold への
リークは起きない。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from model import build_model, merge_best_params
from optimizer import parse_inner_cv, run_optimization


@dataclass
class CVResult:
    """nested CV の生成物。"""

    oof: pd.DataFrame  # row_id, repeat, fold, y_true, proba, pred（学習行 × n_repeats 行）
    best_params: pd.DataFrame  # fold ごとの best_params（n_splits × n_repeats 行）
    importances: pd.DataFrame  # fold ごとの feature_importances_


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    param_space,
    n_splits: int = 5,
    n_repeats: int = 5,
    cv_random_state: int = 227,
    optuna_random_state: int = 227,
    n_trials: int = 300,
    inner_cv: Mapping[str, int] = None,
    scoring: str = "average_precision",
    threshold: float = 0.5,
    n_jobs: int = -1,
    n_jobs_model: int = 2,
    n_jobs_cv: int = 5,
) -> CVResult:
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_random_state
    )
    n_folds = n_splits * n_repeats
    inner_splits, inner_repeats = parse_inner_cv(inner_cv)
    print(
        f"[INFO] Nested CV: {n_splits}-fold x {n_repeats} repeats = {n_folds} folds, "
        f"{n_trials} trials each (scoring={scoring}, "
        f"inner_cv={inner_splits}x{inner_repeats}, threshold={threshold})"
    )

    oof_rows: List[pd.DataFrame] = []
    params_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    for i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat, fold = divmod(i, n_splits)
        print(f"\n[INFO] --- outer fold {i + 1}/{n_folds} (repeat {repeat}, fold {fold}) ---")

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        if not X_tr.index.intersection(X_te.index).empty:
            raise ValueError("外側 fold の学習行とテスト行が重複しています（リーク）")

        # 内側 Optuna は外側の学習 fold のみを見る（外側テスト fold へのリークなし）
        study = run_optimization(
            model_name=model_name,
            X=X_tr,
            y=y_tr,
            param_space=param_space,
            n_trials=n_trials,
            random_state=optuna_random_state,
            scoring=scoring,
            inner_cv=inner_cv,
            n_jobs_model=n_jobs_model,
            n_jobs_cv=n_jobs_cv,
            verbose=False,
        )

        params = merge_best_params(param_space, study.best_params)
        model = build_model(
            params,
            model_name=model_name,
            random_state=optuna_random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        oof_rows.append(
            pd.DataFrame(
                {
                    "row_id": X_te.index,
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": y_te.to_numpy(),
                    "proba": proba,
                    # model.predict は閾値 0.5 固定なので使わない。
                    # config の閾値と predictor.py の pred_* を同じ基準に揃える
                    "pred": (proba >= threshold).astype(int),
                }
            )
        )

        params_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                # inner_best_score の意味はモードによって違うので、指標名を列に残す
                "scoring": scoring,
                "inner_best_score": study.best_value,
                **params,
            }
        )
        importance_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                **dict(zip(X.columns, model.feature_importances_)),
            }
        )

        print(f"[INFO] fold inner best {scoring}={study.best_value:.4f}, params={study.best_params}")

    return CVResult(
        oof=pd.concat(oof_rows, ignore_index=True),
        best_params=pd.DataFrame(params_rows),
        importances=pd.DataFrame(importance_rows),
    )


def aggregate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """row_id ごとに repeats をまたいで OOF 確率を集約する。

    Returns:
        index=row_id, columns=[y_true, proba_mean, proba_std, n_repeats]
    """
    g = oof.groupby("row_id")
    out = pd.DataFrame(
        {
            "y_true": g["y_true"].first(),
            "proba_mean": g["proba"].mean(),
            "proba_std": g["proba"].std(ddof=1),
            "n_repeats": g["proba"].size(),
        }
    )
    # 各行はどの repeat でもちょうど 1 回テスト fold に入るはず
    if out["n_repeats"].nunique() != 1:
        raise ValueError(
            f"OOF の repeat 数が行によって異なります: {sorted(out['n_repeats'].unique())}"
        )
    return out
