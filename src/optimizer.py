import optuna
from omegaconf import OmegaConf
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from typing import Any, Dict, Mapping, Tuple
from tqdm import tqdm

from model import build_model, suggest_params


def parse_inner_cv(inner_cv: Mapping[str, int]) -> Tuple[int, int]:
    """`{n_splits, n_repeats}` を (n_splits, n_repeats) に取り出す。

    以前は int 1 つ（分割数）だったので、古い config を静かに受けないよう
    dict でなければエラーにする。
    """
    if OmegaConf.is_config(inner_cv):
        inner_cv = OmegaConf.to_container(inner_cv, resolve=True)
    if not isinstance(inner_cv, Mapping):
        raise TypeError(
            f"inner_cv は {{n_splits, n_repeats}} の dict で指定してください: {inner_cv!r}"
        )
    missing = [k for k in ("n_splits", "n_repeats") if k not in inner_cv]
    if missing:
        raise KeyError(f"inner_cv に必須キーがありません: {missing}")
    return int(inner_cv["n_splits"]), int(inner_cv["n_repeats"])


class TQDMProgressBar:
    def __init__(self, n_trials: int, desc: str = "Optimization Progress"):
        self.pbar = tqdm(total=n_trials, desc=desc)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


def _objective(
    trial: optuna.trial.Trial,
    X,
    y,
    param_space: Dict[str, Any],
    model_name: str,
    scoring: str,
    inner_cv: Mapping[str, int],
    random_state: int,
    n_jobs_model: int,
    n_jobs_cv: int,
) -> float:
    params = suggest_params(trial, param_space)
    model = build_model(
        params, model_name=model_name, random_state=random_state, n_jobs=n_jobs_model
    )
    n_splits, n_repeats = parse_inner_cv(inner_cv)
    # seed 固定で、trial 間の CV 分割を同一にする（比較可能にするため）。
    # n_repeats=1 の分割は StratifiedKFold(shuffle=True, 同 seed) と一致する
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    # 戻り値は n_splits x n_repeats 個すべての fit の平均
    return cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv
    ).mean()


def run_optimization(
    model_name: str,
    X,
    y,
    param_space: Dict[str, Any],
    n_trials: int = 100,
    random_state: int = 227,
    scoring: str = "average_precision",
    inner_cv: Mapping[str, int] = None,
    n_jobs_model: int = 2,
    n_jobs_cv: int = 5,
    show_progress: bool = True,
    verbose: bool = True,
) -> optuna.Study:
    """config の探索空間で Optuna 最適化を回し、study を返す。

    Args:
        model_name: モデル名（例: "random_forest"）
        X, y: 学習データ
        param_space: パラメータ空間（固定値も含む）
        n_trials: 試行回数
        random_state: サンプラーとモデルのシード
        scoring: 評価指標（例: "average_precision", "neg_brier_score"）
        inner_cv: 目的関数内の CV。`{n_splits, n_repeats}`
        n_jobs_model: RF 1 本あたりの並列数
        n_jobs_cv: 目的関数内 CV を何 fold 同時に回すか（n_splits x n_repeats 以下）

    データが小さいので、RF 内部で全コアを使うよりも CV 側を並列にして
    RF 1 本あたりの並列数を絞るほうが速い。
    RF は random_state 固定なら n_jobs を変えても結果は変わらない。
    """
    n_splits, n_repeats = parse_inner_cv(inner_cv)
    if verbose:
        print(
            f"[INFO] Starting optimization: model={model_name}, scoring={scoring}, "
            f"trials={n_trials}, inner_cv={n_splits}x{n_repeats}, "
            f"n_jobs_model={n_jobs_model}, n_jobs_cv={n_jobs_cv}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    callbacks = []
    progress_bar = None
    if show_progress:
        progress_bar = TQDMProgressBar(n_trials)
        callbacks.append(progress_bar)

    study.optimize(
        lambda trial: _objective(
            trial,
            X,
            y,
            param_space,
            model_name=model_name,
            scoring=scoring,
            inner_cv=inner_cv,
            random_state=random_state,
            n_jobs_model=n_jobs_model,
            n_jobs_cv=n_jobs_cv,
        ),
        n_trials=n_trials,
        callbacks=callbacks,
    )

    if progress_bar is not None:
        progress_bar.close()

    if verbose:
        print(f"[INFO] Optimization complete. Best score: {study.best_value:.4f}")
        print(f"[INFO] Best parameters: {study.best_trial.params}")
    return study
