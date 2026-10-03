"""モデル生成の単一窓口。

config の `optuna.param_space` には固定値（直値）と探索空間（dict）が混在している。
`study.best_params` には探索キーしか入らないため、固定値を足し直さないと
最終モデルが sklearn のデフォルトに落ちる。その事故を防ぐため、
optimizer / trainer / cv はすべてこのモジュール経由でモデルを作る。
"""

from typing import Any, Dict, Tuple

import optuna
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier

# build_model 側で必ず指定するため、param_space に書かれていても無視するキー
_RESERVED_KEYS = ("random_state", "n_jobs")


def _to_plain_dict(param_space) -> Dict[str, Any]:
    """DictConfig でも素の dict でも受けられるようにする。"""
    if OmegaConf.is_config(param_space):
        return OmegaConf.to_container(param_space, resolve=True)
    return dict(param_space)


def split_param_space(param_space) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """param_space を (固定値, 探索空間) に分ける。

    `{"type": ...}` を持つ dict は探索空間、それ以外（None やスカラー）は固定値。
    """
    fixed: Dict[str, Any] = {}
    search: Dict[str, Any] = {}
    for name, space in _to_plain_dict(param_space).items():
        if name in _RESERVED_KEYS:
            continue
        if isinstance(space, dict) and "type" in space:
            search[name] = space
        else:
            fixed[name] = space
    return fixed, search


def suggest_params(trial: optuna.trial.Trial, param_space) -> Dict[str, Any]:
    """固定値 + trial が選んだ探索値 を合わせた完全なパラメータ dict を返す。"""
    fixed, search = split_param_space(param_space)
    params = dict(fixed)
    for name, space in search.items():
        space_type = space["type"]
        if space_type == "int":
            params[name] = trial.suggest_int(
                name, space["low"], space["high"], log=space.get("log", False)
            )
        elif space_type == "float":
            params[name] = trial.suggest_float(
                name, space["low"], space["high"], log=space.get("log", False)
            )
        elif space_type == "categorical":
            params[name] = trial.suggest_categorical(name, space["choices"])
        else:
            raise ValueError(f"Unsupported param type: {space_type}")
    return params


def merge_best_params(param_space, best_params: Dict[str, Any]) -> Dict[str, Any]:
    """`study.best_params`（探索キーのみ）に param_space の固定値を足して完全な形にする。"""
    fixed, _ = split_param_space(param_space)
    params = dict(fixed)
    params.update({k: v for k, v in dict(best_params).items() if k not in _RESERVED_KEYS})
    return params


def build_model(
    params: Dict[str, Any],
    model_name: str = "random_forest",
    random_state: int = 227,
    n_jobs: int = -1,
):
    """完全なパラメータ dict から未学習のモデルを作る。"""
    if model_name != "random_forest":
        raise ValueError(f"Unknown model_name: {model_name}")
    params = {k: v for k, v in params.items() if k not in _RESERVED_KEYS}
    return RandomForestClassifier(**params, random_state=random_state, n_jobs=n_jobs)
