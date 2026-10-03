import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
from omegaconf import OmegaConf, DictConfig
import yaml


def create_output_dirs(output_dir_cfg):
    """
    OmegaConfのoutput_dir_cfgを展開して
    必要なディレクトリをまとめて作成する
    """
    # resolve=True を指定して補間を解決し、Pythonのdictに変換
    output_dir_cfg = OmegaConf.to_container(output_dir_cfg, resolve=True)

    def _mkdir_recursive(cfg):
        for val in cfg.values():
            if isinstance(val, str):
                path = Path(val)
                path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Created: {path}")
            elif isinstance(val, dict):
                _mkdir_recursive(val)

    _mkdir_recursive(output_dir_cfg)


def save_yaml(config: Union[dict, DictConfig], path: str) -> None:
    """
    YAML形式で設定を保存する関数。
    
    - 通常のdict、またはOmegaConfのDictConfigに対応。
    - OmegaConfオブジェクトの場合は、自動で辞書に変換して保存。

    Args:
        config (dict or DictConfig): 保存する設定データ
        path (str): 保存先のファイルパス
    """
    # OmegaConfのDictConfigをdictに変換
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"[INFO] Saved YAML config to: {path}")

def save_dataframe(df: pd.DataFrame, path: str, index: bool = True) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path, index=index)
    print(f"[INFO] Saved DataFrame to: {path}")

def save_pickle(obj: Any, path: str) -> None:
    """任意のPythonオブジェクトを pickle 形式で保存する"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved object to: {path}")

def save_figures_dict(figs: Dict[str, Any], dir_path, suffix: str = ".png") -> None:
    """{図の名前: Figure} をディレクトリにまとめて保存する。値が None の図は飛ばす。"""
    dir_path = Path(dir_path)
    for name, fig in figs.items():
        if fig is None:
            print(f"[WARN] Figure '{name}' is None; skipped.")
            continue
        path = dir_path / f"{name}{suffix}"
        try:
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved figure to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save figure to: {path}. Error: {e}")
