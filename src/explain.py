"""最終モデルの SHAP 可視化。

shap はバージョンに敏感なので、失敗しても警告を出してパイプラインは続行する。
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def explain_model(
    model,
    X: pd.DataFrame,
    *,
    target_class: int = 1,
    max_display: int = 20,
) -> Dict[str, Optional[plt.Figure]]:
    """SHAP の summary(dot) / summary(bar) / heatmap を返す。失敗時は None。"""
    figs: Dict[str, Optional[plt.Figure]] = {
        "shap_summary": None,
        "shap_summary_bar": None,
        "shap_heatmap": None,
    }
    try:
        import shap

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X, check_additivity=False)
        sv = shap_values.values
        if sv.ndim == 3:  # 多クラス形式なら対象クラスを取り出す
            sv = sv[:, :, target_class]

        figs["shap_summary"] = plt.figure()
        shap.summary_plot(sv, X, max_display=max_display, show=False)
        plt.close()

        figs["shap_summary_bar"] = plt.figure()
        shap.summary_plot(sv, X, plot_type="bar", max_display=max_display, show=False)
        plt.close()

        fig_hm, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            np.abs(sv), cmap="coolwarm", xticklabels=X.columns, yticklabels=False, ax=ax
        )
        ax.set_title(f"SHAP Heatmap (class {target_class})")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Sample")
        fig_hm.tight_layout()
        plt.close()
        figs["shap_heatmap"] = fig_hm

        print("[INFO] SHAP visualizations generated successfully.")
    except Exception as e:
        print(f"[WARN] SHAP visualization skipped: {e}")

    return figs
