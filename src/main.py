"""パイプライン本体。

各モデル（NP-Pyro / AP-HP）について
  1. nested CV（外側 n_splits × n_repeats、内側 Optuna）で out-of-fold 確率と指標を得る
  2. 全学習データで Optuna を 1 回回して最終モデルを作る
  3. 学習行は OOF 確率、未学習行は最終モデルの確率を使って全件に確率を付ける
を行い、最後に 2 モデルの結果をマージして候補を抽出する。

最終モデルは学習データを記憶しているため、その自己評価の指標は出力しない。
"""

from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from cv import aggregate_oof, run_nested_cv
from data import load_dataset, make_mode_data
from evaluate import (
    metrics_per_fold,
    metrics_per_repeat,
    plot_calibration_oof,
    plot_confusion_matrix_oof,
    plot_feature_importance_cv,
    plot_feature_importance_final,
    plot_optimization_history,
    plot_pr_curve_oof,
    plot_roc_oof,
    summarize_metrics,
    threshold_sweep,
)
from explain import explain_model
from model import build_model, merge_best_params
from optimizer import run_optimization
from predictor import build_mode_predictions, merge_predictions, predict_proba, shortlist_new_pyro
from utils import (
    create_output_dirs,
    save_dataframe,
    save_figures_dict,
    save_pickle,
    save_yaml,
)

MODES = ["np_p", "ap_hp"]

# modes.<mode> に必ず書いておくキー。デフォルトとのマージはしないので、
# 片方のモードにだけ書き忘れると静かに別設定で走ってしまう。ここで落とす
REQUIRED_MODE_KEYS = [
    "optuna.n_trials",
    "optuna.n_trials_final",
    "optuna.inner_cv.n_splits",
    "optuna.inner_cv.n_repeats",
    "optuna.n_jobs_cv",
    "optuna.n_jobs_model",
    "optuna.random_state",
    "optuna.scoring",
    "optuna.param_space",
    "prediction.threshold",
    "evaluation.top_n_percentages",
]

# 両モードの共通設定。ここが欠けても sklearn / OmegaConf のデフォルトで
# 走ってしまい、後から出力を見ても気づけない
REQUIRED_COMMON_KEYS = [
    "model.name",
    "model.random_state",
    "model.n_jobs",
    "cv.n_splits",
    "cv.n_repeats",
    "cv.random_state",
    "selected_descriptors",
    "prediction.threshold_np_p",
]

# param_space に必ず書いておく RandomForestClassifier のパラメータ。
# 書き忘れると sklearn のデフォルト（例: n_estimators=100）で静かに走るので、
# 値ではなくキーの有無だけを見る（max_depth: null などは正当な指定）
REQUIRED_PARAM_SPACE_KEYS = [
    "n_estimators",
    "criterion",
    "max_features",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_leaf_nodes",
    "bootstrap",
    "class_weight",
]


def validate_config(config) -> None:
    """`modes.np_p` / `modes.ap_hp` が完全な形で書かれているか検査する。

    デフォルトとのマージをしない設計なので、書き忘れは静かに別設定での実行になる。
    キーの有無だけでなく、閾値の範囲もここで落とす。
    """
    if "modes" not in config:
        raise KeyError(
            "config に `modes` がありません。NP-Pyro / AP-HP の設定は "
            "`modes.np_p` / `modes.ap_hp` に分けて書いてください"
        )
    problems = []
    for mode in MODES:
        if mode not in config.modes:
            problems.append(f"modes.{mode} がありません")
            continue
        mcfg = config.modes[mode]
        for key in REQUIRED_MODE_KEYS:
            if OmegaConf.select(mcfg, key, default=None) is None:
                problems.append(f"modes.{mode}.{key} がありません")

        param_space = OmegaConf.select(mcfg, "optuna.param_space", default=None)
        if param_space is not None:
            for key in REQUIRED_PARAM_SPACE_KEYS:
                if key not in param_space:
                    problems.append(
                        f"modes.{mode}.optuna.param_space.{key} がありません"
                        "（sklearn のデフォルトで静かに走るため明示が必要）"
                    )

        threshold = OmegaConf.select(mcfg, "prediction.threshold", default=None)
        if threshold is not None and not 0.0 < float(threshold) <= 1.0:
            problems.append(
                f"modes.{mode}.prediction.threshold は 0 より大きく 1 以下: {threshold}"
            )

    for key in REQUIRED_COMMON_KEYS:
        if OmegaConf.select(config, key, default=None) is None:
            problems.append(f"{key} がありません")

    threshold_np_p = OmegaConf.select(config, "prediction.threshold_np_p", default=None)
    if threshold_np_p is not None and not 0.0 < float(threshold_np_p) <= 1.0:
        problems.append(
            f"prediction.threshold_np_p は 0 より大きく 1 以下: {threshold_np_p}"
        )

    if problems:
        raise KeyError("config の検査に失敗しました:\n  - " + "\n  - ".join(problems))


def run_pipeline(config_path: str) -> None:
    # 1. 前準備
    load_dotenv()
    config = OmegaConf.load(config_path)
    validate_config(config)
    create_output_dirs(config.output_dir)
    save_yaml(config, Path(config.output_dir.results) / f"config_{config.exp_name}.yaml")

    dataset = load_dataset(config.dataset_path)
    descriptors = list(config.selected_descriptors)

    frames = {}

    for mode in MODES:
        print(f"\n\n===== Processing: {mode} model =====")
        # モード別設定。デフォルトとのマージはしない（config に完全な形で書く）
        mcfg = config.modes[mode]
        top_n = list(mcfg.evaluation.top_n_percentages)
        # pred_* 列、metrics_folds.csv の閾値依存指標、混同行列のすべてで同じ値を使う
        threshold = float(mcfg.prediction.threshold)
        out = config.output_dir[mode]
        results_dir, figures_dir, pkl_dir = (
            Path(out.results),
            Path(out.figures),
            Path(out.pkl),
        )

        # 2-1. モードごとのデータ（標準化なし。RF はスケール不変）
        md = make_mode_data(dataset.df, descriptors, mode)

        # 2-2. nested CV: 外側 fold ごとに内側 Optuna を回して OOF 確率を集める
        cv_result = run_nested_cv(
            md.X,
            md.y,
            model_name=config.model.name,
            param_space=mcfg.optuna.param_space,
            n_splits=config.cv.n_splits,
            n_repeats=config.cv.n_repeats,
            cv_random_state=config.cv.random_state,
            optuna_random_state=mcfg.optuna.random_state,
            n_trials=mcfg.optuna.n_trials,
            inner_cv=mcfg.optuna.inner_cv,
            scoring=mcfg.optuna.scoring,
            threshold=threshold,
            n_jobs=config.model.n_jobs,
            n_jobs_model=mcfg.optuna.n_jobs_model,
            n_jobs_cv=mcfg.optuna.n_jobs_cv,
        )
        save_dataframe(cv_result.oof, results_dir / "oof_predictions.csv", index=False)
        save_dataframe(cv_result.best_params, results_dir / "best_params_folds.csv", index=False)
        save_dataframe(cv_result.importances, results_dir / "feature_importance_folds.csv", index=False)

        # 2-3. OOF からの評価（報告値はここだけを使う）
        folds = metrics_per_fold(cv_result.oof, top_n)
        repeats = metrics_per_repeat(cv_result.oof, top_n)
        save_dataframe(folds, results_dir / "metrics_folds.csv", index=False)
        save_dataframe(summarize_metrics(folds, repeats), results_dir / "metrics_summary.csv")
        save_dataframe(repeats, results_dir / "metrics_repeats.csv", index=False)

        # 閾値を決め直すための表。採用値は config の prediction.threshold
        save_dataframe(
            threshold_sweep(cv_result.oof),
            results_dir / "threshold_sweep.csv",
            index=False,
        )

        # NP-Pyro は負例の大半が UK（未報告）なので、calibration は
        # 「未報告であることへの当てはまり」を測っている点に注意が要る
        calib_note = (
            "note: negatives include unreported (UK) rows" if mode == "np_p" else None
        )
        save_figures_dict(
            {
                "confusion_matrix_oof": plot_confusion_matrix_oof(cv_result.oof),
                "roc_oof": plot_roc_oof(cv_result.oof),
                "pr_curve_oof": plot_pr_curve_oof(cv_result.oof, threshold=threshold),
                "calibration_oof": plot_calibration_oof(cv_result.oof, note=calib_note),
                "feature_importance_cv": plot_feature_importance_cv(cv_result.importances),
            },
            figures_dir,
        )

        # 2-4. 最終モデル: 全学習データで Optuna を 1 回 → 学習
        print(f"\n[INFO] --- {mode}: final model on all {len(md.X)} training rows ---")
        study_final = run_optimization(
            model_name=config.model.name,
            X=md.X,
            y=md.y,
            param_space=mcfg.optuna.param_space,
            n_trials=mcfg.optuna.n_trials_final,
            random_state=mcfg.optuna.random_state,
            scoring=mcfg.optuna.scoring,
            inner_cv=mcfg.optuna.inner_cv,
            n_jobs_model=mcfg.optuna.n_jobs_model,
            n_jobs_cv=mcfg.optuna.n_jobs_cv,
        )
        best_params_final = merge_best_params(mcfg.optuna.param_space, study_final.best_params)
        model_final = build_model(
            best_params_final,
            model_name=config.model.name,
            random_state=config.model.random_state,
            n_jobs=config.model.n_jobs,
        )
        model_final.fit(md.X, md.y)

        save_pickle(model_final, pkl_dir / "model_final.pkl")
        save_pickle(study_final, pkl_dir / "study_final.pkl")
        save_yaml(best_params_final, results_dir / "best_params_final.yaml")

        # 最終モデルからは重要度と SHAP だけを出す（自己評価の指標は出さない）
        save_figures_dict(
            {
                "feature_importance_final": plot_feature_importance_final(
                    model_final, md.feature_names
                ),
                "optimization_history_final": plot_optimization_history(study_final),
                **explain_model(model_final, md.X),
            },
            figures_dir,
        )

        # 2-5. 全件推論: 学習行は OOF、未学習行は最終モデル
        frames[mode] = build_mode_predictions(
            mode=mode,
            oof_agg=aggregate_oof(cv_result.oof),
            final_proba=predict_proba(model_final, md.X_predict),
            train_ids=md.train_ids,
            threshold=threshold,
        )

    # 3. 両モデルの予測結果を結合して出力
    df_with_preds = merge_predictions(dataset=dataset, frames=frames)
    save_dataframe(
        df_with_preds,
        Path(config.output_dir.results) / f"predictions_all_{config.exp_name}.csv",
    )

    # 4. 新規パイロクロア候補を抽出して保存
    cands = shortlist_new_pyro(
        input_df_or_path=df_with_preds,
        threshold_np_p=config.prediction.threshold_np_p,
    )
    save_dataframe(
        cands,
        Path(config.output_dir.results) / f"new_pyro_candidates_{config.exp_name}.csv",
    )
    print(f"[INFO] Candidates: {len(cands)} rows")


if __name__ == "__main__":
    config_dir = Path("config/active")
    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        print(
            f"警告: {config_dir} ディレクトリ内にYAMLファイルが見つかりません。"
            "リポジトリのルートで実行しているか確認してください。"
        )
    else:
        for cfg in config_files:
            print(f"\n\n\n===== Running experiment with {cfg.name} =====")
            run_pipeline(str(cfg))
