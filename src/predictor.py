"""推論結果のまとめと候補抽出。

マージキーは `row_id`。以前は標準化後の n_diff_abs の符号から元の 1/3 を
復元して (A2B2O7, n_diff_abs) で結合していたが、標準化の廃止とともに撤去した。
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from data import Dataset, make_row_id


def predict_proba(model: Any, X: pd.DataFrame) -> pd.Series:
    """学習済みモデルで陽性確率を推定する。index は X のまま（row_id）。"""
    print(f"[INFO] Predicting {len(X)} rows with the final model...")
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="proba")


def build_mode_predictions(
    mode: str,
    oof_agg: pd.DataFrame,
    final_proba: pd.Series,
    train_ids: pd.Index,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """学習行の OOF 確率と、未学習行の最終モデル確率を 1 つにまとめる。

    学習行には「その行を含まないモデル」の確率（OOF、repeats 平均）を使う。
    最終モデルは全学習行を記憶しているので、学習行に使うと確率が正解側に偏る。

    Args:
        mode: "np_p" または "ap_hp"
        oof_agg: `cv.aggregate_oof` の戻り値（index=row_id）
        final_proba: 最終モデルによる推論対象全行の確率（index=row_id）
        train_ids: 学習に使った row_id。全行が OOF で埋まることの検証に使う
        threshold: pred 列を作るときの確率しきい値

    Returns:
        index=row_id, columns=[pred_{mode}, proba_{mode}, proba_{mode}_std, source_{mode}]
    """
    # 学習行が 1 行でも OOF から漏れると、その行に最終モデル（自分を学習済み）の
    # 確率が入ってしまう。静かに起きると気づけないのでここで落とす
    missing_oof = train_ids.difference(oof_agg.index)
    if not missing_oof.empty:
        raise ValueError(
            f"{mode}: 学習行 {len(missing_oof)} 件に OOF 確率がありません: "
            f"{list(missing_oof[:5])}"
        )
    not_predicted = train_ids.difference(final_proba.index)
    if not not_predicted.empty:
        raise ValueError(
            f"{mode}: 学習行 {len(not_predicted)} 件が推論対象に含まれていません: "
            f"{list(not_predicted[:5])}"
        )

    proba = final_proba.copy().astype(float)
    std = pd.Series(np.nan, index=proba.index, dtype=float)
    source = pd.Series("final", index=proba.index, dtype=object)

    common = proba.index.intersection(oof_agg.index)
    proba.loc[common] = oof_agg.loc[common, "proba_mean"]
    std.loc[common] = oof_agg.loc[common, "proba_std"]
    source.loc[common] = "oof"

    n_oof = int((source == "oof").sum())
    if n_oof != len(train_ids):
        raise ValueError(
            f"{mode}: OOF 由来の行数 {n_oof} が学習行数 {len(train_ids)} と一致しません"
        )
    print(f"[INFO] {mode}: proba source -> oof {n_oof} rows, final {len(source) - n_oof} rows")

    return pd.DataFrame(
        {
            f"pred_{mode}": (proba >= threshold).astype(int),
            f"proba_{mode}": proba,
            f"proba_{mode}_std": std,
            f"source_{mode}": source,
        }
    )


def merge_predictions(
    dataset: Dataset,
    frames: Dict[str, pd.DataFrame],
    after_col: str = "HPS",
) -> pd.DataFrame:
    """元データに各モデルの予測結果を row_id で結合する。

    Args:
        dataset: `load_dataset` の戻り値
        frames: {"np_p": DataFrame, "ap_hp": DataFrame}。
            index は row_id、列名はすでにモード接尾辞つき
        after_col: 予測列を差し込む位置（この列の直後）

    Returns:
        pd.DataFrame: 元データ全行（index=A2B2O7）に予測列を足したもの。
        アクチノイド等で推論対象外だった行の予測列は NaN になる。
    """
    print("[INFO] Merging predictions on row_id")
    out = dataset.raw.copy()

    # 元データ側の row_id。nA が NaN の行はキーを持たない = 予測なし
    row_id = pd.Series(np.nan, index=out.index, dtype=object)
    has_na = out["nA"].notna()
    row_id[has_na] = make_row_id(out[has_na])

    pred_cols = []
    for frame in frames.values():
        for col in frame.columns:
            out[col] = row_id.map(frame[col])
            pred_cols.append(col)

    # 列順: after_col の直後に予測列をまとめる
    remaining = [c for c in out.columns if c not in pred_cols]
    if after_col not in remaining:
        raise ValueError(f"列順の基準列 '{after_col}' が出力に見つかりません")
    pos = remaining.index(after_col) + 1
    return out[remaining[:pos] + pred_cols + remaining[pos:]]


def shortlist_new_pyro(
    input_df_or_path: Union[pd.DataFrame, str],
    threshold_np_p: float = 0.5,
    proba_col: str = "proba_np_p",
) -> pd.DataFrame:
    """新規パイロクロア候補（exp_label が 0=UK または 1=NP かつ NP-Pyro 確率が閾値以上）を抽出する。

    Args:
        input_df_or_path: マージ済み DataFrame または CSV/Excel のパス
        threshold_np_p: NP-Pyro の確率しきい値
        proba_col: 閾値判定に使う確率列。OOF 平均列に切り替える場合はここを変える

    Returns:
        NP-Pyro 確率の降順に並べた候補 DataFrame
    """
    if isinstance(input_df_or_path, pd.DataFrame):
        df = input_df_or_path.copy()
    elif isinstance(input_df_or_path, str):
        if input_df_or_path.endswith(".csv"):
            df = pd.read_csv(input_df_or_path, index_col=0)
        elif input_df_or_path.endswith(".xlsx"):
            df = pd.read_excel(input_df_or_path, index_col=0)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
    else:
        raise TypeError("input_df_or_path must be a DataFrame or a filepath string.")

    base_cols = ["nA", "exp_label", "HPS"]
    pred_cols = [c for c in df.columns if c.startswith(("pred_", "proba_", "source_"))]
    out_cols = base_cols + pred_cols  # A2B2O7 は index なので含めない

    missing = [c for c in base_cols + [proba_col] if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が見つかりません: {missing}")

    # UK(0) または NP(1) で、NP-Pyro 確率がしきい値以上
    mask = df["exp_label"].isin([0, 1]) & (df[proba_col] >= threshold_np_p)
    out = df.loc[mask, out_cols].sort_values(proba_col, ascending=False)

    # 候補の確率が in-sample（最終モデルの自己予測）由来だと成功率の比較が成立しない
    source_col = proba_col.replace("proba_", "source_")
    if source_col in out.columns:
        n_final = int((out[source_col] == "final").sum())
        if n_final:
            print(
                f"[WARN] 候補 {len(out)} 件のうち {n_final} 件の {proba_col} は "
                "最終モデル由来（t が NaN で学習に含まれない行）。OOF ではない"
            )
    return out
