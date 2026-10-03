"""データ読み込みとモードごとの (X, y, meta) 生成。

方針:
- CSV は 1 回だけ読む
- 行の識別子 `row_id` を読み込み時に生成する（A2B2O7 は非一意なので単独ではキーにならない）
- StandardScaler は使わない。RF はスケール不変で、標準化は
  n_diff_abs をキーとして復元するハックの原因になっていた
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import pandas as pd

Mode = Literal["np_p", "ap_hp"]

# 記述子ではない識別・ラベル列。出力 CSV とマージに使う
ID_COLUMNS = ["A2B2O7", "A", "B", "nA", "nB", "exp_label", "HPS", "t"]


@dataclass
class Dataset:
    """入力 CSV 1 ファイル分。CSV はここで 1 回だけ読む。"""

    raw: pd.DataFrame  # 元 CSV そのまま（index = A2B2O7）。出力 CSV の土台
    df: pd.DataFrame  # 推論対象（index = row_id）。アクチノイドと nA NaN を除去済み


@dataclass
class ModeData:
    """1 モデル分のデータ一式。index はすべて row_id。"""

    mode: Mode
    X: pd.DataFrame  # 学習対象の特徴量
    y: pd.Series  # 学習対象のラベル（0/1）
    X_predict: pd.DataFrame  # 推論対象（全行）の特徴量
    meta: pd.DataFrame  # 推論対象（全行）の識別列
    feature_names: List[str]

    @property
    def train_ids(self) -> pd.Index:
        return self.X.index


def make_row_id(df: pd.DataFrame) -> pd.Series:
    """`row_id = "{A2B2O7}_nA{nA}"`。A2B2O7 は非一意（同一組成で nA が 2 と 3 の 2 行がある）。"""
    return df.index.astype(str) + "_nA" + df["nA"].astype(int).astype(str)


def load_dataset(data_path: str) -> Dataset:
    """CSV を 1 回読み、元データと推論対象データ（row_id を index に持つ）を返す。

    除去するのは以下だけ。`t` が NaN の行（価数不明・Disorder 等）は推論対象として残す。
      - アクチノイド（原子番号 89 以上）
      - atomic_number_sum が NaN
      - nA が NaN（不自然な価数。row_id が作れない）
    """
    print(f"[INFO] Loading data from {data_path}")
    raw = pd.read_csv(data_path, index_col=0, header=0)

    keep = ~(
        (raw["A-site_atomic_number"] >= 89)
        | (raw["B-site_atomic_number"] >= 89)
        | (raw["atomic_number_sum"].isna())
        | (raw["nA"].isna())
    )
    df = raw[keep].copy()
    df[df.index.name or "A2B2O7"] = df.index
    df.index = pd.Index(make_row_id(df), name="row_id")

    if not df.index.is_unique:
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"row_id が一意ではありません: {dupes[:10]}")

    check_invariants(raw)
    print(f"[INFO] Rows: {len(raw)} -> {len(df)} (actinide / nA NaN removed)")
    return Dataset(raw=raw, df=df)


def check_invariants(df: pd.DataFrame) -> None:
    """入力データの不変条件を検査する。

    `HPS=0` は「常圧で安定」ではなく「パイロクロアが得られていない」の意で、
    `exp_label` の 0/1 と 1:1 で対応する。
    ここが崩れると NP-Pyro と AP-HP のラベルが静かに食い違う。
    """
    bad = df[(df["exp_label"] <= 1) != (df["HPS"] == 0)]
    if not bad.empty:
        raise ValueError(
            "データ不変条件違反 (exp_label ∈ {0,1} ⟺ HPS == 0) が "
            f"{len(bad)} 行あります: {list(bad.index[:10])}"
        )


def split_known_unknown(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """`t` が NaN でない行（学習対象）と NaN の行（推論のみ）に分ける。"""
    known = df_all.dropna(subset=["t"])
    unknown = df_all[df_all["t"].isna()]
    return known, unknown


def make_labels(df_known: pd.DataFrame, mode: Mode) -> Tuple[pd.DataFrame, pd.Series]:
    """モードごとに学習対象行とラベルを作る。

    - np_p : 全学習対象行。y = (exp_label >= 2)。合成圧力は問わない
    - ap_hp: HPS != 0 の行のみ（パイロクロアが得られている行）。y = (HPS == 3)
    """
    if mode == "np_p":
        df_fit = df_known
        y = (df_fit["exp_label"] >= 2).astype(int)
    elif mode == "ap_hp":
        df_fit = df_known[df_known["HPS"] != 0]
        y = (df_fit["HPS"] == 3).astype(int)
    else:
        raise ValueError("mode must be either 'np_p' or 'ap_hp'")

    y.name = "y"
    print(
        f"[INFO] {mode}: fit rows={len(df_fit)}, positive={int(y.sum())}, "
        f"negative={int((y == 0).sum())}"
    )
    return df_fit, y


def make_mode_data(
    df_all: pd.DataFrame, selected_descriptors: List[str], mode: Mode
) -> ModeData:
    """読み込み済みデータからモード 1 つ分の ModeData を作る。"""
    missing = [c for c in selected_descriptors if c not in df_all.columns]
    if missing:
        raise ValueError(f"記述子が入力データに見つかりません: {missing}")

    df_known, _ = split_known_unknown(df_all)
    df_fit, y = make_labels(df_known, mode)

    meta_cols = [c for c in ID_COLUMNS if c in df_all.columns]
    return ModeData(
        mode=mode,
        X=df_fit[selected_descriptors],
        y=y,
        X_predict=df_all[selected_descriptors],
        meta=df_all[meta_cols],
        feature_names=list(selected_descriptors),
    )
