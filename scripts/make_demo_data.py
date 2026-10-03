#!/usr/bin/env python
"""パイプラインを動かすための**合成データ**を作る。

研究で使っている実データ（文献から集めた A2B2O7 の合成報告）は未発表のため、この
リポジトリには含めていない。代わりに、同じ列構成を持つ架空のデータをここで生成する。

- 元素は実在しない記号（A サイト: Qa, Qb, ... / B サイト: Ja, Jb, ...）で、
  物性値はすべて乱数。化学的な意味は無い
- ラベルは「半径比がある範囲に入ると安定になりやすい」という単純な隠れルール + ノイズで
  付けている。実データの傾向を再現したものではない
- 実データと同じ癖だけは再現してある:
    - 陽性（合成報告あり）と大量の未報告（UK）からなる positive-unlabeled な構成
    - 同じ組成式が価数違いで 2 行ある（`A2B2O7` 単独では行を識別できない）
    - 価数が決まらない行（`nA` が NaN）、半径が不明で `t` が NaN の行（学習には使わず推論だけ行う）
    - 対象外の重元素（原子番号 89 以上）

したがって、このデータで得られる指標や候補リストは**パイプラインの動作確認用**であり、
研究結果ではない。

使い方（リポジトリのルートから）:

    python scripts/make_demo_data.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

R_OXYGEN = 1.38
N_A_SITE = 24
N_B_SITE = 24


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_elements(rng: np.random.Generator, prefix: str, n: int, valences, r_range) -> pd.DataFrame:
    """架空の元素表を作る。価数ごとの半径を持ち、取れない価数の半径は NaN。"""
    rows = []
    for i in range(n):
        symbol = prefix + chr(ord("a") + i)
        base_r = rng.uniform(*r_range)
        row = {
            "symbol": symbol,
            "atomic_number": int(rng.integers(3, 84)),
            "electronegativity": float(np.round(rng.uniform(0.8, 2.4), 2)),
        }
        # 取れる価数: 両方（5 割）、1 つ目だけ、2 つ目だけ
        pattern = rng.choice(["both", "first", "second"], p=[0.5, 0.25, 0.25])
        for k, v in enumerate(valences):
            # 2 つ目の価数のほうが半径は小さい
            available = pattern == "both" or (pattern == "first") == (k == 0)
            row[f"r_{v}"] = float(np.round(base_r - 0.08 * k, 3)) if available else np.nan
        rows.append(row)
    elements = pd.DataFrame(rows).set_index("symbol")
    # 対象外の重元素を 1 つ混ぜる（読み込み時に除去されることの確認用）
    elements.loc[elements.index[-1], "atomic_number"] = 92
    return elements


def make_dataset(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # A サイトは 3+ / 2+、B サイトは 4+ / 5+ を取る（A3+B4+ と A2+B5+ の 2 系列）
    a_elems = make_elements(rng, "Q", N_A_SITE, valences=(3, 2), r_range=(0.95, 1.30))
    b_elems = make_elements(rng, "J", N_B_SITE, valences=(4, 5), r_range=(0.55, 0.80))

    rows = []
    for a, ea in a_elems.iterrows():
        for b, eb in b_elems.iterrows():
            pairs = [
                (n_a, n_b)
                for n_a, n_b in ((3, 4), (2, 5))
                if not np.isnan(ea[f"r_{n_a}"]) and not np.isnan(eb[f"r_{n_b}"])
            ]
            if not pairs:
                # 電荷が釣り合う価数の組が無い: nA が NaN の行として残す（読み込み時に除去）
                pairs = [(np.nan, np.nan)]
            for n_a, n_b in pairs:
                known = not np.isnan(n_a)
                r_a = ea[f"r_{int(n_a)}"] if known else np.nan
                r_b = eb[f"r_{int(n_b)}"] if known else np.nan
                # 5% の行は半径が不明（t が NaN になり、学習には使わず推論だけ行う行）
                if known and rng.random() < 0.05:
                    r_a = np.nan
                rows.append(
                    {
                        "A2B2O7": f"{a}2{b}2O7",
                        "A": a,
                        "B": b,
                        "nA": n_a,
                        "nB": n_b,
                        "n_diff_abs": abs(n_b - n_a) if known else np.nan,
                        "A-site_ionic_radius": r_a,
                        "B-site_ionic_radius": r_b,
                        "A-site_atomic_number": ea["atomic_number"],
                        "B-site_atomic_number": eb["atomic_number"],
                        "A-site_electronegativity": ea["electronegativity"],
                        "B-site_electronegativity": eb["electronegativity"],
                    }
                )
    df = pd.DataFrame(rows)

    for name in ("ionic_radius", "atomic_number", "electronegativity"):
        df[f"{name}_sum"] = df[f"A-site_{name}"] + df[f"B-site_{name}"]
        df[f"{name}_diff"] = df[f"A-site_{name}"] - df[f"B-site_{name}"]
    df["r_ratio"] = df["A-site_ionic_radius"] / df["B-site_ionic_radius"]
    # ペロブスカイトの Goldschmidt 因子と同じ形の幾何学的指標（デモ用）
    df["t"] = (df["A-site_ionic_radius"] + R_OXYGEN) / (
        np.sqrt(2) * (df["B-site_ionic_radius"] + R_OXYGEN)
    )

    # --- ラベル: 隠れルール + ノイズ ---
    ratio = df["r_ratio"].fillna(df["r_ratio"].median())
    en_diff = df["electronegativity_diff"]
    p_stable = sigmoid((ratio - 1.45) * 10) * sigmoid((1.85 - ratio) * 10) * (0.75 + 0.25 * sigmoid(-en_diff))
    is_stable = rng.random(len(df)) < p_stable

    # 安定なものの 6 割だけが「合成報告あり」、残りは未報告（UK）に紛れる
    reported = is_stable & (rng.random(len(df)) < 0.6)
    # 安定でないものの一部は「パイロクロアにならない」と報告済み（NP）
    reported_negative = ~is_stable & (rng.random(len(df)) < 0.15)

    exp_label = np.zeros(len(df), dtype=int)  # 0 = UK
    exp_label[reported_negative] = 1  # 1 = NP
    # 2 = パイロクロア、3〜6 = その亜種（どれも陽性として扱う）
    exp_label[reported] = rng.choice([2, 3, 4, 5, 6], size=int(reported.sum()), p=[0.8, 0.05, 0.05, 0.05, 0.05])

    # 合成圧力: 安定域の下端に近いほど高圧合成が必要、というルール
    p_high_pressure = sigmoid((1.58 - ratio) * 12)
    needs_hp = rng.random(len(df)) < p_high_pressure
    hps = np.zeros(len(df), dtype=int)  # 0 = パイロクロアが得られていない
    hps[reported] = np.where(
        needs_hp[reported], 3, rng.choice([1, 2], size=int(reported.sum()), p=[0.9, 0.1])
    )

    df.insert(1, "exp_label", exp_label)
    df.insert(2, "HPS", hps)

    # 価数が決まらない行はラベルも持たない（UK のまま）
    unknown = df["nA"].isna()
    df.loc[unknown, ["exp_label", "HPS"]] = 0

    return df.set_index("A2B2O7")


def main() -> None:
    parser = argparse.ArgumentParser(description="パイプライン動作確認用の合成データを生成する")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("DATA_ROOT が未設定です（.env.example を .env にコピーしてください）")

    out = Path(data_root) / "data" / "processed" / "demo_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = make_dataset(seed=args.seed)
    df.to_csv(out)

    fit = df[df["t"].notna()]
    print(f"[INFO] 保存: {out}")
    print(f"[INFO] {len(df)} 行（うち t が有効な学習対象 {len(fit)} 行）")
    print(f"[INFO] 合成報告あり {int((fit['exp_label'] >= 2).sum())} 行 / うち高圧合成 {int((fit['HPS'] == 3).sum())} 行")


if __name__ == "__main__":
    main()
