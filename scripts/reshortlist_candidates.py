#!/usr/bin/env python
"""既存の実験出力から候補リストだけを閾値を変えて作り直す。

再学習はしない。`predictions_all_{exp_name}.csv`（全件の予測結果）を読み直し、
`src/predictor.py` の `shortlist_new_pyro` をそのまま呼ぶ。つまり抽出条件は
本番パイプラインと同一（exp_label が 0=UK または 1=NP かつ proba_np_p >= 閾値、
proba_np_p の降順）で、閾値だけを差し替える。

パイプラインを回し直さずに閾値を下げられるのはこの候補リストだけである。
`pred_*` 列や OOF 指標・混同行列は学習時の閾値で焼き込まれているので、
そちらを変えたければ本番を回し直すしかない。

使い方（リポジトリルートから）:

    python scripts/reshortlist_candidates.py demo --threshold 0.25

`--in-place` を付けると `new_pyro_candidates_{exp_name}.csv` を上書きする
（上書き前に `.bak` を残す）。付けなければ `--out` の位置に書く。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predictor import shortlist_new_pyro  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_name", help="出力ディレクトリ名（例: demo）")
    p.add_argument("--threshold", type=float, default=0.25, help="proba_np_p の下限（既定 0.25）")
    p.add_argument("--in-place", action="store_true", help="既存の候補 CSV を上書きする")
    p.add_argument("--out", type=Path, default=None, help="出力先（--in-place を付けないとき）")
    args = p.parse_args()

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("DATA_ROOT が未設定です（.env を確認）")

    results = Path(data_root) / "outputs" / args.exp_name / "results"
    src = results / f"predictions_all_{args.exp_name}.csv"
    if not src.exists():
        raise SystemExit(f"予測結果が見つかりません: {src}")

    df = pd.read_csv(src, index_col=0)
    cands = shortlist_new_pyro(df, threshold_np_p=args.threshold)
    print(f"[INFO] threshold {args.threshold} -> {len(cands)} 件")

    dst = results / f"new_pyro_candidates_{args.exp_name}.csv" if args.in_place else args.out
    if dst is None:
        raise SystemExit("--in-place か --out のどちらかを指定してください")

    if args.in_place and dst.exists():
        bak = dst.with_name(dst.name + ".bak")
        shutil.copy2(dst, bak)
        print(f"[INFO] 旧ファイルを退避: {bak.name}")

    cands.to_csv(dst)
    print(f"[INFO] 保存: {dst}")


if __name__ == "__main__":
    main()
