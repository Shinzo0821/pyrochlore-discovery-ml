"""config を 1 本だけ指定して実行する。

    python run_one.py config/active/demo.yaml

`python src/main.py` は `config/active/*.yaml` を**順に**処理するので、config が複数ある
と直列になる。config ごとにこのスクリプトを別プロセスで起動すれば並列に流せる。
データが小さくオーバーヘッド律速になりやすいワークロードなので、多コア機では n_jobs を
上げるより config ごとにプロセスを分けるほうが効く。

    nohup python -u run_one.py config/active/exp_a.yaml > exp_a.log 2>&1 &
    nohup python -u run_one.py config/active/exp_b.yaml > exp_b.log 2>&1 &

`main.py` と同じくリポジトリのルートから実行すること。`src/` を import path に足すのと、
`.env` の `DATA_ROOT` が相対パスでも解決できるようにするため cwd に依存する。
"""

import sys
from pathlib import Path

import matplotlib

# SSH 越しの実行で GUI バックエンドを掴みに行かないようにする（図の保存だけなら Agg で足りる）
matplotlib.use("Agg")

sys.path.insert(0, "src")

from main import run_pipeline  # noqa: E402  （sys.path を通してから import する必要がある）


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(sys.argv[0]).name} <config.yaml>")

    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        sys.exit(f"config が見つかりません: {config_path}")
    if not Path("src").is_dir():
        sys.exit("リポジトリのルートで実行してください（src/ が見つかりません）")

    print(f"\n\n===== Running experiment with {config_path.name} =====")
    run_pipeline(str(config_path))


if __name__ == "__main__":
    main()
