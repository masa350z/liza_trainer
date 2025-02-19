# simulate.py
"""シミュレーション用のエントリーポイント

利確(s.lik)、損切り(s.son)の複数パラメータを試し、
・並列処理でランダムエントリーのシミュレーションを実行
・最終資産をヒートマップとして可視化
・ログやヒートマップを保存
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.simulator_core import run_simulations_with_paramgrid


def main(pair):
    print(f"[INFO] Start simulation for {pair} with random entry...")

    # === 1. 利確/損切りの候補を設定 ===
    rik_values = np.linspace(0.001/100, 0.05/100, 50)
    son_values = np.linspace(0.010/100, 0.500/100, 50)

    # === 2. シミュレーション実行 ===
    # run_simulations_with_paramgridが
    # multiprocessing.Poolを用いて並列処理し、最後に( rik x son )の行列を返す。
    final_asset_matrix = run_simulations_with_paramgrid(
        pair=pair,
        rik_values=rik_values,
        son_values=son_values,
        num_chunks=10,       # データを何分割するか
        output_logs=True    # CSVログを残すか
    )

    # === 3. ヒートマップとして可視化 ===
    os.makedirs(f"simulator_results/{pair}", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(final_asset_matrix, annot=False, fmt=".2f",
                cmap="RdYlGn", cbar=True, ax=ax, square=True)

    ax.set_title(f"Final Asset Heatmap ({pair}, Random Entry)")
    ax.set_xlabel("SON parameter")
    ax.set_ylabel("RIK parameter")

    # 軸ラベルをパラメータ値に
    ax.set_xticks(np.arange(len(son_values)) + 0.5)
    ax.set_yticks(np.arange(len(rik_values)) + 0.5)
    ax.set_xticklabels([f"{v*100:.3f}" for v in son_values],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{v*100:.3f}" for v in rik_values], rotation=0)

    heatmap_path = f"simulator_results/{pair}/heatmap_{pair}.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=120)
    plt.close()

    print(f"[INFO] Heatmap saved at {heatmap_path}")
    print("[INFO] Simulation complete.")


if __name__ == "__main__":
    main(pair='EURUSD')
    main(pair='USDJPY')