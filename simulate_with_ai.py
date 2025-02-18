# simulate_with_ai.py
"""
AIモデル(weightsファイル)を用いてエントリー方向を決定し、
利確/損切り (rik, son) を指定してシミュレーションを行うスクリプト。

全ステップでポジションを判定し、資産を更新し、ステップごとにログを保存。
パラレル処理は行わない（単一プロセス）。

Usage:
  python simulate_with_ai.py --pair EURUSD \
      --weights results/EURUSD/Affine_k30_f5_20250126-230851/best_model.weights.h5 \
      --k 30 --rik 0.001 --son 0.01
"""

import argparse
import os
import csv
import numpy as np
import tensorflow as tf
from modules.data_loader import load_csv_data
from modules.models import build_simple_affine_model  # 例としてAffineモデル構造を使用

# 必要な分だけGPUメモリを確保する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="EURUSD",
                        choices=["USDJPY", "EURUSD"],
                        help="Which currency pair to simulate.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (saved via model.save_weights(...)).")
    parser.add_argument("--k", type=int, default=30,
                        help="Window size (number of past prices) for the model input.")
    parser.add_argument("--rik", type=float, default=0.001,
                        help="Profit threshold (fraction) for take profit.")
    parser.add_argument("--son", type=float, default=0.01,
                        help="Loss threshold (fraction) for stop loss.")
    args = parser.parse_args()

    pair = args.pair
    weights_path = args.weights
    k = args.k
    rik = args.rik
    son = args.son

    print(f"[INFO] Start AI-based simulation for {pair}")
    print(f"      Model weights: {weights_path}")
    print(f"      k={k}, rik={rik}, son={son}")

    # 1) ヒストリカルデータ読み込み
    csv_file = f"data/sample_{pair}_1m.csv"
    print(f"[INFO] Loading CSV data from: {csv_file}")
    timestamps, prices = load_csv_data(csv_file)
    prices_arr = np.array(prices, dtype=np.float32)
    print(f"[INFO] Loaded {len(prices_arr)} price data points.")

    # 2) モデル構築 + ロード
    #    「model.save_weights()」で保存したファイルを読み込むために
    #    事前に全く同じ構造のモデルを作ってから「model.load_weights()」する
    print("[INFO] Building the model architecture (Affine example).")
    model = build_simple_affine_model(input_dim=k)
    print(f"[INFO] Loading weights from {weights_path}")
    model.load_weights(weights_path)
    print("[INFO] Weights loaded successfully.")

    # 3) シミュレーション
    #    pos=0, asset=0 で開始。price_bufferに過去k個の価格を貯めて推論。
    print("[INFO] Starting simulation...")

    pos = 0          # +1=LONG, -1=SHORT, 0=NOPOS
    asset = 0.0
    entry_price = 0.0
    price_buffer = []

    step_assets = []  # ステップごとの資産ログ (step, asset)

    for step_i, price in enumerate(prices_arr):
        # 進捗をprint (例: 10000ステップあるうち毎1000ステップで状況を表示)
        if step_i % 10000 == 0:
            print(
                f"  Step {step_i}/{len(prices_arr)} ... Current asset={asset:.4f}")

        # まず、price_bufferに当日価格を追加
        price_buffer.append(price)
        if len(price_buffer) > k:
            price_buffer.pop(0)

        # もしpos!=0なら、利確/損切りをチェック
        if pos != 0:
            diff = (price - entry_price) * pos
            # 利確
            if diff > rik * entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0
            # 損切り
            elif diff < -son * entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0

        # pos=0 かつ price_bufferが十分たまっていれば、AI判定でエントリー
        if pos == 0 and len(price_buffer) == k:
            # ---- ここでMin-Max正規化を行う ----
            buffer_array = np.array(price_buffer, dtype=np.float32)
            min_val = buffer_array.min()
            max_val = buffer_array.max()

            scaled_array = (buffer_array - min_val) / \
                (max_val - min_val + 1e-8)

            # モデルの入力は (1, k) 形
            data_in = scaled_array.reshape(1, k)
            pred = model(data_in, training=False).numpy()  # shape=(1,2)

            # pred[0,0] => "上がる"の確率, pred[0,1] => "下がる"の確率
            if pred[0][0] > 0.5:
                pos = 1
            else:
                pos = -1
            entry_price = price

        # ステップごとの資産を記録
        step_assets.append((step_i, float(asset)))

    # シミュレーション終了
    print("[INFO] Simulation completed.")
    print(f"[INFO] Final asset: {asset:.4f}")

    # 4) ログの書き込み
    out_dir = f"simulator_results/{pair}_AI_logs"
    os.makedirs(out_dir, exist_ok=True)
    log_name = f"log_ai_k{k}_rik{rik:.6f}_son{son:.6f}.csv"
    log_path = os.path.join(out_dir, log_name)
    print(f"[INFO] Saving step-by-step asset log to {log_path}")

    with open(log_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["step", "asset"])
        for (stp, val) in step_assets:
            writer.writerow([stp, val])

    print(f"[INFO] Finished. Log saved. Final asset: {asset:.4f}")


if __name__ == "__main__":
    main()
