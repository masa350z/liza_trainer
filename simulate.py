"""
simulate.py

学習済みのActor-Criticモデル(重みファイル)を用い、シミュレーションエピソードを実行して
ステップごとの資産変動ログを出力します。

Usage:
    python simulate.py --pair EURUSD --weights results/models/EURUSD/ActorCritic_ws30_YYYYMMDD-HHMMSS/best_model_weights.h5 --window_size 30
"""

import os
import csv
import argparse
import numpy as np
import tensorflow as tf

from modules.data_loader import load_csv_data
from modules.env import TradingEnv
from modules.models import build_actor_critic_model

# GPUメモリの必要分だけ確保する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="EURUSD",
                        choices=["EURUSD", "USDJPY"], help="通貨ペア")
    parser.add_argument("--weights", type=str,
                        required=True, help="学習済みモデルの重みファイルのパス")
    parser.add_argument("--window_size", type=int,
                        default=30, help="状態として使用する過去の価格数")
    args = parser.parse_args()

    pair = args.pair
    weights_path = args.weights
    window_size = args.window_size

    csv_file = os.path.join("data", f"sample_{pair}_1m.csv")
    print(f"[INFO] Loading CSV data from: {csv_file}")
    _, prices = load_csv_data(csv_file)
    if len(prices) < window_size + 1:
        raise ValueError("価格データがwindow_sizeよりも短いため、シミュレーションできません。")

    # 環境の構築
    env = TradingEnv(prices, window_size)
    feature_dim = 1
    num_actions = 4

    # モデル構築
    model = build_actor_critic_model(
        time_steps=window_size, feature_dim=feature_dim, num_actions=num_actions, lstm_units=64)
    print(f"[INFO] Loading model weights from: {weights_path}")
    model.load_weights(weights_path)
    print("[INFO] Model weights loaded.")

    # シミュレーション実行(エピソード1回分)
    state = env.reset()
    pos = env.position  # 状態管理は環境側で行うため、こちらは参考情報
    asset = 0.0  # シミュレーション中の累積報酬(資産)として算出
    step_log = []

    while True:
        state_input = state[None, ...]
        policy, _ = model(state_input, training=False)
        policy = policy.numpy()[0]
        action = np.argmax(policy)  # 最尤行動を採用
        next_state, reward, done = env.step(action)
        asset += reward
        step_log.append((env.current_index, asset, action, reward))
        if done:
            break
        state = next_state

    print(f"[INFO] Simulation complete. Final asset: {asset:.4f}")

    # ログ保存
    out_dir = os.path.join("results/simulations", pair + "_AI_logs")
    os.makedirs(out_dir, exist_ok=True)
    log_filename = f"log_ai_ws{window_size}.csv"
    log_path = os.path.join(out_dir, log_filename)
    with open(log_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["step_index", "asset", "action", "reward"])
        for row in step_log:
            writer.writerow(row)
    print(f"[INFO] Step-by-step log saved to: {log_path}")


if __name__ == "__main__":
    main()
