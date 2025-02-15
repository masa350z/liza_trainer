"""
simulate.py (単一環境版)

学習済みのActor-Criticモデル(重みファイル)を用い、シミュレーションエピソードを1回実行し、
ステップごとの行動・リワード・資産推移をCSV出力します。

Usage:
    python simulate.py --pair USDJPY \
                       --weights results/models/USDJPY/ActorCritic_ws30_YYYYMMDD-HHMMSS/best_model.weights.h5 \
                       --window_size 30
"""

import os
import csv
import argparse
import numpy as np
import tensorflow as tf

from modules.data_loader import load_csv_data
from modules.env import TradingEnv
from modules.models import LSTM_ActorCriticModel, Affine_ActorCriticModel

# GPUメモリの必要分だけ確保
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

    csv_file = os.path.join("data", f"{pair}_1m.csv")
    print(f"[INFO] Loading CSV data from: {csv_file}")
    _, prices = load_csv_data(csv_file)
    if len(prices) < window_size + 1:
        raise ValueError("価格データがwindow_sizeよりも短いため、シミュレーションできません。")

    # 環境の構築
    env = TradingEnv(prices, window_size)
    num_actions = 4
    feature_dim = 1

    # ここではLSTMモデルを例とする
    model = LSTM_ActorCriticModel(
        num_actions=num_actions, feature_dim=feature_dim, lstm_units=64)
    print(f"[INFO] Loading model weights from: {weights_path}")
    model.load_weights(weights_path)
    print("[INFO] Model weights loaded.")

    # シミュレーション実行(1エピソード)
    state = env.reset()
    done = False

    total_reward = 0.0
    step_log = []
    step_count = 0

    while not done:
        price_window, position, pl = state

        price_window_input = price_window[None, ...]   # (1, window_size, 1)
        position_input = np.array([position], dtype=np.int32)
        pl_input = np.array([pl], dtype=np.float32)

        policy, _ = model(
            [price_window_input, position_input, pl_input], training=False)
        policy = policy.numpy()[0]
        action = np.argmax(policy)  # 最も確率の高い行動を選択

        next_state, reward, done = env.step(action)
        total_reward += reward

        step_log.append((step_count, action, reward, position, pl))
        step_count += 1

        if not done:
            state = next_state

    print(
        f"[INFO] Simulation complete. Final reward (profit): {total_reward:.4f}")

    # ログ保存
    out_dir = os.path.join("results", "simulations", f"{pair}_AI_logs")
    os.makedirs(out_dir, exist_ok=True)
    log_filename = f"log_ai_ws{window_size}.csv"
    log_path = os.path.join(out_dir, log_filename)

    with open(log_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["step_index", "action", "reward",
                        "position", "unrealized_pl"])
        for row in step_log:
            writer.writerow(row)

    print(f"[INFO] Step-by-step log saved to: {log_path}")


if __name__ == "__main__":
    main()
