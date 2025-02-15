"""
単一環境版の学習エントリーポイント

- CSVデータを読み込み
- 1つのTradingEnvを生成
- LSTM_ActorCriticModel などを生成
- RLTrainerSingleEnv を使って学習
- 学習したモデルの重みを保存
"""

import os
import datetime
import numpy as np
import tensorflow as tf

from modules.data_loader import load_csv_data
from modules.env import TradingEnv
from modules.models import LSTM_ActorCriticModel
from modules.trainer import RLTrainerSingleEnv


# GPUメモリの必要分だけ確保
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main(pair="USDJPY", window_size=30, num_episodes=10,
         mini_batch_size=32, historical_length=""):
    """
    Args:
        pair (str): "EURUSD" or "USDJPY"
        window_size (int): 状態として使用する過去の価格数
        num_episodes (int): 学習エピソード数
        mini_batch_size (int): ミニバッチサイズ
        historical_length (str): CSVファイル名の付加文字列 (例: '_len10000')
    """
    csv_file = os.path.join("data", f"{pair}_1m{historical_length}.csv")
    print(f"[TRAIN] Loading CSV data from: {csv_file}")
    _, prices = load_csv_data(csv_file, skip=100)
    if len(prices) < window_size + 1:
        raise ValueError("価格データがwindow_sizeよりも短いため、環境を構築できません。")

    # 単一環境を構築
    env = TradingEnv(prices, window_size)
    num_actions = 3  # 0:Hold, 1:Long, 2:Short
    feature_dim = 1

    # モデルを選択(LSTM or Affine)
    model = LSTM_ActorCriticModel(
        num_actions=num_actions, feature_dim=feature_dim, lstm_units=32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    trainer = RLTrainerSingleEnv(
        model, optimizer, env, gamma=0.99, num_actions=num_actions)

    print("[TRAIN] Start training (single-env) ...")
    trainer.train(num_episodes=num_episodes, print_interval=1,
                  mini_batch_size=mini_batch_size)
    print("[TRAIN] Training finished.")

    # モデル重みを保存
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "results", "models", pair, f"ActorCritic_ws{window_size}_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "best_model.weights.h5")
    model.save_weights(save_path)
    print(f"[TRAIN] Model weights saved to: {save_path}")


if __name__ == "__main__":
    # サンプル実行: USDJPY, window_size=30, episodes=10
    main(pair="USDJPY",
         window_size=30,
         num_episodes=100,
         mini_batch_size=1000,
         historical_length="_len100000")
