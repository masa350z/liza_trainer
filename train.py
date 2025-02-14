"""
liza_trainer の学習エントリーポイント (ベクトル化環境版)

指定した通貨ペアのCSVデータを読み込み、ヒストリカルデータを32分割して各環境に割り当て、
VectorizedTradingEnv を構築し、RLTrainerVectorized を用いて学習を行い、最良モデルの重みを保存します。
"""

import os
import datetime
import tensorflow as tf
import numpy as np

from modules.data_loader import load_csv_data
from modules.env import TradingEnv
from modules.env import VectorizedTradingEnv
from modules.models import build_actor_critic_model
from modules.trainer import RLTrainerVectorized

# GPUメモリの必要分だけ確保
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def split_prices(prices, num_splits):
    """
    ヒストリカルデータを num_splits 個に分割する。

    Args:
        prices (list or np.array): フルの価格データ
        num_splits (int): 分割数(環境数)

    Returns:
        list: 各環境に割り当てる価格データのセグメントのリスト
    """
    L = len(prices)
    segment_length = L // num_splits
    segments = []
    for i in range(num_splits):
        start = i * segment_length
        # 最後のセグメントは残りすべてを含む
        end = (i + 1) * segment_length if i < num_splits - 1 else L
        segments.append(prices[start:end])
    return segments


def main(pair, window_size, num_episodes, num_envs=32,
         mini_batch_size=1024, historical_length=''):
    """
    Args:
        pair (str): "EURUSD" または "USDJPY"
        window_size (int): 状態として使用する過去の価格数
        num_episodes (int): 学習エピソード数
        num_envs (int): 並列に実行する環境の数(＝バッチサイズ)
        historical_length (str): CSVファイル名に付加する文字列(例: '_len10000')
    """
    csv_file = os.path.join("data", f"{pair}_1m{historical_length}.csv")
    print(f"[TRAIN] Loading CSV data for {pair} from: {csv_file}")
    _, prices = load_csv_data(csv_file)
    if len(prices) < window_size + 1:
        raise ValueError("価格データがwindow_sizeよりも短いため、環境を構築できません。")

    # ヒストリカルデータを num_envs 個に分割して各環境に割り当てる
    segments = split_prices(prices, num_envs)
    envs = [TradingEnv(segment, window_size) for segment in segments]
    vector_env = VectorizedTradingEnv(envs)
    feature_dim = 1  # 価格のみの場合

    num_actions = 4  # 0: Hold, 1: Enter Long, 2: Enter Short, 3: Exit
    model = build_actor_critic_model(time_steps=window_size,
                                     feature_dim=feature_dim,
                                     num_actions=num_actions,
                                     lstm_units=64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    trainer = RLTrainerVectorized(model=model,
                                  optimizer=optimizer,
                                  vector_env=vector_env,
                                  num_actions=num_actions,
                                  gamma=0.99)
    print("[TRAIN] Starting vectorized training...")
    trainer.train(num_episodes=num_episodes, print_interval=1,
                  mini_batch_size=mini_batch_size)
    print("[TRAIN] Training complete.")

    # 保存用ディレクトリ作成
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("results/models", pair,
                              f"ActorCritic_ws{window_size}_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "best_model.weights.h5")
    model.save_weights(weights_path)
    print(f"[TRAIN] Model weights saved to: {weights_path}")


if __name__ == "__main__":
    # 例としてUSDJPY、ウィンドウサイズ30、エピソード数10、並列環境数32を用いる
    for pair in ['USDJPY']:
        main(pair=pair,
             window_size=30,
             num_episodes=100,
             num_envs=1000,
             mini_batch_size=10000,
             historical_length='_len1000000')
