"""
liza_trainer の学習エントリーポイント

データを読み込み、TradingEnv環境を構築し、Actor-Criticモデルを学習後、最良モデルの重みを保存します。
"""

import os
import datetime
import tensorflow as tf

from modules.data_loader import load_csv_data
from modules.env import TradingEnv
from modules.models import build_actor_critic_model
from modules.trainer import RLTrainer

# GPUメモリの必要分だけ確保
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main(pair, window_size, num_episodes, histrical_length=''):
    """
    Args:
        pair (str): "EURUSD" または "USDJPY"
        window_size (int): 状態として使用する過去の価格数
        num_episodes (int): 学習エピソード数
    """
    csv_file = os.path.join("data", f"{pair}_1m{histrical_length}.csv")
    print(f"[INFO] Loading CSV data for {pair} from: {csv_file}")
    _, prices = load_csv_data(csv_file)
    if len(prices) < window_size + 1:
        raise ValueError("価格データがwindow_sizeよりも短いため、環境を構築できません。")

    # 環境の構築
    env = TradingEnv(prices, window_size)
    feature_dim = 1  # 価格のみの場合

    # モデル構築
    num_actions = 4  # 0: Hold, 1: Enter Long, 2: Enter Short, 3: Exit
    model = build_actor_critic_model(
        time_steps=window_size, feature_dim=feature_dim, num_actions=num_actions, lstm_units=64)

    # オプティマイザ設定
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # トレーナーの構築
    trainer = RLTrainer(model=model, optimizer=optimizer,
                        env=env, num_actions=num_actions, gamma=0.99)

    print("[INFO] Starting training...")
    trainer.train(num_episodes=num_episodes, print_interval=10)
    print("[INFO] Training complete.")

    # 保存用ディレクトリ作成
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "results/models", pair, f"ActorCritic_ws{window_size}_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "best_model.weights.h5")
    model.save_weights(weights_path)
    print(f"[INFO] Model weights saved to: {weights_path}")


if __name__ == "__main__":
    # 例としてEURUSD、ウィンドウサイズ30、エピソード数1000で学習
    # for pair in ['EURUSD', 'USDJPY']:
    for pair in ['USDJPY']:
        main(pair=pair, window_size=30, num_episodes=10,
             histrical_length='_len10000')
