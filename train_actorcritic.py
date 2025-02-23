from actor_critic_modules.reward_function import fill_zeros_with_partial_fractions, compute_position_returns, compute_returns
from actor_critic_modules.trainer import RLTrainerVectorized
from actor_critic_modules.models import LSTM_ActorCriticModel
from actor_critic_modules.env import TradingEnvPrediction, VectorizedTradingEnv
from modules.models import build_lstm_cnn_attention_indicator_model
from modules.data_loader import load_csv_data
from modules.dataset import create_dataset
from datetime import datetime
import tensorflow as tf
import os


def split_array(arr, num_splits):
    """
    配列 arr を num_splits 個に等分割して返す。
    最終セグメントは余りがあればすべて含む。
    """
    L = len(arr)
    seg_size = L // num_splits
    segments = []
    for i in range(num_splits):
        start = i * seg_size
        end = (i+1)*seg_size if i < num_splits-1 else L
        segments.append(arr[start:end])
    return segments


def main(pair, k, future_k):
    csv_file_name = f"sample_{pair}_1m.csv"
    csv_file = os.path.join("data", csv_file_name)
    print(f"[INFO] Loading CSV data for {pair} from: {csv_file}")

    timestamps, prices = load_csv_data(csv_file)

    data_x, data_y = create_dataset(
        prices, k, future_k,
        balance_class=False,
        split=False
    )

    prices_raw, _ = create_dataset(
        prices, k, future_k, down_sampling=1,
        balance_class=False,
        split=False,
        normalize=False
    )

    # 予測モデル(例: LSTM_CNN_ATTENTION_INDICATOR)で確率を予測
    model_class_name = "LSTM_CNN_INDICATOR"
    input_dim = data_x.shape[1]
    indicator_model = build_lstm_cnn_attention_indicator_model(input_dim)
    indicator_model.load_weights(
        'results/BTCJPY/LSTM_CNN_INDICATOR_m1_k240_f30_20250217-145133/best_model.weights.h5')

    print("[INFO] Predicting with the indicator model...")
    prediction = indicator_model.predict(
        data_x, batch_size=10000)   # shape: (N, 2)

    # 実際の価格列 (N,) に整形
    prices_raw = prices_raw[:, -1]  # shape: (N,)
    # ========== ベクトル化用に複数環境を作る ==========

    # 環境数(並列数)を適当に設定
    num_envs = 1000

    # prices_raw, prediction を num_envs個に分割
    prices_segments = split_array(prices_raw, num_envs)
    pred_segments = split_array(prediction, num_envs)

    # 各セグメントで TradingEnvPrediction を構築
    window_size = 30
    envs = []
    for i in range(num_envs):
        seg_prices = prices_segments[i]
        seg_preds = pred_segments[i]
        # 短すぎるセグメントは学習にならないので対策が必要な場合も
        if len(seg_prices) < window_size + 1:
            # 適当にスキップ or ダミーを入れるなど
            continue
        env = TradingEnvPrediction(prices=seg_prices,
                                   predictions=seg_preds,
                                   window_size=window_size)
        envs.append(env)

    # VectorizedTradingEnv でまとめる
    vector_env = VectorizedTradingEnv(envs)

    # モデル (Actor-Critic)
    num_actions = 3  # 0: Hold, 1: Long, 2: Short
    feature_dim = 1
    actor_critic_model = LSTM_ActorCriticModel(num_actions=num_actions,
                                               feature_dim=feature_dim,
                                               lstm_units=32)

    # オプティマイザ
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # ベクトル化環境用トレーナ
    trainer = RLTrainerVectorized(
        model=actor_critic_model,
        optimizer=optimizer,
        vector_env=vector_env,
        num_actions=num_actions
    )

    num_episodes = 50
    mini_batch_size = 10000

    print("[TRAIN] Start training (vectorized-env) ...")

    for episode in range(1, num_episodes + 1):
        # run_episode() は (states_list, actions_list, rewards_list) を返す
        (states_list, actions_list, rewards_list) = trainer.run_episode()

        # 例えば 報酬配列を加工する場合:
        # rewards_list = compute_position_returns(rewards_list, actions_list)
        # rewards_list = compute_returns(rewards_list)
        rewards_list = fill_zeros_with_partial_fractions(rewards_list)

        # 親クラス(RLTrainerSingleEnv)の train_on_episode をそのまま使える
        loss, sum_return, mean_advantage = trainer.train_on_episode(
            states_list, actions_list, rewards_list,
            mini_batch_size=mini_batch_size
        )

        print(
            f"Episode {episode}: Loss={loss:.4f}, SumReturn={sum_return:.4f}, MeanAdv={mean_advantage:.4f}"
        )

    print("[TRAIN] Training finished.")

    # モデル重みを保存
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "results", pair, f"ActorCritic_ws{window_size}_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "best_model.weights.h5")
    actor_critic_model.save_weights(save_path)
    print(f"[TRAIN] Model weights saved to: {save_path}")


if __name__ == "__main__":
    pair = 'BTCJPY'
    k = 240
    future_k = 30

    main(pair, k, future_k)
