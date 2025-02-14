"""
モデル定義モジュール

ここでは、LSTMを用いたActor-Criticモデル(ポリシーヘッドとバリューヘッドを持つ)を定義します。
"""

from tensorflow.keras import layers, Model


def build_actor_critic_model(time_steps, feature_dim, num_actions=4, lstm_units=64):
    """
    Actor-Criticモデルを構築して返す

    Args:
        time_steps (int): 時系列の長さ(例: window_size)
        feature_dim (int): 各時刻の特徴次元(価格のみの場合は1)
        num_actions (int): 行動数(0: Hold, 1: Enter Long, 2: Enter Short, 3: Exit)
        lstm_units (int): LSTM層のユニット数

    Returns:
        tf.keras.Model: 入力に対して、[policy, value]を出力するモデル
    """
    inputs = layers.Input(shape=(time_steps, feature_dim))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(lstm_units)(x)
    common = layers.Dense(64, activation="relu")(x)

    # ポリシーヘッド(行動確率をsoftmaxで出力)
    policy = layers.Dense(
        num_actions, activation="softmax", name="policy")(common)
    # バリューヘッド(状態価値を1値で出力)
    value = layers.Dense(1, name="value")(common)

    model = Model(inputs=inputs, outputs=[policy, value])
    return model
