"""
モデル定義モジュール

ここでは、LSTMを用いたActor-Criticモデル(ポリシーヘッドとバリューヘッドを持つ)を定義します。
"""

from tensorflow.keras import layers  # type: ignore


class Affine_ActorCriticModel(layers.Layer):
    def __init__(self, num_actions, feature_dim):
        super(Affine_ActorCriticModel, self).__init__()
        self.affine1 = layers.Dense(64, activation="relu")
        self.affine2 = layers.Dense(64, activation="relu")

        self.common = layers.Dense(64, activation="relu")
        self.policy = layers.Dense(
            num_actions, activation="softmax", name="policy")
        self.value = layers.Dense(feature_dim, name="value")

    def call(self, inputs):
        x = self.affine1(inputs)
        x = self.affine2(x)
        common = self.common(x)
        policy = self.policy(common)
        value = self.value(common)

        return policy, value


class LSTM_ActorCriticModel(layers.Layer):
    def __init__(self, num_actions, feature_dim, lstm_units=64):
        super(LSTM_ActorCriticModel, self).__init__()
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units)

        self.common = layers.Dense(64, activation="relu")
        self.policy = layers.Dense(
            num_actions, activation="softmax", name="policy")
        self.value = layers.Dense(feature_dim, name="value")

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        common = self.common(x)
        policy = self.policy(common)
        value = self.value(common)

        return policy, value
