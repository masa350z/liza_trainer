"""
モデル定義モジュール

ここでは、LSTMを用いたActor-Criticモデル(ポリシーヘッドとバリューヘッドを持つ)を定義します。
"""

import tensorflow as tf
from tensorflow.keras import layers  # type: ignore


# レイヤー基底から tf.keras.Model に変更するなど
class Affine_ActorCriticModel(tf.keras.Model):
    def __init__(self, num_actions, feature_dim):
        super(Affine_ActorCriticModel, self).__init__()
        # 価格系列用
        self.affine1 = layers.Dense(64, activation="relu")
        self.affine2 = layers.Dense(64, activation="relu")

        # ポジション用 (position = -1, 0, 1 を想定 => input_dim=3 の embedding)
        self.position_embedding = layers.Embedding(input_dim=3, output_dim=8)
        self.position_affine = layers.Dense(8, activation="relu")

        # 評価損益用
        self.pl_affine = layers.Dense(8, activation="relu")

        # 結合後の共通層
        self.common = layers.Dense(64, activation="relu")

        # Actor / Critic の出力ヘッド
        self.policy = layers.Dense(
            num_actions, activation="softmax", name="policy")
        self.value = layers.Dense(feature_dim, name="value")

    def call(self, inputs, training=False):
        """
        inputs はタプル or リストで [price_data, position, pl] が入る想定

        price_data.shape: (batch_size, window_size, 1)  (Affineの場合は flatten するなど)
        position.shape:   (batch_size,)
        pl.shape:         (batch_size,)
        """
        price_data, position, pl = inputs

        # 1. 価格系列 (Affineの場合は window_size*feature_dim をflattenしてDenseへ)
        x = tf.reshape(price_data, [tf.shape(price_data)[0], -1])  # flatten
        x = self.affine1(x)
        x = self.affine2(x)

        # 2. ポジション (position = -1,0,1 -> embedding用に 0,1,2 へシフトする場合)
        pos_index = position + 1  # -1->0, 0->1, 1->2
        pos_emb = self.position_embedding(pos_index)
        pos_out = self.position_affine(pos_emb)

        # 3. 評価損益
        pl_out = tf.expand_dims(pl, axis=-1)  # shape: (batch_size, 1)
        pl_out = self.pl_affine(pl_out)

        # 4. 結合して common へ
        concat_x = tf.concat([x, pos_out, pl_out], axis=1)
        common = self.common(concat_x)

        # 5. ポリシーと価値
        policy = self.policy(common)
        value = self.value(common)  # shape: (batch_size, 1)

        return policy, value


class LSTM_ActorCriticModel(tf.keras.Model):
    def __init__(self, num_actions, feature_dim, lstm_units=32):
        super(LSTM_ActorCriticModel, self).__init__()
        # 価格系列用
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units)

        # ポジション用
        self.position_embedding = layers.Embedding(input_dim=3, output_dim=8)
        self.position_affine = layers.Dense(8, activation="relu")

        # 評価損益用
        self.pl_affine = layers.Dense(8, activation="relu")

        self.common01 = layers.Dense(64, activation="relu")
        self.common02 = layers.Dense(32, activation="relu")

        self.policy = layers.Dense(
            num_actions, activation="softmax", name="policy")

        self.value = layers.Dense(feature_dim, name="value")

    def call(self, inputs, training=False):
        # inputs: [price_data, position, pl]
        price_data, position, pl = inputs

        # 1. 価格系列 (LSTM)
        x = self.lstm1(price_data, training=training)
        x = self.lstm2(x, training=training)

        # 2. ポジション
        pos_index = position + 1  # -1->0, 0->1, 1->2
        pos_emb = self.position_embedding(pos_index)
        pos_out = self.position_affine(pos_emb)

        # 3. 評価損益
        pl_out = tf.expand_dims(pl, axis=-1)  # (batch_size, 1)
        pl_out = self.pl_affine(pl_out)

        # 4. 結合 => common
        concat_x = tf.concat([x, pos_out, pl_out], axis=1)
        common = self.common01(concat_x)
        common = self.common02(common)

        # 5. ポリシーとバリュー
        policy = self.policy(common)  # (batch_size, num_actions)
        value = self.value(common)    # (batch_size, 1)

        return policy, value
