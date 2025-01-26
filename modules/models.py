# modules/models.py
"""モデル定義モジュール

   * ニューラルネットワークの構造を定義する関数をまとめる。
   * 例: 全結合モデル, Transformerなど
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential


def build_simple_affine_model(input_dim):
    """単純な全結合モデルを構築して返す

    Args:
        input_dim (int): 入力次元 (kに相当)

    Returns:
        tf.keras.Model: コンパイル済みのKerasモデル (2クラス分類)
    """
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    return model


class TransformerBlock(layers.Layer):
    """簡易的なTransformerエンコーダブロック"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(input_dim, embed_dim=32, num_heads=2, ff_dim=64):
    """Transformerブロックを使ったモデルの例

    注意:
        input_dimのスカラーに対してembeddingをかけるため
        (k, 1) に reshape してEmbedding層に相当するDenseを挟むなど
        実装の工夫が必要

    Args:
        input_dim (int):
        embed_dim (int): 埋め込み次元
        num_heads (int):
        ff_dim (int): FFN内部の次元

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=(input_dim,))
    # [batch, k] -> [batch, k, 1]
    x = layers.Reshape((input_dim, 1))(inputs)
    # 1次元を embed_dim 次元に埋め込む
    x = layers.Dense(embed_dim)(x)

    # TransformerBlock
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # seq方向をpooling
    x = layers.GlobalAveragePooling1D()(x)
    # 最後に2クラスのソフトマックス
    x = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model
