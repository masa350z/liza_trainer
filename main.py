# main.py
"""liza_trainer のエントリーポイント
学習ロジックを実行し、最終的にベストモデルを保存する。
"""
from modules.trainer import Trainer
from modules.models import build_simple_affine_model
from modules.dataset import create_dataset
from modules.data_loader import load_csv_data
import os
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    """学習のメインフローを実行する関数。

    1. CSV読み込み
    2. 特徴量/ラベル作成
    3. モデル定義
    4. 学習・評価のループ

    """

    # === 1. CSV読み込み ===
    csv_file = os.path.join("data", "sample_EURUSD_1m.csv")
    # CSVファイルには "timestamp, price" カラムがある前提
    print(f"[INFO] Loading CSV data from: {csv_file}")
    timestamps, prices = load_csv_data(csv_file)

    # === 2. 特徴量/ラベル作成 ===
    #   - k=90, pr_k=15 のように、直近k個の価格から「pr_k後の価格が上がるかどうか」を予測
    #   - ここでは例として k=30, pr_k=5
    k = 30
    future_k = 5
    print(f"[INFO] Creating dataset with k={k}, future_k={future_k}")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = create_dataset(
        prices, k, future_k,
        train_ratio=0.6, valid_ratio=0.2, down_sampling=10
    )

    # === 3. モデル定義 ===
    #   - モデルの構成は modules/models.py に定義
    #   - 今回は単純な全結合モデル build_simple_affine_model() を用いる
    input_dim = train_x.shape[1]
    print(f"[INFO] Building model with input_dim={input_dim}")
    model = build_simple_affine_model(input_dim)

    # === 4. 学習・評価のループ ===
    #   - Trainerクラスを用いて、バリデーションロスが改善しなくなったら
    #     重みランダム初期化などのロジックを入れる
    print("[INFO] Starting training process...")

    trainer = Trainer(
        model=model,
        train_data=(train_x, train_y),
        valid_data=(valid_x, valid_y),
        test_data=(test_x, test_y),
        learning_rate_initial=1e-4,
        learning_rate_final=1e-5,
        switch_epoch=250,         # 学習率を切り替えるステップ数
        random_init_ratio=1e-4,  # バリデーション損失が改善しなくなった場合の部分的ランダム初期化率
        max_epochs=1000,
        patience=10,             # validationが改善しなくなってから再初期化までの猶予回数
        num_repeats=5,            # 学習→バリデーション→（初期化）を繰り返す試行回数
        batch_size=40000
    )

    trainer.run()
    print("[INFO] Training finished.")

    # 結果としてモデルの重みは trainer.best_weights に記録されているので、
    # 任意のファイルに保存しておく
    best_model_path = "best_model_weights.h5"
    trainer.save_best_weights(best_model_path)

    print(f"[INFO] Best model weights saved to {best_model_path}")


if __name__ == "__main__":
    main()
