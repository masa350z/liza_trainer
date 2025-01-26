# modules/trainer.py
"""学習ロジックを扱うモジュール

    概要:
        - train_data, valid_data, test_data を受け取り、
          バリデーションを見ながら学習し、改善しなければ重みを一部ランダムに再初期化
        - 指定回数まで繰り返し、最後にテストデータで評価し、一番良かったモデルを保存

    主なクラス:
        Trainer:
            * run(): 学習を開始する
            * save_best_weights(): ベストモデルの重みを保存する
"""

import numpy as np
import tensorflow as tf
from tqdm import trange


class GradualDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学習率を徐々に減衰させるスケジューラー
       PolyDecayを利用
    """

    def __init__(self, initial_lr, final_lr, decay_steps):
        super().__init__()
        self.decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=final_lr,
            power=1.0
        )

    def __call__(self, step):
        return self.decay_fn(step)


class Trainer:
    """学習を管理するクラス

    機能:
        - train_data / valid_data / test_data に対して学習
        - バリデーション損失が改善しなくなったら重みを部分的に再初期化して局所解を脱出
        - これを num_repeats 回繰り返して最良モデルを記録
        - 最後にsave_best_weights() でモデルを保存
    """

    def __init__(self,
                 model,
                 train_data,
                 valid_data,
                 test_data,
                 learning_rate_initial=1e-4,
                 learning_rate_final=1e-5,
                 switch_epoch=50,
                 random_init_ratio=1e-4,
                 max_epochs=1000,
                 patience=10,
                 num_repeats=5,
                 batch_size=1024):
        """
        Args:
            model (tf.keras.Model): 学習対象のモデル
            train_data (tuple): (train_x, train_y)
            valid_data (tuple): (valid_x, valid_y)
            test_data (tuple): (test_x, test_y)
            learning_rate_initial (float): 最初の学習率
            learning_rate_final (float): 最終的な学習率
            switch_epoch (int): 学習率の減衰をかけるステップ数
            random_init_ratio (float): 
                バリデーションが改善しなくなった時に、一部重みをランダム化する際の割合
            max_epochs (int): エポック数の上限
            patience (int): validationロスが改善しないまま何エポック進んだら再初期化を行うか
            num_repeats (int): 
                「(学習→再初期化)を繰り返してテスト評価し、最良を保存」する繰り返し回数
        """
        self.model = model

        self.train_x, self.train_y = train_data
        self.valid_x, self.valid_y = valid_data
        self.test_x, self.test_y = test_data

        self.learning_rate_schedule = GradualDecaySchedule(
            learning_rate_initial, learning_rate_final, switch_epoch
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule
        )

        self.random_init_ratio = random_init_ratio
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_repeats = num_repeats

        # ベストモデル情報
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_weights = None

        # テスト時のスコア
        self.best_test_loss = float("inf")
        self.best_test_acc = 0.0

        self.batch_size = batch_size

    def run(self):
        """メインの学習ループを実行し、ベストモデルを確定させる"""
        for repeat_i in range(self.num_repeats):
            print(
                f"\n[INFO] ===== Start Repeat {repeat_i+1}/{self.num_repeats} =====")

            # まず最初に重みを初期化して開始(ベースライン)
            self._init_model_weights()
            # 学習を行う
            self._train_loop()

            # 学習が終わったら、テスト評価を行う
            test_loss, test_acc = self._evaluate(self.test_x, self.test_y)
            print(
                f"[INFO] Test result -> loss={test_loss:.6f}, acc={test_acc:.6f}")

            # ベストモデルを更新できれば更新
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.best_test_acc = test_acc
                self.best_weights = self.model.get_weights()
                self.best_val_loss = self.temp_val_loss  # 参考までに保管
                self.best_val_acc = self.temp_val_acc
                print("[INFO] Updated best model based on test loss.")
            else:
                print("[INFO] No improvement in test loss. Keep searching...")

        print("[INFO] Done all repeats.")
        print(
            f"[INFO] Best test loss={self.best_test_loss:.6f}, test acc={self.best_test_acc:.6f}")

    def save_best_weights(self, filepath):
        """ベストモデルの重みを指定ファイルに保存する"""
        if self.best_weights is None:
            print("[WARN] No best weights found. Not saving anything.")
            return
        # 現モデルにbest_weightsをセットしてセーブ
        self.model.set_weights(self.best_weights)
        self.model.save_weights(filepath)
        print(f"[INFO] Saved best weights to: {filepath}")

    def _train_loop(self):
        """1回の学習ループ。バリデーションを見ながら学習し、改善しなければ再初期化など行う"""
        # 学習用 一時変数
        self.temp_val_loss = float("inf")
        self.temp_val_acc = 0.0
        patience_counter = 0

        num_samples = len(self.train_x)
        steps_per_epoch = max(1, num_samples // self.batch_size)

        # epochループ
        for epoch_i in range(self.max_epochs):
            print(f"[INFO] Epoch {epoch_i+1}/{self.max_epochs}")

            # ミニバッチ学習
            rand_idx = np.random.permutation(num_samples)
            for step_i in range(steps_per_epoch):
                batch_idx = rand_idx[step_i *
                                     self.batch_size: (step_i+1)*self.batch_size]
                bx = self.train_x[batch_idx]
                by = self.train_y[batch_idx]

                self._train_step(bx, by)

            # 1epoch終了 -> バリデーションチェック
            val_loss, val_acc = self._evaluate(self.valid_x, self.valid_y)
            print(f"   [VALID] loss={val_loss:.6f}, acc={val_acc:.6f}")

            if val_loss < self.temp_val_loss:
                # 改善した
                self.temp_val_loss = val_loss
                self.temp_val_acc = val_acc
                patience_counter = 0
            else:
                # 改善しない
                patience_counter += 1
                if patience_counter >= self.patience:
                    # 部分的に重みをランダム初期化し、局所解からの脱出を図る
                    print(
                        "[INFO] Validation not improving. Re-initializing partial weights.")
                    self._partial_random_init()
                    # カウンターリセット
                    patience_counter = 0

    def _train_step(self, bx, by):
        with tf.GradientTape() as tape:
            pred = self.model(bx, training=True)
            loss_val = tf.keras.losses.CategoricalCrossentropy()(by, pred)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def _evaluate(self, x, y):
        """損失と正解率を返す"""
        preds = self.model(x, training=False)
        loss_val = tf.keras.losses.CategoricalCrossentropy()(y, preds)
        # acc計算
        correct = tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1))
        acc_val = tf.reduce_mean(tf.cast(correct, tf.float32))
        return float(loss_val), float(acc_val)

    def _init_model_weights(self):
        """モデルの全重みを再初期化"""
        dummy_input = tf.zeros((1, self.train_x.shape[1]))
        self.model(dummy_input, training=False)  # ビルド(重み生成)
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.assign(layer.kernel_initializer(
                    layer.kernel.shape, layer.kernel.dtype))
            if hasattr(layer, "bias_initializer"):
                layer.bias.assign(layer.bias_initializer(
                    layer.bias.shape, layer.bias.dtype))

    def _partial_random_init(self):
        """モデル重みの一部をランダムに初期化する(局所解回避)"""
        weights = self.model.get_weights()
        for i, w in enumerate(weights):
            # 全結合層など (2D) のみ対象とするなど工夫可能
            if len(w.shape) == 2:
                # random_init_ratio の確率で置き換え
                mask = np.random.binomial(
                    1, self.random_init_ratio, size=w.shape)
                rand_w = np.random.randn(*w.shape).astype(w.dtype) * 0.05
                new_w = w * (1 - mask) + rand_w * mask
                weights[i] = new_w
        self.model.set_weights(weights)
