"""
強化学習トレーナーモジュール(単一環境版)

TradingEnv を利用して、1つの環境をエピソード単位で実行し、
得られた状態・行動・報酬を使ってActor-Criticモデルを学習します。
"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf


class RLTrainerSingleEnv:
    """単一環境用のシンプルなActor-Criticトレーナー"""

    def __init__(self, model, optimizer, env, num_actions=4):
        """
        Args:
            model (tf.keras.Model): Actor-Criticモデル
            optimizer (tf.keras.optimizers.Optimizer): オプティマイザ
            env (TradingEnv): 単一のトレーディング環境
            gamma (float): 割引率
            num_actions (int): 行動数(0:Hold,1:Long,2:Short,3:Exit など)
        """
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.num_actions = num_actions

    def predict_policy_and_value(self, price, pos, pl):
        """
        方策と状態価値を推論する。

        Args:
            price (np.ndarray): shape (window_size, 1)
                直近の価格ウィンドウ。
            pos (int): スカラー。現在のポジション状態(-1, 0, 1など)。
            pl (float): スカラー。現在の含み損益。

        Returns:
            tuple of tf.Tensor:
                (policy, values) のタプル。
                policy は shape (1, num_actions)、
                values は shape (1, 1)。
        """

        # price は (window_size, 1) なので、先頭に軸を追加して
        # (1, window_size, 1) の形に変換
        # shape => (1, window_size, 1)
        price_window_input = price[None, ...]

        # pos と pl はスカラーなので、[pos], [pl] として (1,) の形に
        position_input = np.array([pos], dtype=np.int32)  # shape => (1,)
        pl_input = np.array([pl], dtype=np.float32)       # shape => (1,)

        # shape => policy: (1, num_actions), values: (1,1) を返す
        policy, values = self.model(
            [price_window_input, position_input, pl_input], training=False)

        return policy, values

    def compute_actor_critic_loss(self, price, pos, pl, actions, rewards):
        """Actor-Criticの損失を計算する。

        Args:
            price (np.ndarray): shape (batch_size, window_size, 1)
            pos (np.ndarray): shape (batch_size,)
            pl (np.ndarray): shape (batch_size,)
            actions (np.ndarray): shape (batch_size,)
            returns (np.ndarray): shape (batch_size,)

        Returns:
            tuple of tf.Tensor:
                (loss, advantage) のタプル。
                advantage は (batch_size,) 形状。
        """
        with tf.GradientTape() as tape:
            policy, values = self.model([price, pos, pl], training=True)
            # Criticの値を squeeze して shape (batch_size,)
            values = tf.squeeze(values, axis=1)

            # 行動の one-hot
            action_one_hot = tf.one_hot(actions, depth=self.num_actions)
            # 各行動が選択される確率
            prob_actions = tf.reduce_sum(policy * action_one_hot, axis=1)
            # log π(a|s)
            log_prob = tf.math.log(prob_actions + 1e-8)

            advantage = rewards - values  # shape (batch_size,)

            actor_loss = -tf.reduce_mean(log_prob * advantage)
            critic_loss = tf.reduce_mean(tf.square(advantage))
            # エントロピー
            entropy_loss = -tf.reduce_mean(
                tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=1))

            # 損失合計
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        return loss, advantage, tape

    def run_episode(self):
        """
        1エピソードを実行し、各ステップの (state, action, reward) を収集

        Returns:
            states_list (list): 各ステップの状態 (price_window, position, pl)
            actions_list (list): 各ステップの行動 (int)
            rewards_list (list): 各ステップの報酬 (float)
        """
        states_list, actions_list, rewards_list = [], [], []

        state = self.env.reset()
        done = False

        total_steps = len(self.env.prices) - self.env.window_size
        pbar = tqdm(total=total_steps, desc="Episode Steps", unit="step")

        while not done:
            # state は (price_window, position, pl)
            price_window, position, pl = state
            policy, _ = self.predict_policy_and_value(
                price_window, position, pl)
            policy = policy.numpy()[0]  # (num_actions,)

            # 行動をサンプリング (確率的方策)
            action = np.random.choice(self.num_actions, p=policy)

            # 環境を1ステップ進める
            next_state, reward, done = self.env.step(action)

            # 結果を保存
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)

            if not done:
                state = next_state

            pbar.update(1)
        pbar.close()

        return states_list, actions_list, rewards_list

    def train_on_episode(self, states_list, actions_list, rewards_list, mini_batch_size=32):
        """
        1エピソード分の (state, action, reward) から学習を行う
        """
        # モデル学習用に配列へ変換
        price_data_list = []
        pos_list = []
        pl_list = []

        for (price_window, position, pl) in states_list:
            price_data_list.append(price_window)
            pos_list.append(position)
            pl_list.append(pl)

        price_data_array = np.array(price_data_list, dtype=np.float32)
        pos_array = np.array(pos_list, dtype=np.int32)
        pl_array = np.array(pl_list, dtype=np.float32)
        actions_array = np.array(actions_list, dtype=np.int32)
        rewards_array = np.array(rewards_list, dtype=np.float32)

        # バッチ化してミニバッチ単位で最適化
        total_samples = len(rewards_array)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        total_loss = 0.0
        num_batches = 0

        for start_idx in range(0, total_samples, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            batch_idx = indices[start_idx:end_idx]

            batch_price = price_data_array[batch_idx]
            batch_pos = pos_array[batch_idx]
            batch_pl = pl_array[batch_idx]
            batch_rewards = rewards_array[batch_idx]
            batch_actions = actions_array[batch_idx]

            loss, advantage, tape = self.compute_actor_critic_loss(price=batch_price,
                                                                   pos=batch_pos,
                                                                   pl=batch_pl,
                                                                   actions=batch_actions,
                                                                   rewards=batch_rewards)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            total_loss += loss.numpy()
            num_batches += 1

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        sum_return = rewards_array.sum()
        mean_advantage = advantage.numpy().mean()

        return average_loss, sum_return, mean_advantage


class RLTrainerVectorized(RLTrainerSingleEnv):
    """
    ベクトル化環境用のActor-Criticトレーナー。
    RLTrainerSingleEnvを継承し、run_episodeのみをオーバーライドして
    戻り値の形式を (states_list, actions_list, rewards_list) に合わせる。
    """

    def __init__(self, model, optimizer, vector_env, num_actions=3):
        """
        Args:
            model: Actor-Criticモデル
            optimizer: TFオプティマイザ
            vector_env: VectorizedTradingEnv
            num_actions: 行動数
        """
        # 親クラスの __init__ を呼ぶ (env は None など適当に)
        super().__init__(model, optimizer, env=None, num_actions=num_actions)

        # 自身が保持するベクトル化環境
        self.vector_env = vector_env
        self.batch_size = vector_env.num_envs

    def run_episode(self):
        """
        複数環境を並列に進める。
        収集した (T, batch_size, ...) のデータを flatten して
        親クラスと同じ形 (リスト of (price_window, pos, pl)) に変換して返す。
        """
        states_price = []
        states_pos = []
        states_pl = []
        actions_list = []
        rewards_list = []

        # reset
        state = self.vector_env.reset()  # => (price_batch, pos_batch, pl_batch)
        done = np.array([False]*self.batch_size)

        # 参考:（先頭envの総ステップでループ）
        total_steps = len(
            self.vector_env.envs[0].prices) - self.vector_env.envs[0].window_size
        pbar = tqdm(total=total_steps, desc="Episode Steps", unit="step")

        while not np.all(done):
            # shape: (batch_size, ...)
            price_batch, pos_batch, pl_batch = state

            states_price.append(price_batch)  # (batch_size, window_size, 1)
            states_pos.append(pos_batch)      # (batch_size,)
            states_pl.append(pl_batch)        # (batch_size,)

            # モデル推論
            policy, _ = self.model(
                [price_batch, pos_batch, pl_batch], training=False)
            policy_np = policy.numpy()  # shape (batch_size, num_actions)

            # 行動をサンプリング
            actions = np.array([
                np.random.choice(self.num_actions, p=policy_np[i])
                for i in range(self.batch_size)
            ], dtype=np.int32)
            actions_list.append(actions)  # shape (batch_size,)

            next_state, rewards, dones = self.vector_env.step(actions)
            rewards_list.append(rewards)  # shape (batch_size,)

            state = next_state
            done = dones
            pbar.update(1)
        pbar.close()

        # shape 整形
        # (T, batch_size, window_size, 1)
        price_array = np.stack(states_price, axis=0)
        pos_array = np.stack(states_pos, axis=0)        # (T, batch_size)
        pl_array = np.stack(states_pl, axis=0)          # (T, batch_size)
        actions_array = np.stack(actions_list, axis=0)  # (T, batch_size)
        rewards_array = np.stack(rewards_list, axis=0)  # (T, batch_size)

        T = price_array.shape[0]
        bs = price_array.shape[1]

        # Flatten => (T*bs, window_size, 1)
        price_array = price_array.reshape(
            T*bs, price_array.shape[2], price_array.shape[3])
        pos_array = pos_array.reshape(T*bs)
        pl_array = pl_array.reshape(T*bs)
        actions_array = actions_array.reshape(T*bs)
        rewards_array = rewards_array.reshape(T*bs)

        # 親クラスの train_on_episode と同じ形式に変換
        # states_list: List of (price_window, pos, pl)
        # actions_list: List of action
        # rewards_list: List of reward
        # => 長さは (T*bs)

        states_list = []
        for i in range(T*bs):
            # price_array[i]: shape (window_size, 1)
            # pos_array[i]:   int
            # pl_array[i]:    float
            states_list.append((price_array[i], pos_array[i], pl_array[i]))

        actions_list_flat = actions_array.tolist()
        rewards_list_flat = rewards_array.tolist()

        return states_list, actions_list_flat, rewards_list_flat
