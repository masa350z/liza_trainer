"""
強化学習トレーナーモジュール(ベクトル化環境版)

VectorizedTradingEnv を利用して、複数環境のエピソードを並列に処理し、
経験を集約してネットワークのパラメータを更新します。
"""

import numpy as np
import tensorflow as tf


def compute_position_returns(rewards_list, actions_list):
    # shape (T, batch_size)
    rewards_list = np.array(rewards_list)
    actions_list = np.array(actions_list)

    T, batch_size = rewards_list.shape

    # 各環境ごとに、取引開始時のインデックスとその取引で得られた報酬を記録するリストを用意
    index_list_batch = [[] for _ in range(batch_size)]
    # 結果を格納する配列（各環境の各ステップにおける取引の報酬を0で初期化）
    position_rewards = np.zeros_like(rewards_list)  # shape: (T, batch_size)

    # 各環境（バッチ）ごとにループ
    for b in range(batch_size):
        pos = 0          # 現在のポジション状態（0: 未ポジション, 1: ポジションあり）
        start_index = None  # 取引開始時のインデックス
        # 時系列に沿ってループ（Tステップ）
        for i in range(T):
            a = actions_list[i, b]   # 環境bでのiステップ目の行動
            profit = rewards_list[i, b]  # 環境bでのiステップ目の報酬

            if pos == 0:
                # 未ポジションの場合、新規エントリー（Enter Long or Enter Short）なら開始する
                if a == 1 or a == 2:
                    start_index = i
                    pos = 1
            else:
                # すでにポジションを持っている場合
                # 報酬が0でない＝決済が行われたと仮定
                if profit != 0:
                    index_list_batch[b].append((start_index, profit))
                    # 決済後、もしそのステップで再度エントリーがあれば、新たに取引を開始
                    if a == 1 or a == 2:
                        start_index = i
                        pos = 1
                    else:
                        pos = 0

    # 各環境ごとに、記録された取引開始時のインデックスに対して報酬を割り当てる
    for b in range(batch_size):
        for (start_idx, profit) in index_list_batch[b]:
            position_rewards[start_idx, b] = profit

    # 結果として、position_rewards は shape (T, batch_size) で、
    # 各環境で取引開始時にその取引の報酬（利益または損失）が記録され、他は0になっている。

    return position_rewards


class RLTrainerVectorized:
    def __init__(self, model, optimizer, vector_env, num_actions=4, gamma=0.99):
        """
        Args:
            model (tf.keras.Model): Actor-Criticモデル
            optimizer (tf.keras.optimizers.Optimizer): オプティマイザ
            vector_env (VectorizedTradingEnv): 複数環境をラップした環境
            num_actions (int): 行動数 (例: 4)
            gamma (float): 割引率
        """
        self.model = model
        self.optimizer = optimizer
        self.vector_env = vector_env
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = vector_env.num_envs

    def run_episode(self):
        """
        複数環境で1エピソードを同時に実行し、各ステップの状態、行動、報酬を収集する。

        Returns:
            states_list: 各ステップごとの状態のバッチのリスト (shape: (T, batch_size, window_size, feature_dim))
            actions_list: 各ステップごとの行動のバッチのリスト (shape: (T, batch_size))
            rewards_list: 各ステップごとの報酬のバッチのリスト (shape: (T, batch_size))
        """
        # 初期状態の取得
        states = self.vector_env.reset()  # shape: (batch_size, window_size, feature_dim)
        done = np.array([False] * self.batch_size)
        states_list = []
        actions_list = []
        rewards_list = []
        # policy_accumulator に各ステップでのポリシー出力を蓄積する
        policy_accumulator = []  # 各要素の形状: (batch_size, num_actions)

        # 各環境での最大ステップ数 (例: ヒストリカルデータの長さ - window_size)
        total_steps = len(
            self.vector_env.envs[0].prices) - self.vector_env.envs[0].window_size
        from tqdm import tqdm
        pbar = tqdm(total=total_steps, desc="Episode Steps", unit="step")
        step_counter = 0

        while not np.all(done):
            # 状態の蓄積
            states_list.append(states)
            # モデル推論: 入力 shape: (batch_size, window_size, feature_dim)
            # 出力 policy: (batch_size, num_actions)
            policy, _ = self.model(states, training=False)
            policy_np = policy.numpy()  # shape: (batch_size, num_actions)
            # policy_np を蓄積する（各環境の行動確率）
            policy_accumulator.append(policy_np)
            # 各環境で行動をサンプリングする
            actions = np.array(
                [np.random.choice(self.num_actions, p=p) for p in policy_np])
            actions_list.append(actions)
            # 各環境で step() を実行
            next_states, rewards, dones = self.vector_env.step(actions)
            rewards_list.append(rewards)
            # 次の状態、終了フラグを更新
            states = next_states
            done = dones
            step_counter += 1
            pbar.update(1)
        pbar.close()

        # エピソード全体のポリシー出力をフラットにする
        # 各ステップでの policy_accumulator の各要素は (batch_size, num_actions)
        # これらを連結すると、全サンプル数 = (T * batch_size) の行列となる
        # shape: (T * batch_size, num_actions)
        all_policy = np.concatenate(policy_accumulator, axis=0)
        # 全サンプルで各行動の平均確率を計算する
        overall_avg_policy = np.mean(
            all_policy, axis=0)  # shape: (num_actions,)
        print(
            f"[RL] Overall Average Action Probabilities: {overall_avg_policy}")

        print(f"[RL] Episode finished in {step_counter} steps.")
        return states_list, actions_list, rewards_list

    def train_on_episode(self, states_list, actions_list, rewards_list, mini_batch_size=1024):
        """
        ベクトル化環境で収集した1エピソード分のデータを用いて、モデルパラメータを更新する。

        Args:
            states_list: リスト (T個) 各要素: (batch_size, window_size, feature_dim)
            actions_list: リスト (T個) 各要素: (batch_size,)
            rewards_list: リスト (T個) 各要素: (batch_size,)
            mini_batch_size (int): ミニバッチ処理に用いるサンプル数

        Returns:
            average_loss, mean_return, mean_advantage
        """
        T = len(rewards_list)

        returns = compute_position_returns(
            rewards_list, actions_list)  # shape: (T, batch_size)
        # Flatten時間とバッチ軸: 総サンプル数 = T * batch_size
        # shape: (T*batch_size, window_size, feature_dim)
        states = np.concatenate(states_list, axis=0)
        # shape: (T*batch_size,)
        actions = np.concatenate(actions_list, axis=0)
        # shape: (T*batch_size,)
        returns_flat = returns.reshape(-1)

        total_samples = states.shape[0]
        total_loss = 0.0
        num_batches = 0

        # シャッフルしてミニバッチ処理
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        states = states[indices]
        actions = actions[indices]
        returns_flat = returns_flat[indices]

        for start in range(0, total_samples, mini_batch_size):
            end = start + mini_batch_size
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_returns = returns_flat[start:end]
            with tf.GradientTape() as tape:
                policy, values = self.model(batch_states, training=True)
                values = tf.squeeze(values, axis=1)  # shape: (batch, )

                action_one_hot = tf.one_hot(
                    batch_actions, depth=self.num_actions)

                prob_actions = tf.reduce_sum(policy * action_one_hot, axis=1)
                log_prob = tf.math.log(prob_actions + 1e-8)
                advantage = batch_returns - values

                actor_loss = -tf.reduce_mean(log_prob * advantage)
                critic_loss = tf.reduce_mean(tf.square(advantage))
                entropy_loss = - \
                    tf.reduce_mean(tf.reduce_sum(
                        policy * tf.math.log(policy + 1e-8), axis=1))

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            total_loss += loss.numpy()
            num_batches += 1

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return average_loss, np.sum(returns_flat), np.mean(advantage)

    def train(self, num_episodes, print_interval=10, mini_batch_size=1024):
        for episode in range(1, num_episodes + 1):
            states_list, actions_list, rewards_list = self.run_episode()

            loss, sum_return, mean_advantage = self.train_on_episode(
                states_list, actions_list, rewards_list,
                mini_batch_size=mini_batch_size)

            if episode % print_interval == 0:
                print(
                    f"Episode {episode}: Loss: {loss:.4f}, Sum Return: {sum_return:.24f}, Mean Advantage: {mean_advantage:.4f}")
