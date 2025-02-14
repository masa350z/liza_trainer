"""
強化学習トレーナーモジュール(ベクトル化環境版)

VectorizedTradingEnv を利用して、複数環境のエピソードを並列に処理し、
経験を集約してネットワークのパラメータを更新します。
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
        states = self.vector_env.reset()  # shape: (batch_size, window_size, feature_dim)
        done = np.array([False] * self.batch_size)
        states_list = []
        actions_list = []
        rewards_list = []
        # 最大ステップ数は、各環境でのヒストリカルデータ長 - window_size (全環境で同じ前提)
        total_steps = len(
            self.vector_env.envs[0].prices) - self.vector_env.envs[0].window_size
        pbar = tqdm(total=total_steps, desc="Episode Steps", unit="step")
        step_counter = 0

        while not np.all(done):
            states_list.append(states)
            # バッチ入力でモデル推論: (batch_size, window_size, feature_dim)
            # policy: (batch_size, num_actions)
            policy, _ = self.model(states, training=False)
            policy_np = policy.numpy()  # (batch_size, num_actions)
            actions = np.array(
                [np.random.choice(self.num_actions, p=p) for p in policy_np])
            actions_list.append(actions)
            next_states, rewards, dones = self.vector_env.step(actions)
            rewards_list.append(rewards)
            states = next_states
            done = dones
            step_counter += 1
            pbar.update(1)
        pbar.close()

        return states_list, actions_list, rewards_list

    def compute_returns(self, rewards_list):
        """
        各環境ごとの割引累積報酬(リターン)を計算する。

        Args:
            rewards_list: リスト(長さ T)の各要素は (batch_size,) の報酬配列

        Returns:
            returns: numpy array (T, batch_size)
        """
        rewards_array = np.stack(
            rewards_list, axis=0)  # shape: (T, batch_size)
        T = rewards_array.shape[0]
        returns = np.zeros_like(rewards_array)
        for b in range(self.batch_size):
            G = 0.0
            for t in reversed(range(T)):
                G = rewards_array[t, b] + self.gamma * G
                returns[t, b] = G
        return returns

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
        returns = self.compute_returns(rewards_list)  # shape: (T, batch_size)
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
        return average_loss, np.mean(returns_flat), np.mean(advantage)

    def train(self, num_episodes, print_interval=10, mini_batch_size=1024):
        for episode in range(1, num_episodes + 1):
            states_list, actions_list, rewards_list = self.run_episode()
            loss, mean_return, mean_advantage = self.train_on_episode(
                states_list, actions_list, rewards_list,
                mini_batch_size=mini_batch_size)
            if episode % print_interval == 0:
                print(
                    f"Episode {episode}: Loss: {loss:.4f}, Mean Return: {mean_return:.4f}, Mean Advantage: {mean_advantage:.4f}")
