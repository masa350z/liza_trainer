"""
強化学習トレーナーモジュール

TradingEnv環境上で、Actor-Criticモデルを用いてエピソード単位で学習を行います。
"""

import numpy as np
import tensorflow as tf


class RLTrainer:
    def __init__(self, model, optimizer, env, num_actions=4, gamma=0.99):
        """
        Args:
            model (tf.keras.Model): Actor-Criticモデル
            optimizer (tf.keras.optimizers.Optimizer): オプティマイザ
            env (TradingEnv): トレーディング環境
            num_actions (int): 行動数
            gamma (float): 割引率
        """
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.num_actions = num_actions
        self.gamma = gamma

    def run_episode(self):
        """
        1エピソード分の状態、行動、報酬を収集する

        Returns:
            states (list): 各ステップの状態 (shape: (window_size, feature_dim))
            actions (list): 各ステップで取った行動（整数）
            rewards (list): 各ステップで得た報酬（float）
        """
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        done = False
        step_counter = 0

        # ヒストリカルデータの長さとウィンドウサイズから最大ステップ数を算出
        total_steps = len(self.env.prices) - self.env.window_size
        from tqdm import tqdm
        pbar = tqdm(total=total_steps, desc="Episode Steps", unit="step")

        while not done:
            # バッチ次元を追加してモデル推論
            # shape: (1, window_size, feature_dim)
            state_input = state[None, ...]
            policy, value = self.model(state_input, training=False)
            policy = policy.numpy()[0]  # shape: (num_actions,)
            action = np.random.choice(self.num_actions, p=policy)

            next_state, reward, done = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            step_counter += 1
            pbar.update(1)  # 1ステップ進むたびに更新
            state = next_state if next_state is not None else state

        pbar.close()
        print(
            f"[RL] Episode finished in {step_counter} steps. Total Reward: {np.sum(rewards):.6f}")
        return states, actions, rewards

    def compute_returns(self, rewards):
        """
        割引累積報酬(リターン)を計算する

        Args:
            rewards (list of float): エピソード中の報酬

        Returns:
            np.array: 割引累積報酬 (shape: (episode_length,))
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns, dtype=np.float32)

    def train_on_episode(self, states, actions, rewards):
        """
        1エピソード分で収集したデータを用いてモデルのパラメータを更新する

        Args:
            states: list of state (各stateの形状: (window_size, feature_dim))
            actions: list of int
            rewards: list of float

        Returns:
            total_loss (float), mean_return (float), mean_advantage (float)
        """
        returns = self.compute_returns(rewards)
        # shape: (episode_length, window_size, feature_dim)
        states = np.array(states)
        actions = np.array(actions)
        returns = returns.reshape(-1)

        with tf.GradientTape() as tape:
            policy, values = self.model(states, training=True)
            values = tf.squeeze(values, axis=1)
            action_one_hot = tf.one_hot(actions, depth=self.num_actions)
            prob_actions = tf.reduce_sum(policy * action_one_hot, axis=1)
            log_prob = tf.math.log(prob_actions + 1e-8)
            advantage = returns - values
            actor_loss = -tf.reduce_mean(log_prob * advantage)
            critic_loss = tf.reduce_mean(tf.square(advantage))
            entropy_loss = - \
                tf.reduce_mean(tf.reduce_sum(
                    policy * tf.math.log(policy + 1e-8), axis=1))
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return total_loss.numpy(), np.mean(returns), np.mean(advantage)

    def train(self, num_episodes, print_interval=10):
        """
        指定されたエピソード数だけ学習を実施し、各エピソード毎の損失・リターン・アドバンテージを表示する

        Args:
            num_episodes (int): 学習エピソード数
            print_interval (int): 経過表示するエピソード間隔
        """
        for episode in range(1, num_episodes + 1):
            states, actions, rewards = self.run_episode()
            loss, mean_return, mean_advantage = self.train_on_episode(
                states, actions, rewards)
            if episode % print_interval == 0:
                print(
                    f"Episode {episode}, Loss: {loss:.4f}, Mean Return: {mean_return:.4f}, Mean Advantage: {mean_advantage:.4f}")
