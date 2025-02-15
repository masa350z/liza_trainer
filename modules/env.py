"""
トレーディング環境モジュール

ヒストリカルな価格データを用いて、1エピソード単位のトレーディング環境をシミュレートします。
エージェントは以下の4つの行動を取ります。
    0: Hold (何もしない)
    1: Enter Long (ロングエントリー)
    2: Enter Short (ショートエントリー)
    3: Exit (ポジションクローズ)
状態は直近window_size個の価格データ(shape: (window_size, 1))として提供されます。
報酬はポジションをクローズした際の利益(または損失)として計算され、エピソード終了時に強制的にクローズされます。
"""

import numpy as np


class TradingEnv:
    def __init__(self, prices, window_size, max_hold_steps=60*24):
        """
        Args:
            prices (list or np.array): 時系列の価格データ
            window_size (int): 状態として使用する過去価格の数
            max_hold_steps (int): 同じポジションを保持できる最大ステップ数。これを超えると強制決済。
        """
        self.prices = np.array(prices, dtype=np.float32)
        self.window_size = window_size
        self.current_index = window_size
        self.done = False
        self.position = 0  # 0: ノーポジション, 1: ロング, -1: ショート
        self.entry_price = 0.0
        self.max_hold_steps = max_hold_steps
        self.hold_steps = 0  # 現在のポジションを保持しているステップ数

    def reset(self):
        """
        環境を初期状態にリセットし、最初の状態を返す
        """
        self.current_index = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        state = self._get_state()
        return state

    def _get_state(self):
        """
        現在の状態（直近window_size個の価格）を返す
        出力形状: (window_size, 1)
        """
        state = self.prices[self.current_index -
                            self.window_size:self.current_index]
        return state.reshape(-1, 1)

    def step(self, action):
        """
        1ステップ進める

        Args:
            action (int): 0: Hold, 1: Enter Long, 2: Enter Short, 3: Exit

        Returns:
            next_state: 次の状態 (shape: (window_size, 1)) または None (エピソード終了時)
            reward (float): このステップでの報酬
            done (bool): エピソード終了フラグ
        """
        # まず、次の価格にアクセスする前に終了チェックを行う
        if self.current_index >= len(self.prices):
            reward = 0.0
            if self.position != 0:
                final_price = self.prices[-1]
                reward += (final_price - self.entry_price) * self.position
                self.position = 0
                self.entry_price = 0.0
                self.hold_steps = 0
            self.done = True
            return None, reward, self.done

        # 現在の価格を取得
        price = self.prices[self.current_index]
        reward = 0.0

        if self.position == 0:
            # 新規エントリーの場合
            if action == 1:
                self.position = 1
                self.entry_price = price
                self.hold_steps = 0  # ポジション開始時にリセット

            elif action == 2:
                self.position = -1
                self.entry_price = price
                self.hold_steps = 0

            else:
                reward = 0

        else:
            # ポジションを保持中の場合
            self.hold_steps += 1
            profit = (price - self.entry_price) * self.position

            if action == 1 and self.position == -1:
                reward = profit
                self.position = 1
                self.entry_price = price
                self.hold_steps = 0  # ポジション開始時にリセット

            elif action == 2 and self.position == 1:
                reward = profit
                self.position = -1
                self.entry_price = price
                self.hold_steps = 0

            # Exit 行動が取られた場合または所定の保持ステップ数に達した場合、強制決済
            elif action == 3 or self.hold_steps >= self.max_hold_steps:
                reward = profit
                self.position = 0
                self.entry_price = 0.0
                self.hold_steps = 0

        self.current_index += 1

        # エピソード終了のチェック
        if self.current_index >= len(self.prices):
            self.done = True

        next_state = self._get_state() if not self.done else None

        return next_state, reward, self.done


class VectorizedTradingEnv:
    def __init__(self, env_list):
        """
        Args:
            env_list (list): TradingEnv のインスタンスのリスト
        """
        self.envs = env_list
        self.num_envs = len(env_list)

    def reset(self):
        """
        すべての環境をリセットし、状態をバッチで返す。
        各環境の状態は (window_size, feature_dim) なので、出力は (num_envs, window_size, feature_dim)
        """
        states = []
        for env in self.envs:
            states.append(env.reset())
        return np.stack(states, axis=0)

    def step(self, actions):
        """
        バッチの行動を各環境に適用し、次状態、報酬、done フラグをまとめて返す。

        Args:
            actions (np.array): 各環境での行動 (shape: (num_envs,))

        Returns:
            next_states (np.array): 各環境の次状態 (shape: (num_envs, window_size, feature_dim))
            rewards (np.array): 各環境の報酬 (shape: (num_envs,))
            dones (np.array): 各環境のエピソード終了フラグ (shape: (num_envs,))
        """
        next_states = []
        rewards = []
        dones = []
        for env, action in zip(self.envs, actions):
            ns, r, done = env.step(action)
            # エピソードが終了している場合、状態はゼロ行列で埋める(または reset() を呼ぶなどの処理)
            if done:
                ns = np.zeros_like(env._get_state())
            next_states.append(ns)
            rewards.append(r)
            dones.append(done)
        return np.stack(next_states, axis=0), np.array(rewards), np.array(dones)
