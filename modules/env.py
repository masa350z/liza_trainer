"""
トレーディング環境モジュール

ヒストリカルな価格データを用いて、1エピソード単位のトレーディング環境をシミュレートします。
エージェントは以下の4つの行動を取ります。
    0: Hold (何もしない)
    1: Enter Long (ロングエントリー)
    2: Enter Short (ショートエントリー)
    3: Exit (ポジションクローズ)
状態は直近window_size個の価格データ（shape: (window_size, 1)）として提供されます。
報酬はポジションをクローズした際の利益（または損失）として計算され、エピソード終了時に強制的にクローズされます。
"""

import numpy as np


class TradingEnv:
    def __init__(self, prices, window_size):
        """
        Args:
            prices (list or np.array): 時系列の価格データ
            window_size (int): 状態として使用する過去価格の数
        """
        self.prices = np.array(prices, dtype=np.float32)
        self.window_size = window_size
        self.current_index = window_size
        self.done = False
        self.position = 0  # 0: ノーポジション, 1: ロング, -1: ショート
        self.entry_price = 0.0

    def reset(self):
        """
        環境を初期状態にリセットし、最初の状態を返す
        """
        self.current_index = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0.0
        state = self._get_state()
        print(
            f"[ENV] Reset: current_index={self.current_index}, position={self.position}, entry_price={self.entry_price:.4f}")
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
        reward = 0.0
        price = self.prices[self.current_index]
        old_position = self.position
        old_entry_price = self.entry_price

        # 行動に基づく処理
        if self.position == 0:
            if action == 1:
                self.position = 1
                self.entry_price = price
            elif action == 2:
                self.position = -1
                self.entry_price = price
            # Holdの場合は何もしない
        else:
            if action == 3:
                reward = (price - self.entry_price) * self.position
                self.position = 0
                self.entry_price = 0.0

        # デバッグ用出力
        print(f"[ENV] Index: {self.current_index}, Price: {price:.4f}, Action: {action}, "
              f"Old Pos: {old_position}, New Pos: {self.position}, "
              f"Old Entry: {old_entry_price:.4f}, New Entry: {self.entry_price:.4f}, Reward: {reward:.6f}")

        self.current_index += 1
        if self.current_index >= len(self.prices):
            # エピソード終了時、未決済のポジションがあれば強制クローズ
            if self.position != 0:
                final_price = self.prices[-1]
                reward += (final_price - self.entry_price) * self.position
                print(f"[ENV] Force Exit at End: Final Price: {final_price:.4f}, "
                      f"Entry Price: {self.entry_price:.4f}, "
                      f"Additional Reward: {(final_price - self.entry_price) * self.position:.6f}")
                self.position = 0
                self.entry_price = 0.0
            self.done = True

        next_state = self._get_state() if not self.done else None
        return next_state, reward, self.done
