"""
トレーディング環境モジュール(単一環境版)

ヒストリカルな価格データを用いて、1エピソード単位のトレーディング環境をシミュレートします。
エージェントは以下の4つの行動を取ります。
    0: Hold (何もしない)
    1: Enter Long (ロングエントリー)
    2: Enter Short (ショートエントリー)
    3: Exit (ポジションクローズ)

状態は下記のタプル (price_window, position, unrealized_pl) を返却します:
    - price_window.shape: (window_size, 1)
    - position (int): 現在のポジション (0: ノーポジション, 1: ロング, -1: ショート)
    - unrealized_pl (float): 現在の含み損益

報酬はポジションをクローズした際の利益(または損失)とし、
エピソード終了時に未決済ポジションがあれば自動でクローズした想定で清算します。
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

        self.position = 0       # 0: ノーポジション, 1: ロング, -1: ショート
        self.entry_price = 0.0  # 保有ポジションの建値

    def reset(self):
        """
        環境を初期状態にリセットし、最初の状態を返す

        Returns:
            (price_window, position, unrealized_pl)
        """
        self.current_index = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0.0

        return self._get_state()

    def _get_state(self):
        """
        現在の観測状態を返す (window_size 個の価格、ポジション、含み損益)

        Returns:
            (price_window, position, unrealized_pl)
        """
        price_window = self.prices[self.current_index -
                                   self.window_size:self.current_index]
        price_window = price_window.reshape(-1, 1)

        # 含み損益
        current_price = price_window[-1, 0]
        unrealized_pl = 0.0
        if self.position != 0:
            unrealized_pl = (current_price - self.entry_price) * self.position

        return price_window, self.position, unrealized_pl

    def step(self, action):
        """
        1ステップ実行して次状態・報酬・終了フラグを返す

        Args:
            action (int):
                0: Hold
                1: Enter Long
                2: Enter Short

        Returns:
            next_state (tuple or None):
                (price_window, position, unrealized_pl)
                エピソード終了時(None)
            reward (float):
            done (bool):
        """
        if self.current_index >= len(self.prices):
            # すでに末尾まで来ている場合は強制終了
            reward = 0.0
            if self.position != 0:
                # 未決済ポジションの清算
                final_price = self.prices[-1]
                reward = (final_price - self.entry_price) * self.position
            self.done = True
            return None, reward, self.done

        current_price = self.prices[self.current_index]
        reward = 0.0

        if self.position == 0:
            # ノーポジ時
            if action == 1:   # Enter Long
                self.position = 1
                self.entry_price = current_price
            elif action == 2:  # Enter Short
                self.position = -1
                self.entry_price = current_price

        else:
            # すでにロング or ショート
            if action == 1 and self.position == -1:
                reward = (current_price - self.entry_price) * self.position
                self.position = 1
                self.entry_price = current_price

            elif action == 2 and self.position == 1:
                reward = (current_price - self.entry_price) * self.position
                self.position = -1
                self.entry_price = current_price

        self.current_index += 1

        if self.current_index >= len(self.prices):
            # 終了処理
            self.done = True
            # 未決済ポジションがあれば清算
            if self.position != 0:
                final_price = self.prices[-1]
                reward += (final_price - self.entry_price) * self.position
                self.position = 0
                self.entry_price = 0.0
            return None, reward, self.done

        next_state = self._get_state()
        return next_state, reward, self.done
