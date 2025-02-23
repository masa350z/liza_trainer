import numpy as np


class TradingEnv:
    """単一のトレーディング環境クラス"""

    def __init__(self, prices, window_size):
        """
        時系列の価格データとウィンドウサイズを受け取り、環境を初期化する。

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
        環境を初期状態にリセットし、最初の状態を返す。

        Returns:
            self._get_state()
        """
        self.current_index = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0.0

        return self._get_state()

    def get_state_shape(self):
        """
        この環境が返す「価格ウィンドウ」の形状を返す。

        Returns:
            tuple of int:
                (window_size, 1)
        """
        return (self.window_size, 1)

    def _get_state(self):
        """
        現在の観測状態を返す (window_size 個の価格、ポジション、含み損益)

        Returns:
            (price_window, position, unrealized_pl):
                price_window (np.ndarray): (window_size, 1)
                position (int): ポジション
                unrealized_pl (float): 含み損益
        """
        # 直近 window_size 個の価格を取得
        price_window = self.prices[self.current_index -
                                   self.window_size:self.current_index]
        price_window = price_window.reshape(-1, 1)

        # 含み損益
        current_price = price_window[-1, 0]
        unrealized_pl = 0.0
        if self.position != 0:
            unrealized_pl = (current_price - self.entry_price) * self.position

        # (価格ウィンドウ, ポジション, 含み損益) を返す
        return price_window, self.position, unrealized_pl

    def step(self, action):
        """
        1ステップ進める。

        action (int): 
            0: Hold,
            1: Enter Long,
            2: Enter Short,
            3: Exit

        Returns:
            next_state (tuple or None):
                次状態( (price_window, position, unrealized_pl) ) または None (done時)
            reward (float): 報酬
            done (bool): エピソード終了フラグ
        """

        reward = 0.0

        # 終端判定
        if self.current_index >= len(self.prices):
            self.done = True
            # 未決済ポジションがあれば清算
            if self.position != 0:
                final_price = self.prices[-1]
                reward += ((final_price - self.entry_price)
                           * self.position)/self.entry_price
                self.position = 0
                self.entry_price = 0.0
            return None, reward, self.done

        current_price = self.prices[self.current_index]

        # --- ポジションに関するロジックは従来通り ---
        if self.position == 0:
            # ノーポジの時
            if action == 1:  # Enter Long
                self.position = 1
                self.entry_price = current_price
            elif action == 2:  # Enter Short
                self.position = -1
                self.entry_price = current_price
        else:
            # すでにロングorショート
            if (action == 1 and self.position == -1) or \
               (action == 2 and self.position == 1) or \
               (action == 3):  # Exit
                # 決済
                reward = ((current_price - self.entry_price)
                          * self.position)/self.entry_price
                self.position = 0
                self.entry_price = 0.0
                # もし Enter し直すならここでまた position =1 or -1 にする

        self.current_index += 1

        # 次状態を返す
        next_state = self._get_state()

        return next_state, reward, self.done


class TradingEnvPrediction(TradingEnv):
    """価格予測情報(predictions)を状態として利用する環境(単一)"""

    def __init__(self, prices, predictions, window_size):
        """
        Args:
            prices (list or np.array): 実際の価格データ（報酬計算で使用）
            predictions (list or np.array): 予測モデルの出力確率 (形状: (N, 2))
            window_size (int): 状態として使用する過去ステップ数
        """
        # 親クラス(TradingEnv)を初期化
        super().__init__(prices, window_size)
        # 予測値を float32 で保持
        self.predictions = np.array(predictions, dtype=np.float32)

    def get_state_shape(self):
        """
        この環境が返す「予測ウィンドウ」の形状を返す。

        Returns:
            tuple of int:
                (window_size, 2)
        """
        return (self.window_size, 2)

    def _get_state(self):
        """
        現在の観測状態を返す。

        Returns:
            (pred_window, position, unrealized_pl):
                pred_window (np.ndarray): (window_size, 2) 直近の予測確率
                position (int): 現在のポジション
                unrealized_pl (float): 含み損益
        """
        # 確率ウィンドウ: (window_size, 2)
        pred_window = self.predictions[self.current_index -
                                       self.window_size: self.current_index]

        # 実際の価格は報酬計算にも使うため
        current_price = self.prices[self.current_index - 1]

        # 含み損益 (position が 1 => ロング, -1 => ショート)
        unrealized_pl = 0.0
        if self.position != 0:
            unrealized_pl = ((current_price - self.entry_price)
                             * self.position)/self.entry_price

        # ここでは (pred_window, position, unrealized_pl) のタプルを返す
        return pred_window, self.position, unrealized_pl


class VectorizedTradingEnv:
    """複数の TradingEnv インスタンスをまとめて同時に扱うクラス"""

    def __init__(self, env_list):
        """
        Args:
            env_list (list of TradingEnv): 複数の単一環境のリスト
        """
        self.envs = env_list           # 全ての環境を保持
        self.num_envs = len(env_list)  # 環境数(並列数)

    def reset(self):
        """
        全ての環境をリセットし、状態をバッチでまとめて返す。

        Returns:
            tuple (price(prediction)_states, positions, pls):
                price(prediction)_states (np.ndarray): shape (num_envs, window_size, (1 or 2))
                positions (np.ndarray):    shape (num_envs,)
                pls (np.ndarray):          shape (num_envs,)
        """
        price_states = []
        positions = []
        pls = []
        for env in self.envs:
            # 各環境をリセット
            state = env.reset()  # (price_window, pos, pl)
            price_states.append(state[0])   # price(prediction)_window
            positions.append(state[1])      # (num_envs,)
            pls.append(state[2])            # (num_envs,)

        # 各環境の状態をまとめて numpy 配列にスタックする
        return (np.stack(price_states, axis=0),  # (num_envs, window_size, 1)
                np.array(positions, dtype=np.int32),
                np.array(pls, dtype=np.float32))

    def step(self, actions):
        """
        複数環境に対して同時にアクションを適用し、
        次の状態・報酬・doneフラグをまとめて返す。

        Args:
            actions (np.ndarray): shape (num_envs,)
                                  各環境への行動指定

        Returns:
            next_states (tuple): (price_states, positions, pls)
            rewards (np.ndarray): shape (num_envs,)
            dones (np.ndarray):   shape (num_envs,) (bool)
        """
        price_states = []
        positions = []
        pls = []
        rewards = []
        dones = []

        # 全環境に対してアクションを適用
        for env, act in zip(self.envs, actions):
            shape = env.get_state_shape()

            if env.done:  # すでにdoneの環境は、その後も何もしない
                price_states.append(
                    np.zeros(shape, dtype=np.float32))
                positions.append(0)
                pls.append(0.0)
                rewards.append(0.0)
                dones.append(True)
                continue

            # 環境を1ステップ進める
            next_state, rew, done = env.step(act)
            if done:
                if next_state is None:
                    # エピソード終了
                    price_states.append(
                        np.zeros(shape, dtype=np.float32))
                    positions.append(0)
                    pls.append(0.0)
                else:
                    # next_state は本来 None のはずなので実質同じ
                    price_states.append(next_state[0])
                    positions.append(next_state[1])
                    pls.append(next_state[2])
            else:
                price_states.append(next_state[0])
                positions.append(next_state[1])
                pls.append(next_state[2])

            rewards.append(rew)
            dones.append(done)

        # 次状態を (num_envs, ...) 形式でまとめ
        return ((np.stack(price_states, axis=0),
                 np.array(positions, dtype=np.int32),
                 np.array(pls, dtype=np.float32)),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool))
