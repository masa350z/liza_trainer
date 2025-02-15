"""
強化学習トレーナーモジュール(単一環境版)

TradingEnv を利用して、1つの環境をエピソード単位で実行し、
得られた状態・行動・報酬を使ってActor-Criticモデルを学習します。
"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf


def compute_position_returns(rewards_list, actions_list):
    """
    単一環境版の「ポジションエントリー時に報酬を一括で記録する」関数。

    元のベクトル化環境用コードでは rewards_list, actions_list が (T, batch_size)
    の形でしたが、単一環境では (T,) になるように変更しています。

    Args:
        rewards_list (list or np.array of float): 各ステップの報酬 (長さ T)
        actions_list (list or np.array of int): 各ステップの行動 (長さ T)
            0: Hold
            1: Enter Long
            2: Enter Short
            3: Exit

    Returns:
        position_rewards (np.array):
            shape (T,) の配列で、
            「取引を開始したステップ(start_index)」にのみ
            その取引で発生した報酬をまとめて記録し、他ステップは 0。
    """
    rewards_list = np.array(rewards_list)
    actions_list = np.array(actions_list)

    T = len(rewards_list)
    position_rewards = np.zeros_like(rewards_list)  # shape: (T,)

    pos = 0         # 現在のポジション状態（0: 未ポジション, 1: ポジションあり）
    start_index = None

    for i in range(T):
        action = actions_list[i]
        profit = rewards_list[i]

        if pos == 0:
            # 未ポジションの場合、新規エントリー（Enter Long or Enter Short）なら開始する
            if action == 1 or action == 2:
                start_index = i
                pos = 1
        else:
            if profit != 0:  # 報酬が0でない＝(このステップで)決済が行われたと仮定
                # 取引開始ステップに報酬を記録
                position_rewards[start_index] = profit

                # 決済後に再度エントリーした場合
                if action == 1 or action == 2:
                    start_index = i
                    pos = 1
                else:
                    pos = 0

    return position_rewards


def fill_zeros_with_partial_fractions(arr):
    """
    連続する2つの非0要素 a[i], a[j] の間にある 0 の部分を、
    例: [a, 0, 0, 0, b] -> [a, b/4, 2b/4, 3b/4, b]
    のように「次の非0要素の値 b を (区間の長さ) で分割」して埋める関数。

    ただし、ユーザー例では a は無視して b のみを基準に埋める形になっています。
    例:
       [ a, 0, 0, 0, 0, b ] -> [ a, b/5, 2b/5, 3b/5, 4b/5, b ]
       [ a, 0, 0, 0, 0, 0, 0, b ] -> [ a, b/7, 2b/7, 3b/7, 4b/7, 5b/7, 6b/7, b ]
       [ a, 0, 0, 0, 0, 0, 0, b, 0, 0, c ]
         -> [ a, b/7, 2b/7, 3b/7, 4b/7, 5b/7, 6b/7, b, c/3, 2c/3, c ]

    Args:
        arr (np.ndarray): 1次元配列。0 と 非0 が混在している。

    Returns:
        np.ndarray: 入力 arr と同じ形状で、0 を上記ルールで置き換えた配列。
    """
    arr = arr.copy()
    n = len(arr)
    i = 0

    while i < n:
        # 現在位置が非0かどうかチェック
        if arr[i] != 0:
            # 次の非0要素を探す
            j = i + 1
            while j < n and arr[j] == 0:
                j += 1

            # j < n ならば arr[j] が非0
            if j < n:
                # i+1 ~ j-1 が0だった区間を埋める
                # 区間の長さ = j - i
                # 間にある 0 の個数 = (j - i - 1)
                length = j - i
                # arr[j] の値を length で分割して埋める (ユーザー例のロジックに準拠)
                for k in range(1, length):
                    arr[i + k] = (k * arr[j]) / length

                i = j  # 次の非0地点へ移動
            else:
                # 次の非0要素が見つからない場合 -> 末尾まで 0 だけ
                # → ここでは特に埋めずに終了
                break
        else:
            # 今が 0 の場合は、とりあえず次へ
            i += 1

    return arr


def fill_zeros_with_reverse_partial_fractions(arr):
    """
    連続する2つの非0要素 a[i], a[j] の間にある 0 の部分を、
    [a, 0, 0, 0, ..., b] -> [a, (length-1)*b/length, (length-2)*b/length, ..., (1)*b/length, b]
    のように「次の非0要素 b を、区間の長さで分割し、逆順に埋める」関数。

    例えば:
      [ a, 0, 0, 0, 0, b ]
        -> [ a, 4b/5, 3b/5, 2b/5, 1b/5, b ]

      [ a, 0, 0, 0, 0, 0, 0, b ]
        -> [ a, 6b/7, 5b/7, 4b/7, 3b/7, 2b/7, 1b/7, b ]

    Args:
        arr (np.ndarray): 1次元配列。0 と 非0 が混在している。

    Returns:
        np.ndarray: 入力 arr と同じ形状で、0 を上記ルールで置き換えた配列。
    """
    arr = arr.copy()
    n = len(arr)
    i = 0

    while i < n:
        if arr[i] != 0:
            # 次の非0要素を探す
            j = i + 1
            while j < n and arr[j] == 0:
                j += 1

            if j < n:
                # i+1 ~ j-1 の0を「次の非0 arr[j] を 逆順で分配」して埋める
                length = j - i  # (例: 5なら 間に4つ0がある)
                for k in range(1, length):
                    # ここだけ逆順に埋める処理
                    arr[i + k] = (length - k) * arr[j] / length
                i = j  # 次の非0地点へ
            else:
                # 次の非0要素が見つからない(末尾まで0だけ)
                break
        else:
            i += 1

    return arr


class RLTrainerSingleEnv:
    """単一環境用のシンプルなActor-Criticトレーナー"""

    def __init__(self, model, optimizer, env, gamma=0.99, num_actions=4):
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
        self.gamma = gamma
        self.num_actions = num_actions

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

            # (バッチサイズ=1で)モデルの推論
            # shape: (1, window_size, 1)
            price_window_input = price_window[None, ...]
            position_input = np.array([position], dtype=np.int32)
            pl_input = np.array([pl], dtype=np.float32)

            policy, _ = self.model([price_window_input, position_input, pl_input],
                                   training=False)
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

    def compute_returns(self, rewards_list):
        """
        割引累積報酬(リターン)を計算する

        Args:
            rewards_list (list of float): ステップごとの報酬

        Returns:
            returns (numpy.ndarray): 各ステップに対応するリターン
        """
        returns = np.zeros_like(rewards_list, dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards_list))):
            G = rewards_list[t] + self.gamma * G
            returns[t] = G
        return returns

    def train_on_episode(self, states_list, actions_list, rewards_list, mini_batch_size=32):
        """
        1エピソード分の (state, action, reward) から学習を行う
        """
        # 割引累積報酬
        returns = self.compute_returns(rewards_list)
        # returns = fill_zeros_with_partial_fractions(rewards_list)
        # returns = fill_zeros_with_reverse_partial_fractions(rewards_list)
        # returns = compute_position_returns(rewards_list, actions_list)

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
        returns_array = np.array(returns, dtype=np.float32)

        # バッチ化してミニバッチ単位で最適化
        total_samples = len(returns_array)
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
            batch_returns = returns_array[batch_idx]
            batch_actions = actions_array[batch_idx]

            with tf.GradientTape() as tape:
                policy, values = self.model(
                    [batch_price, batch_pos, batch_pl], training=True)
                values = tf.squeeze(values, axis=1)  # (batch_size,)

                # 取り得る行動の one-hot ベクトル
                action_one_hot = tf.one_hot(
                    batch_actions, depth=self.num_actions)
                # その行動が選択される確率
                prob_actions = tf.reduce_sum(policy * action_one_hot, axis=1)
                # log π(a|s)
                log_prob = tf.math.log(prob_actions + 1e-8)

                advantage = batch_returns - values  # (batch_size,)

                actor_loss = -tf.reduce_mean(log_prob * advantage)
                critic_loss = tf.reduce_mean(tf.square(advantage))
                # エントロピーによる探索ボーナス
                entropy_loss = - \
                    tf.reduce_mean(tf.reduce_sum(
                        policy * tf.math.log(policy + 1e-8), axis=1))

                # 損失の合計
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            total_loss += loss.numpy()
            num_batches += 1

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        sum_return = returns_array.sum()
        mean_advantage = advantage.numpy().mean()

        return average_loss, sum_return, mean_advantage

    def train(self, num_episodes=10, print_interval=1, mini_batch_size=32):
        """
        指定したエピソード数だけ学習を行う

        Args:
            num_episodes (int): エピソード数
            print_interval (int): ログ表示の間隔
            mini_batch_size (int): 1エピソード内でのミニバッチサイズ
        """
        for episode in range(1, num_episodes + 1):
            states_list, actions_list, rewards_list = self.run_episode()

            loss, sum_return, mean_advantage = self.train_on_episode(
                states_list, actions_list, rewards_list,
                mini_batch_size=mini_batch_size)

            if episode % print_interval == 0:
                print(
                    f"Episode {episode}: Loss={loss:.4f}, SumReturn={sum_return:.4f}, MeanAdv={mean_advantage:.4f}")
