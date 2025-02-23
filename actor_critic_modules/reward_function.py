import numpy as np


def compute_returns(rewards_list, gamma=0.99):
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
        G = rewards_list[t] + gamma * G
        returns[t] = G
    return returns


def compute_position_returns(rewards_list, actions_list):
    """
    単一環境版の「ポジションエントリー時に報酬を一括で記録する」関数。

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


def fill_zeros_with_partial_fractions(rewards_list):
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
    rewards_list = rewards_list.copy()
    n = len(rewards_list)
    i = 0

    while i < n:
        # 現在位置が非0かどうかチェック
        if rewards_list[i] != 0:
            # 次の非0要素を探す
            j = i + 1
            while j < n and rewards_list[j] == 0:
                j += 1

            # j < n ならば arr[j] が非0
            if j < n:
                # i+1 ~ j-1 が0だった区間を埋める
                # 区間の長さ = j - i
                # 間にある 0 の個数 = (j - i - 1)
                length = j - i
                # arr[j] の値を length で分割して埋める (ユーザー例のロジックに準拠)
                for k in range(1, length):
                    rewards_list[i + k] = (k * rewards_list[j]) / length

                i = j  # 次の非0地点へ移動
            else:
                # 次の非0要素が見つからない場合 -> 末尾まで 0 だけ
                # → ここでは特に埋めずに終了
                break
        else:
            # 今が 0 の場合は、とりあえず次へ
            i += 1

    return rewards_list


def fill_zeros_with_reverse_partial_fractions(rewards_list):
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
    rewards_list = rewards_list.copy()
    n = len(rewards_list)
    i = 0

    while i < n:
        if rewards_list[i] != 0:
            # 次の非0要素を探す
            j = i + 1
            while j < n and rewards_list[j] == 0:
                j += 1

            if j < n:
                # i+1 ~ j-1 の0を「次の非0 arr[j] を 逆順で分配」して埋める
                length = j - i  # (例: 5なら 間に4つ0がある)
                for k in range(1, length):
                    # ここだけ逆順に埋める処理
                    rewards_list[i + k] = (length - k) * \
                        rewards_list[j] / length
                i = j  # 次の非0地点へ
            else:
                # 次の非0要素が見つからない(末尾まで0だけ)
                break
        else:
            i += 1

    return rewards_list
