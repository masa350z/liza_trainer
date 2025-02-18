# modules/simulator_core.py
"""並列処理でランダムエントリーの利確/損切りシミュレーションを行うモジュール
   - 全ステップのログではなく、最終資産だけをログ保存するバージョン
"""

import os
import csv
import random
import numpy as np
from multiprocessing import Pool

from modules.data_loader import load_csv_data


def run_simulations_with_paramgrid(pair, rik_values, son_values, num_chunks=4,
                                   output_logs=True, spread=0.0):
    """
    複数の (rik, son) パラメータを試し、終値資産を格子状にまとめた行列を返す。

    Args:
        pair (str): "USDJPY" or "EURUSD"
        rik_values (list of float): 利確パラメータの候補
        son_values (list of float): 損切りパラメータの候補
        num_chunks (int): 価格データを何分割するか (並列用)
        output_logs (bool): TrueならCSVログを残す (最終資産のみ)
        spread (float): 絶対値で指定するスプレッド

    Returns:
        np.ndarray: shape (len(rik_values), len(son_values)) の行列
                    [i, j] に (rik_values[i], son_values[j]) の最終資産を格納
    """
    # 1. ヒストリカルデータ読み込み
    csv_file = f"data/sample_{pair}_1m.csv"
    timestamps, prices = load_csv_data(csv_file)
    prices_arr = np.array(prices, dtype=np.float64)

    # 2. データ分割 (連続したままnum_chunksに切り分け)
    chunk_size = len(prices_arr) // num_chunks
    chunks = []
    start_idx = 0
    for i in range(num_chunks):
        end_idx = start_idx + chunk_size
        if i == num_chunks - 1:
            end_idx = len(prices_arr)
        chunk_data = prices_arr[start_idx:end_idx]
        chunks.append(chunk_data)
        start_idx = end_idx

    # 3. パラメータグリッドを for ループで回す
    final_asset_matrix = np.zeros(
        (len(rik_values), len(son_values)), dtype=np.float64)

    # 保存ディレクトリ
    out_dir = f"simulator_results/{pair}/logs"
    os.makedirs(out_dir, exist_ok=True)

    for i, rik in enumerate(rik_values):
        for j, son in enumerate(son_values):
            param_str = f"rik{rik:.6f}_son{son:.6f}"
            log_path = os.path.join(out_dir, f"log_{param_str}.csv")

            # 既存ログがあればスキップ
            if os.path.exists(log_path):
                # 最終資産だけが2行目に書かれている想定
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")
                    # lines[0] -> "final_asset"
                    # lines[1] -> "123.456"
                    final_asset = float(lines[1])
                final_asset_matrix[i, j] = final_asset
                print(
                    f"[SKIP] param {param_str} found in logs. final_asset={final_asset:.6f}")
                continue

            # 新規シミュレーション
            final_asset = simulate_param_with_chunks(chunks, rik, son,
                                                     out_dir=out_dir,
                                                     output_logs=output_logs,
                                                     spread=spread)
            final_asset_matrix[i, j] = final_asset

    return final_asset_matrix


def simulate_param_with_chunks(chunks, rik, son, out_dir, output_logs=True, spread=0.0):
    """
    num_chunks個に分けたprice配列を、それぞれ並列にシミュレート。
    前のチャンクの最終資産を次のチャンクに継承し、最後に得られる最終資産を返す。

    Args:
        chunks (list of np.ndarray): 分割された価格配列
        rik (float): 利確閾値
        son (float): 損切り閾値
        out_dir (str): ログ保存先ディレクトリ
        output_logs (bool): True なら最終資産CSVを保存
        spread (float): 絶対値で指定するスプレッド

    Returns:
        float: (最終的な合算資産)
    """
    with Pool(processes=len(chunks)) as pool:
        # chunkごとに simulate_one_chunk(資産0スタート)
        results = pool.starmap(simulate_one_chunk,
                               [(chunk, rik, son, idx, spread) for idx, chunk in enumerate(chunks)])

    # resultsは [(final_asset, asset_list, chunk_index), ...]
    # chunk_index順に並べ替えて "前チャンクの最終値" を次チャンクに足し合わせる
    results.sort(key=lambda x: x[2])  # chunk_index

    current_offset = 0.0
    for final_asset_chunk, asset_list_chunk, c_idx in results:
        current_offset += final_asset_chunk

    total_asset = current_offset

    if output_logs:
        param_str = f"rik{rik:.6f}_son{son:.6f}"
        log_path = os.path.join(out_dir, f"log_{param_str}.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 最終資産のみを保存
            writer.writerow(["final_asset"])
            writer.writerow([f"{total_asset:.6f}"])

    print(f"rik {rik:.6f}_son {son:.6f}, asset:{total_asset:.6f}")
    return total_asset


def simulate_one_chunk(price_array, rik, son, chunk_index, spread=0.0):
    """
    chunk内を「資産0スタート」でランダムエントリーシミュレーション。

    Args:
    price_array (np.ndarray): 分割された価格配列
    rik (float): 利確閾値
    son (float): 損切り閾値
    chunk_index (int): チャンクのインデックス
    spread (float): 絶対値で指定するスプレッド

    Returns:
        (final_asset, asset_history, chunk_index)
    """
    asset = 0.0
    pos = 0
    entry_price = 0.0

    for price in price_array:
        if pos != 0:
            diff = (price - entry_price) * pos
            # 利確
            if diff > rik * entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0
            # 損切り
            elif diff < -son * entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0

        if pos == 0:
            # 新規エントリー (50%でLONG, 50%でSHORT)
            pos = 1 if random.random() > 0.5 else -1
            entry_price = price
            asset -= spread

    return (asset, None, chunk_index)
