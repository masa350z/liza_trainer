# modules/simulator_core.py
"""並列処理でランダムエントリーの利確/損切りシミュレーションを行うモジュール

(参考コードを整理し、変数名や構造を簡潔にしたもの)
"""

import os
import csv
import random
import numpy as np
from multiprocessing import Pool

from modules.data_loader import load_csv_data


def run_simulations_with_paramgrid(pair, rik_values, son_values, num_chunks=4,
                                   output_logs=True):
    """
    複数の (rik, son) パラメータを試し、終値資産を格子状にまとめた行列を返す。

    Args:
        pair (str): "USDJPY" or "EURUSD"
        rik_values (list of float): 利確パラメータの候補
        son_values (list of float): 損切りパラメータの候補
        num_chunks (int): 価格データを何分割するか (並列用)
        output_logs (bool): TrueならCSVログを残す

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
            # 事前にログファイルが存在するかチェック
            param_str = f"rik{rik:.4f}_son{son:.4f}"
            log_path = os.path.join(out_dir, f"log_{param_str}.csv")

            if os.path.exists(log_path):
                # 既存ログから最終資産を読み込み、シミュレーションをスキップ
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")
                    # 最終行が "step, asset" として並んでいるはず
                    # 例: lines[-1] == "1234, 56.78"
                    last_line = lines[-1].split(",")
                    final_asset = float(last_line[1])
                final_asset_matrix[i, j] = final_asset
                print(
                    f"[SKIP] param {param_str} found in logs. final_asset={final_asset:.4f}")
                continue

            final_asset = simulate_param_with_chunks(
                chunks, rik, son, out_dir, output_logs=output_logs)
            final_asset_matrix[i, j] = final_asset

    return final_asset_matrix


def simulate_param_with_chunks(chunks, rik, son, out_dir, output_logs=True):
    """
    num_chunks個に分けたprice配列を、それぞれ並列にシミュレートし、最後に
    「チャンク順に資産推移を繋げる」形で最終的な時系列を得る。

    前のチャンクの最終値が次のチャンクに引き継がれ、
    連続的な資産推移として整合が取れた結果を返す。

    Returns:
        float: (最終的な合算資産)
    """
    from multiprocessing import Pool
    import csv
    import os

    # 各chunkを並列処理。simulate_one_chunkは「chunk内で資産0スタート」で計算している
    # -> chunk間の連続性は後段で再構成する
    with Pool(processes=len(chunks)) as pool:
        # 各chunkに対してシミュレーションを実行
        # chunk_indexも渡し、後で正しい順番に並べ替える
        results = pool.starmap(
            simulate_one_chunk,
            [(chunk, rik, son, idx) for idx, chunk in enumerate(chunks)]
        )

    # resultsは [(final_asset, asset_list, chunk_index), ...]
    # chunk順に並べ替えてから、「前のチャンクの最後の資産」を次のチャンクに引き継ぐ
    results.sort(key=lambda x: x[2])  # chunk_indexでソート

    merged_asset_list = []
    current_offset = 0.0  # 前チャンクの最終資産
    for final_asset_chunk, asset_list_chunk, chunk_idx in results:
        # chunk内は0スタートで計算しているので、current_offsetを全体に加算
        offset_chunk_list = [current_offset + val for val in asset_list_chunk]
        merged_asset_list.extend(offset_chunk_list)

        # このチャンク終了時点の資産
        current_offset += final_asset_chunk

    total_asset = current_offset

    # ログ出力
    if output_logs:
        param_str = f"rik{rik:.4f}_son{son:.4f}"
        log_path = os.path.join(out_dir, f"log_{param_str}.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "asset"])
            for idx, val in enumerate(merged_asset_list):
                writer.writerow([idx, val])

    print(f"rik {rik:.4f}_son {son:.4f}, asset:{total_asset:.4f}")
    return total_asset


def simulate_one_chunk(price_array, rik, son, chunk_index):
    """
    1つの価格配列 (chunk) に対し、ランダムエントリーで利確/損切りシミュレーションを実行。
    ここでは chunk内は「資産0スタート」で計算し、最後にfinal_assetだけ返す。

    Returns:
        (final_asset, asset_history, chunk_index)
          final_asset: このchunk内での最終資産(最初0→最後まで)
          asset_history: chunk内の資産推移リスト(最初0→最後まで)
          chunk_index: チャンク番号(並べ直す用)
    """
    import random

    asset = 0.0
    pos = 0  # +1=LONG, -1=SHORT, 0=NOPOS
    entry_price = 0.0

    asset_history = []

    for price in price_array:
        if pos != 0:
            # エントリー中の場合、利確/損切り判定
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
            # 新規エントリー
            pos = 1 if random.random() > 0.5 else -1
            entry_price = price

        asset_history.append(asset)

    return (asset, asset_history, chunk_index)
