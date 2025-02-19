# modules/simulator_core_ai.py
import os
import csv
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from modules.data_loader import load_csv_data
from modules.models import build_simple_affine_model


def run_simulations_with_paramgrid_ai(pair,
                                      model_path,
                                      k,
                                      rik_values,
                                      son_values,
                                      num_chunks=4,
                                      output_logs=True,
                                      force_run=False):
    print(f"[AI] Loading data for {pair} ...")
    csv_file = f"data/sample_{pair}_1m.csv"
    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        return np.zeros((len(rik_values), len(son_values)))

    _, price_list = load_csv_data(csv_file)
    prices_arr = np.array(price_list, dtype=np.float64)
    print(f"[AI] Data length = {len(prices_arr)}")

    chunk_size = len(prices_arr) // num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size
        if i == num_chunks - 1:
            end = len(prices_arr)
        chunk_data = prices_arr[start:end]
        chunks.append(chunk_data)
        start = end

    out_dir = f"simulator_results/{pair}_AI/logs"
    os.makedirs(out_dir, exist_ok=True)

    final_matrix = np.zeros(
        (len(rik_values), len(son_values)), dtype=np.float64)

    total_skipped = 0
    total_runcalc = 0

    for i, rik in enumerate(rik_values):
        for j, son in enumerate(son_values):
            param_str = f"rik{rik:.4f}_son{son:.4f}"
            log_path = os.path.join(out_dir, f"log_{param_str}.csv")

            if (not force_run) and os.path.exists(log_path):
                # 既存ログがあればスキップ
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")
                    if len(lines) >= 2:
                        last_line = lines[-1].split(",")
                        final_asset = float(last_line[1])
                        final_matrix[i, j] = final_asset
                        print(
                            f"[SKIP][AI] {param_str} => final_asset={final_asset:.3f}")
                        total_skipped += 1
                        continue

            # 新規計算
            print(f"[AI] Simulating param={param_str} (rik={rik}, son={son})")
            final_asset = simulate_param_chunks_ai(chunks, model_path, k, rik, son,
                                                   out_dir, param_str,
                                                   output_logs)
            final_matrix[i, j] = final_asset
            total_runcalc += 1

    # すべての処理終了
    if total_runcalc == 0 and total_skipped > 0:
        print("[AI] All param combos were skipped (existing logs).")
    elif total_runcalc == 0 and total_skipped == 0:
        print("[AI] No param combos found or no data.")
    else:
        print(
            f"[AI] Done. calculated={total_runcalc} combos, skipped={total_skipped} combos.")

    return final_matrix


def simulate_param_chunks_ai(chunks, model_path, k, rik, son,
                             out_dir, param_str,
                             output_logs=True):
    print(f"[AI] Running parallel tasks for {param_str}, chunks={len(chunks)}")

    with Pool(processes=len(chunks)) as pool:
        results = pool.starmap(
            simulate_one_chunk_ai,
            [
                (idx, chunk, model_path, k, rik, son)
                for idx, chunk in enumerate(chunks)
            ]
        )

    # [(chunk_final_asset, asset_history, chunk_idx), ...]
    results.sort(key=lambda x: x[2])
    merged_asset_list = []
    current_offset = 0.0

    for final_chunk_asset, asset_history_chunk, c_idx in results:
        offset_chunk = [current_offset + val for val in asset_history_chunk]
        merged_asset_list.extend(offset_chunk)
        current_offset += final_chunk_asset
        print(
            f"  [AI] chunk={c_idx} done, chunk_final={final_chunk_asset:.3f}")

    total_asset = current_offset
    print(f"[AI] param={param_str} -> total_asset={total_asset:.3f}")

    if output_logs:
        log_path = os.path.join(out_dir, f"log_{param_str}.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as fw:
            writer = csv.writer(fw)
            writer.writerow(["step", "asset"])
            for step_i, val in enumerate(merged_asset_list):
                writer.writerow([step_i, f"{val:.6f}"])

    return total_asset


def simulate_one_chunk_ai(chunk_idx, chunk_prices, model_path, k, rik, son):
    """1つのチャンクをAI推論で計算 (資産0スタート)"""
    import random

    # 学習時のモデル構造を再定義 (Affine例)
    model = build_simple_affine_model(k)
    model.load_weights(model_path)

    asset = 0.0
    pos = 0
    entry_price = 0.0

    price_buffer = []
    asset_history = []

    for price in chunk_prices:
        price_buffer.append(price)
        if len(price_buffer) > k:
            price_buffer.pop(0)

        if pos != 0:
            diff = (price - entry_price)*pos
            if diff > rik*entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0
            elif diff < -son*entry_price:
                asset += diff
                pos = 0
                entry_price = 0.0

        if pos == 0 and len(price_buffer) == k:
            inp = np.array(price_buffer, dtype=np.float32).reshape(1, k)
            pred = model(inp, training=False).numpy()
            if pred[0, 0] > pred[0, 1]:
                pos = 1
            else:
                pos = -1
            entry_price = price

        asset_history.append(asset)

    return (asset, asset_history, chunk_idx)