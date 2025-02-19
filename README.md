# liza_trainer

為替の1分足CSVデータを用いて、以下の作業を行うプロジェクトです:

## 1. 学習 (train.py)

### 概要
- 過去の価格 (`k` ステップ) と、指定した先 (`future_k` ステップ) の価格上昇・下降を2値分類として学習。
- 学習ロジックは `modules/trainer.py` の `Trainer` クラスが管理。
- バリデーション損失が改善しなくなった際、一部重みをランダム初期化して局所解回避。
- 一定エポック改善しなければ早期終了。
- 学習完了後、最良のモデル重みが `results/<通貨ペア>/<モデル種別>_k<k>_f<future_k>_<日時>/best_model.weights.h5` に保存。

### 実行方法
```bash
python train.py
```
- 指定した通貨ペア (`EURUSD` / `USDJPY`) に対して、
- `k` (30, 60, 90, 120, 150, 180) と `future_k` の組み合わせを試行。
- 結果は `results/<pair>/training_info.txt` に保存。

## 2. ランダムエントリーシミュレーション (simulate.py)

### 概要
- ランダムエントリーに基づいて利確 (`rik`)、損切り (`son`) を試行。
- 並列処理を用いて複数パラメータのバックテストを実行。
- 最終資産を `heatmap_<pair>.png` にヒートマップとして保存。
- 結果は `simulator_results/<pair>/logs/log_rikX_sonY.csv` に記録。

### 実行方法
```bash
python simulate.py
```
- `pair` の指定は `simulate.py` 内の `main()` にて。
- `rik`, `son` の範囲やステップは `np.linspace` で指定。

## 3. AIモデルを用いたシミュレーション (simulate_with_ai.py)

### 概要
- 学習済みモデル (`best_model.weights.h5`) を使用。
- AIの予測に基づき、LONG または SHORT でエントリー。
- 各ステップで利確 (`rik`)、損切り (`son`) を判定。
- 結果は `simulator_results/<pair>/AI_logs/` に保存。

### 実行方法
```bash
python simulate_with_ai.py --pair EURUSD \
    --weights results/EURUSD/Affine_k30_f5_20250126-230851/best_model.weights.h5 \
    --k 30 \
    --rik 0.001 \
    --son 0.01
```

## 4. サマリーツール (modules/summary.py)

### 概要
- 学習結果 (`training_info.txt`) から `k, future_k, Best Test Loss, Best Test Acc` などを集計。
- `summary_EURUSD.csv`、`summary_USDJPY.csv` にまとめる。

### 実行方法
```bash
python modules/summary.py
```

## 補足
- **学習時の正規化**: `dataset.py` にて「各サンプルごとに Min-Max 正規化」。
- **推論時の正規化**: `simulate_with_ai.py` にて「各ステップで過去 `k` 個の価格に対し Min-Max を計算し正規化」。
- **モデルの保存・読み込み**:
  - `model.save_weights(...)` で保存。
  - `build_simple_affine_model` でネットワークを再生成し、`load_weights(...)` で読み込み。

