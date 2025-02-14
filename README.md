# Liza Trainer

本プロジェクトは、為替の1分足データ（例：EURUSD、USDJPY）を用いて、エンドツーエンドで収益最大化を目指す深層強化学習トレーディングシステムを実現するものです。

このシステムは、**Actor-Critic**アーキテクチャを採用しており、LSTM層による時系列特徴抽出と、ポリシーヘッド・バリューヘッドによる行動選択・状態価値評価を行います。

また、環境側（TradingEnv）では、**現在のポジション**や**ポジション取得後の評価損益**を計算し、これを報酬としてエージェントにフィードバックします。結果として、モデルは単に「上がる／下がる」の分類ではなく、実際の取引結果に基づいた最適なエントリー、ホールド、決済のタイミングを学習し、リスク管理も内包した取引戦略を獲得します。

---

## 特徴

- **時系列入力:**  
  各状態は直近の `window_size` 分の価格データを (window_size, 1) の形で与え、LSTM層により時間的依存性を学習します。

- **Actor-Criticモデル:**  
  モデルは2つの出力ヘッドを持ちます。
  - **ポリシーヘッド:** 4種類の行動（0: Hold, 1: Enter Long, 2: Enter Short, 3: Exit）の選択確率を出力。
  - **バリューヘッド:** 現在の状態における将来的な累積報酬（状態価値）をスカラーで出力。

- **トレーディング環境 (TradingEnv):**  
  環境は、入力価格系列とは別に内部で「現在のポジション（0, 1, -1）」および「エントリー価格」を保持し、Exitアクション時に現在価格とエントリー価格の差から評価損益を計算し、報酬として返します。

- **強化学習トレーナー (RLTrainer):**  
  エピソード単位で環境と対話し、各ステップの状態、行動、報酬を収集。

- **シミュレーション:**  
  学習済みモデルの重みを読み込み、過去データに対してシミュレーションを実行。

---

## ディレクトリ構成
```
liza_trainer/
├── README.md
├── data/
│   ├── EURUSD_1m.csv
│   └── USDJPY_1m.csv
├── modules/
│   ├── data_loader.py
│   ├── env.py
│   ├── models.py
│   └── trainer.py
├── results/
│   ├── models/
│   └── simulations/
├── train.py
└── simulate.py
```

---

## 必要環境

- Python 3.x
- TensorFlow 2.x
- NumPy
- pandas

---

## 各モジュールの詳細

### 1. `modules/data_loader.py`
CSVファイルから `timestamp` と `price` のリストを抽出。

### 2. `modules/env.py` (TradingEnv)
- **内部状態:**  
  - `current_index`: 次に参照する価格のインデックス
  - `position`: 現在のポジション（0: ノーポジション、1: ロング、-1: ショート）
  - `entry_price`: ポジションを開いたときの価格

- **step() メソッド:**
  - Enter Long / Short: ポジションを設定。
  - Hold: 何もしない。
  - Exit: 評価損益（報酬）を計算し、ポジションをクリア。

### 3. `modules/models.py`
- **build_actor_critic_model():**  
  LSTM層で時系列特徴を抽出し、ポリシーヘッドとバリューヘッドを出力。

### 4. `modules/trainer.py` (RLTrainer)
- **run_episode():** 環境との対話を実施。
- **compute_returns():** 割引累積報酬を計算。
- **train_on_episode():** モデルの損失を更新。
- **train():** 指定エピソード数だけ学習を繰り返す。

### 5. `train.py`
- **概要:**  
  指定した通貨ペア（EURUSDまたはUSDJPY）のCSVデータを読み込み、Actor-Criticモデルで学習を実施。
- **実行:**
  ```bash
  python train.py
  ```

### 6. `simulate.py`
- **概要:**  
  学習済みモデルの重みを指定してシミュレーションを実施。
- **実行:**
  ```bash
  python simulate.py --pair EURUSD --weights results/models/EURUSD/ActorCritic_ws30_YYYYMMDD-HHMMSS/best_model.weights.h5 --window_size 30
  ```

---

## 注意事項
- 価格データ（CSV）のフォーマットは `timestamp,price` であることを前提。
- 学習はオフライン環境で実施し、学習済みモデルはシミュレーションで評価。

---

## 実行手順

1. **データ準備:**
   ```bash
   cp EURUSD_1m.csv data/
   cp USDJPY_1m.csv data/
   ```

2. **学習の実行:**
   ```bash
   python train.py
   ```

3. **シミュレーションの実行:**
   ```bash
   python simulate.py --pair EURUSD --weights results/models/EURUSD/ActorCritic_ws30_YYYYMMDD-HHMMSS/best_model.weights.h5 --window_size 30
   ```

---

