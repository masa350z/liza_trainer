# Liza Trainer

本プロジェクトは、強化学習を用いたFXトレーディングの学習・シミュレーションを行うためのサンプルプロジェクトです。
各モジュールは以下の機能を担当しています:

## 各モジュールの概要

- **modules/data_loader.py**  
  CSVファイルから時系列の価格データ（タイムスタンプと価格）を読み込みます。

- **modules/env.py**  
  シンプルなトレーディング環境を定義。
  - `TradingEnv`: 単一環境
  - `VectorizedTradingEnv`: 複数のトレーディング環境を一括で管理するベクトル化環境

- **modules/models.py**  
  LSTMを用いたActor-Criticモデル（ポリシーヘッドとバリューヘッド）を定義します。

- **modules/trainer.py**  
  ベクトル化環境を使って効率的に経験を収集し、Actor-Criticモデルを学習するためのトレーナを定義します。

- **train.py**  
  上記モジュールを使って実際に学習を実行するスクリプト。

- **simulate.py**  
  学習済みのモデル（重みファイル）を用い、価格データをもとにシミュレーションを実行。

---

## セットアップ

### Python バージョン
Python 3.7 以上を推奨します。

### 依存ライブラリのインストール
プロジェクトのルートディレクトリに移動し、以下のコマンドで `requirements.txt` を使って依存関係をインストールしてください。

```bash
pip install -r requirements.txt
```

また、GPUを利用する場合は `tensorflow-gpu` のインストールが必要な場合があります。

### データの配置
`data` ディレクトリに CSV ファイルを用意してください。
本プロジェクトでは `timestamp, price` の形で保存されていることを前提としています。

例: `EURUSD_1m.csv`, `USDJPY_1m.csv` など。

---

## トレーニングの実行

トレーニングのエントリーポイントは `train.py` です。
デフォルトでは、USDJPY の価格データを使い、ウィンドウサイズ 30、エピソード数 100、並列環境数 1000 などの設定例がハードコードされています。

```bash
python train.py
```

実行が完了すると、結果が `results/models/` 以下に日付つきフォルダで保存され、学習済みモデルの重み (`best_model.weights.h5`) が出力されます。

### パラメータを変更したい場合
`train.py` の `main()` 関数にある以下の引数を変更して、自分のデータに合った学習を行ってください。

```python
main(
    pair='USDJPY',           # "EURUSD" 等、通貨ペアを指定
    window_size=30,         # 過去価格何本分を状態として見るか
    num_episodes=100,       # 学習エピソード数
    num_envs=1000,          # 並列環境数 (データを分割して学習)
    mini_batch_size=10000,  # ミニバッチサイズ
    historical_length='_len1000000'  # CSVファイル名に付与するオプション文字列
)
```

---

## シミュレーションの実行

学習済みモデルの重みを用いて、トレード結果をシミュレーションするには `simulate.py` を実行します。

```bash
python simulate.py \
    --pair USDJPY \
    --weights results/models/USDJPY/ActorCritic_ws30_YYYYMMDD-HHMMSS/best_model.weights.h5 \
    --window_size 30
```

**コマンドライン引数:**

- `--pair`: "EURUSD" や "USDJPY" を指定
- `--weights`: 学習済みモデルの重みファイル (`.h5`) のパス
- `--window_size`: 学習時と同じウィンドウサイズを指定

シミュレーションのステップごとの行動ログと累積報酬（資産推移）が
`results/simulations/<pair>_AI_logs/log_ai_ws<window_size>.csv` というファイル名で出力されます。

---

## ディレクトリ構成

```
liza_trainer
├── .gitignore
├── README.md
├── data
│   ├── .gitkeep
│   ├── EURUSD_1m.csv
│   ├── ... (他の csv データ)
├── modules
│   ├── data_loader.py
│   ├── env.py
│   ├── models.py
│   ├── trainer.py
├── requirements.txt
├── results
│   ├── models
│   │   └── USDJPY
│   │       └── ActorCritic_ws30_YYYYMMDD-HHMMSS
│   │           └── best_model.weights.h5
│   └── simulations
│       └── USDJPY_AI_logs
│           └── log_ai_ws30.csv
├── simulate.py
└── train.py
```

---

## 注意点

- 本実装はサンプルコードであり、スプレッドやスリッページ、リスク管理など実際のトレードに必要な要素は含まれていません。
- 大量のヒストリカルデータ（数百万行）を使用する場合、ハードウェアの性能（メモリやGPU）に依存して学習時間が大きく変化します。
- 本コードは強化学習のトレーディング応用例として提供しているものであり、実運用で使用する場合は十分な検証を行ってください。

