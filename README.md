# liza_trainer

為替の1分足CSVデータを用いて、以下の2つを行うプロジェクトです:

1. **学習 (train.py)**  
   過去の価格から、指定した先の価格が上昇か下降かを推定する2値分類モデルを学習する。  
   バリデーションロスが改善しなければ一部重みをランダム初期化して局所解を回避し、一定エポック改善しなければ早期終了する仕組みを備えている。  
   学習完了後は `results/<通貨ペア>/<モデル名>_<日時>/best_model_weights.h5` に最良モデルが保存される。

2. **シミュレーション (simulate.py)**  
   ランダムエントリーと利確・損切り(複数パラメータ)のバックテストシミュレーションを並列で行う。  
   最終資産をヒートマップにまとめ、 `simulator_results/<通貨ペア>/heatmap_<通貨ペア>.png` として出力する。  

### コンセプト

このプロジェクトのアルゴリズム設計は、以下のコンセプトに基づいています:

- **「最悪AIがポンコツでも勝てる」**: ランダムエントリーでも勝てるアルゴリズムを構築することを第一目標とし、
  利確・損切りのロジックによって安定した利益を追求します。
- その上で、ニューラルネットワーク(NNモデル)を用いてさらに精度を高めることを目指します。
- このアプローチにより、AIモデルの予測精度に完全に依存しない堅牢なシステムを実現します。

---

## ディレクトリ構成

```
liza_trainer
├── train.py          // 学習用のメインスクリプト
├── simulate.py       // シミュレーション用のメインスクリプト
├── data
│   ├── sample_EURUSD_1m.csv
│   └── sample_USDJPY_1m.csv
├── modules
│   ├── data_loader.py
│   ├── dataset.py
│   ├── models.py
│   ├── trainer.py
│   └── simulate_core.py
├── results
│   ├── EURUSD
│   └── USDJPY
└── simulator_results
    ├── EURUSD
    └── USDJPY
```

---

## 学習 (train.py)

- `train.py` では、以下の処理を行う:
  1. 指定した通貨ペア（`EURUSD`または`USDJPY`）のCSV (`data/sample_<pair>_1m.csv`) を読み込み。
  2. `modules/dataset.py` の `create_dataset()` で `(train_x, train_y), (valid_x, valid_y), (test_x, test_y)` を作成。
  3. 全結合モデル (`build_simple_affine_model`) などを構築。
  4. `trainer.py` の `Trainer` クラスを使い、エポックごとにバリデーションを確認。  
     - 改善しなければ一部重みをランダムに初期化し、局所解を回避。  
     - 一定エポック改善が無ければ早期終了。  
  5. テストセット評価が最良のモデルを `results/<pair>/<モデル>_<日時>/best_model_weights.h5` に保存。

### 実行手順

1. CSVファイルを `data/sample_USDJPY_1m.csv` や `data/sample_EURUSD_1m.csv` として配置。  
2. `train.py` の `main()` 内で `pair` を指定。  
3. 下記のように実行すると学習が走り、最終的なモデル重みとパラメータログが保存される。  

```bash
python train.py
```

- 出力結果:
  - `results/<pair>/<モデル>_<日時>/best_model_weights.h5`
  - `training_info.txt` (学習結果やパラメータ)

---

## シミュレーション (simulate.py)

`simulate.py` は、ランダムエントリー + 利確(rik)・損切り(son)の複数パラメータに対するバックテストを行う。

- データを複数チャンクに分割し、`multiprocessing.Pool` を用いて並列にシミュレート。
- チャンクを連結し、最終的な資産を集計。
- 組み合わせごとの最終資産を 2次元配列に格納し、ヒートマップで可視化。

### 実行手順

1. 同様に `data/sample_<pair>_1m.csv` を配置 (EURUSD または USDJPY)。
2. `simulate.py` を実行時に `--pair` オプションで通貨ペアを指定:

```bash
python simulate.py --pair EURUSD
```

- 出力結果:
  - `simulator_results/<pair>/logs/` (パラメータ別のCSVログ)
  - `simulator_results/<pair>/heatmap_<pair>.png` (最終資産のヒートマップ)

---

## 必要環境

- Python 3系
- TensorFlow 2系 (GPU利用可)
- NumPy
- pandas
- matplotlib
- seaborn

---

## 各ファイルの役割

- **train.py**  
  学習のメインスクリプト。適宜 `pair` (`EURUSD` / `USDJPY`) を選択し、学習を実行。

- **simulate.py**  
  シミュレーションのメインスクリプト。`--pair` オプションで通貨ペアを選択し、ヒートマップを出力。

- **data_loader.py**  
  CSVを読み込み、`timestamp` と `price` のリストを返す。

- **dataset.py**  
  `(train_x, train_y, valid_x, valid_y, test_x, test_y)` を生成する。ラベルを2値分類形式で作成。

- **models.py**  
  全結合ネットワーク (`build_simple_affine_model`) や Transformer (`build_transformer_model`) などのモデル定義。

- **trainer.py**  
  バリデーションロスが改善しなければ重みの一部を初期化して局所解回避、一定エポック改善無ければ早期終了し、
  複数回試行 (`num_repeats`) して最良を保存する仕組み。

- **simulate_core.py**  
  `multiprocessing.Pool` を使ってチャンクごとにランダムエントリーシミュレーションを並列実行し、資産を合算する。
  `(rik, son)` のパラメータを網羅的に走査し、最終資産をヒートマップ行列にまとめる。

---

## 注意点

- CSVのカラムは `timestamp, price` のみ。
- GPUメモリを必要分だけ確保するため、`tf.config.experimental.set_memory_growth` を使用。
- 価格データは1分足想定だが、行数の制約は特にない。
- 同じペアで複数回の学習/シミュレーションを行うと、日時つきのフォルダが都度生成され、結果が混ざらない。

