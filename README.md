# liza_trainer

為替の1分足CSVデータを読み込み、単純なニューラルネットワーク(またはTransformerなど)を用いて
「過去の価格から、指定した先の価格が上昇か下降かを2値分類する」学習を行うプロジェクトです。

さらに、学習ロジックとして以下を備えています:
- バリデーションロスが改善しなくなったら重みを一部ランダム再初期化して局所解回避
- 一定エポック改善がなければ早期終了
- 複数回のリピート(試行)を行い、テストセットで最良だったモデルの重みを保存

実行のたびに「日時 + 使用したモデル種別」を組み合わせたディレクトリに結果が出力されるため、
通貨ペア（USDJPY / EURUSD）とモデルを変えて複数実行した場合でもログが混ざりません。

---

## ディレクトリ構成

```
liza_trainer
├── main.py
├── modules
│   ├── data_loader.py
│   ├── dataset.py
│   ├── models.py
│   └── trainer.py
├── data
│   ├── sample_EURUSD_1m.csv
│   └── sample_USDJPY_1m.csv
└── results
    ├── EURUSD
    │   └── Affine_20250126-153045
    │       ├── best_model_weights.h5
    │       └── training_info.txt
    └── USDJPY
        └── ...
```

### ファイル概要

- **`main.py`**
  - エントリーポイント。学習を実行する。
  - 通貨ペアや、学習に使うモデルを指定する。
  - 結果を `results/<通貨ペア>/<モデル名_日時>/` に出力。

- **`modules/data_loader.py`**
  - CSVファイル(`timestamp,price`構成)を読み込む関数。

- **`modules/dataset.py`**
  - データを `(train_x, train_y), (valid_x, valid_y), (test_x, test_y)` に分割し、2値分類ラベルの作成やクラスバランス調整を行う関数。

- **`modules/models.py`**
  - ニューラルネットワーク（Affine / Transformer）の定義を行うモジュール。

- **`modules/trainer.py`**
  - 学習ロジック全般を管理するクラス。
    - バリデーションロスが改善しなくなったら重みを部分的にランダム初期化する
    - 一定エポック最良スコアが更新されなければ早期終了
    - 複数回リピートし、テストデータ評価が最良のモデルを保管

---

## 必要環境

- Python 3系
- TensorFlow 2系
- NumPy
- pandas

---

## 使い方

1. **リポジトリをクローンまたはダウンロード**
   ディレクトリ構成は上記の通りに置いてください。

2. **CSVデータを用意**
   `data` フォルダ内に `sample_USDJPY_1m.csv` および `sample_EURUSD_1m.csv` を配置してください。
   形式は以下の2列のみです。行数は何行あっても構いません。

```
timestamp,price
1579068180,1.11508
1579068240,1.11509
...  
```
- `timestamp` は秒単位などのタイムスタンプ(整数)
- `price` は浮動小数点数  

3. **main.py を編集または実行**
   `main.py` 冒頭の `pair = "EURUSD"` を `"USDJPY"` に切り替えれば通貨ペアが変わります。

```python
pair = "EURUSD"  # => "USDJPY" にすればドル円データ
```

   他にも、モデル定義やハイパーパラメータを変えたい場合、`build_simple_affine_model` を別のモデルに変更できます。

4. **実行**

```bash
python main.py
```

   学習が始まり、ターミナルにエポックごとのロス・精度が表示されます。
   エポックごとに同じ行を上書き表示するため、最新のエポック結果のみが確認できます。
   完了後、`results/<pair>/<ModelClass>_<YYYYMMDD-HHMMSS>/` ディレクトリが作られ、
   - `best_model_weights.h5` : テストセット評価が最良だったモデル重み
   - `training_info.txt` : 各種パラメータと精度を記録したファイル

   が保存されます。

---

## 出力確認

`training_info.txt` には、学習日時や使用モデル、ベストバリデーション損失・精度、テスト損失・精度などが書かれています。
同じペア・同じモデルで再度 `main.py` を実行すると、再び日時を含む別ディレクトリが作成され、過去の結果と混ざらずに保存されます。

---

## 学習ロジックの詳細

### Train / Valid / Test に分割

`dataset.py` の `create_dataset` にて `k` (入力区間長) と `future_k` (予測先) を指定し、
`(train_x, train_y), (valid_x, valid_y), (test_x, test_y)` を作成。
`train_ratio=0.6, valid_ratio=0.2` の場合、 6:2:2 に分割されます。

### モデル構築

例: `build_simple_affine_model()` は 全結合レイヤーを2段重ね + 2出力(softmax) という2値分類用の構造です。
`models.py` 内に、`build_transformer_model()` の例もあります。

### Trainer クラスによる学習管理

- `_train_loop()` でエポックを回す。
- エポックごとにバリデーション評価をし、改善しなければ `patience` の猶予後に一部重みをランダム初期化して局所解を回避。
- さらに `early_stop_patience` の猶予を超えてバリデーションの最良記録が更新されなければエポックを打ち切り(早期終了)。
- これを `num_repeats` 回試行し、その都度テストセットを評価して最良の損失を更新し続け、最後にベストを保存。

---

## 備考

- GPU利用時は `main.py` 内で `tf.config.experimental.set_memory_growth(gpu, True)` を行い、必要な分だけメモリを確保する形にしています。
- 価格予測の精度を上げるには、特徴量の拡張やモデル構造の工夫などが必要です。
- ロジックの分割売買や実稼働の自動化はこのプロジェクトには含まれていません。
- ソースコードはすべて `modules/` ディレクトリ内にあるため、学習ロジックの詳細を参照したい場合はそちらを読んでください。

