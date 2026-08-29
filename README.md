* 模型：**MDTC-TA (Ours)**
* 数据：**S&P500**
* 时间：**2005/01/03–2026/05/22**
* 输入：**OHLC 四维特征**
* 预测：**下一交易日 Close**
* 模块：

  * Multi-Scale TCN
  * Temporal Attention
  * Feature  Transformer
  * Residual Prediction
* 损失：

  * Huber Loss
* 优化：

  * AdamW
  * CosineAnnealingLR
  * Early Stopping
* 消融：

  * wo_Attention
  * wo_MultiScale
  * wo_Transformer

你直接复制保存为：

```
README.md
```

放在项目根目录即可。

---

```markdown
# MDTC-TA: Stock Index Closing Price Forecasting

## 1. Introduction

This repository provides the implementation of the paper:

**"Stock Index Closing Price Forecasting Based on Multi-Scale Dilated Temporal Convolutional Network with  Temporal Attention"**

The proposed model is:

**Multi-Scale Dilated Temporal Convolutional Network with Temporal Attention (MDTC-TA)**

for next-day stock index closing price prediction.

The model is designed for financial time series forecasting by integrating:

- Multi-Scale Dilated Temporal Convolutional Network (MDTC)
- Temporal Attention Mechanism
- Feature Transformer Module
- Residual Prediction Strategy


The framework aims to improve the ability of deep learning models to capture:

- short-term market fluctuations,
- medium-term temporal patterns,
- long-term price evolution trends.


---

# 2. Research Task


## Dataset

Stock Index:

**S&P500 Index**

Time Period:

```

2005-01-03 ~ 2026-05-22

```


Number of trading days:

```

5381

```


Input features:

```

Open
High
Low
Close

```


The input sequence is constructed using a sliding window:

```

Window length = 30 trading days

```


Prediction target:

```

Next trading day's Closing Price

```


The forecasting task can be formulated as:

Given historical OHLC information:

\[
X_t=
[
Open_t,
High_t,
Low_t,
Close_t
]
\]


predict:

\[
\hat{Close}_{t+1}
\]


---

# 3. Model Architecture


The proposed MDTC-TA consists of four main components.


## 3.1 Multi-Scale Dilated Temporal Convolutional Network


Multiple TCN branches are used to extract temporal features with different receptive fields.


The dilation rates are:


\[
d=\{1,2,4\}
\]


Each branch contains a temporal convolution block:


\[
F_i=TCN_i(X)
\]


where different dilation factors capture different temporal dependencies.


The extracted multi-scale features are:


\[
F=
[F_1,F_2,F_3]
\]


The motivation is that financial markets contain:

- short-term volatility,
- medium-term movements,
- long-term trends.


---

## 3.2 Temporal Attention Mechanism


A temporal attention module is introduced after multi-scale feature extraction.


The attention weight is calculated by:


\[
A=\sigma(Conv_1(F))
\]


where:

- \(Conv_1\) represents a one-dimensional convolution,
- \(\sigma\) denotes sigmoid activation.


The weighted feature representation is:


\[
F'=F\times A
\]


The temporal attention mechanism enables the model to automatically emphasize important temporal information.


---

## 3.3 Feature Transformer Module


The multi-scale features are concatenated:


\[
F_c=
Concat(F_1,F_2,F_3)
\]


Then a fully connected Transformer layer is applied:


\[
F_f=
ReLU(WF_c+b)
\]


The transformer module integrates information from different temporal scales.


---

## 3.4 Residual Prediction Strategy


Instead of directly predicting the absolute price:


\[
\hat{y}_{t+1}
\]


the model predicts the price change:


\[
\Delta y
\]


Then the final prediction is:


\[
\hat{y}_{t+1}
=
y_t+\Delta y
\]


where:

- \(y_t\) is the current closing price,
- \(\Delta y\) is the predicted residual variation.


This strategy reduces the difficulty caused by the non-stationary characteristics of financial prices.


---

# 4. Experimental Environment


## Hardware

Example:

```

CPU: Intel/AMD processor
GPU: Optional CUDA GPU
RAM: >=8GB

```


## Software


Python:

```

Python >= 3.9

```


Deep learning framework:

```

PyTorch

```


Operating systems:

```

Windows / Linux

```


---

# 5. Project Structure


```

MDTC-TA/

│
├── README.md
├── requirements.txt
│
├── train_mdtc_ta.py
│
├── train_ablation.py
│
├── models/
│   │
│   ├── mdtc_ta.py
│   └── mdtc_ta_ablation.py
│
├── utils/
│   │
│   └── dataset.py
│
├── data/
│   │
│   └── raw/
│       └── raw_SP500.csv
│
├── results/
│
│   ├── experiment_summary.csv
│   ├── ablation_metrics.csv
│   │
│   ├── weights/
│   │
│   └── ablation/
│
└── draw_all_figures.py
````


---

# 6. Data Processing


The preprocessing pipeline contains:


## 6.1 Feature Selection


Only OHLC features are used:


```text
Open
High
Low
Close
````

## 6.2 Normalization

RobustScaler is applied:

[
X'=
\frac{x-Q_2}{Q_3-Q_1}
]

where:

* (Q_1) is the first quartile,
* (Q_2) is the median,
* (Q_3) is the third quartile.

RobustScaler is selected because financial prices contain abnormal fluctuations.

---

## 6.3 Dataset Split

The dataset is divided chronologically:

```
Training set : 70%

Validation set : 20%

Test set : 10%

```

No random shuffling is applied during splitting to avoid future information leakage.

---

# 7. Training Configuration

Main model:

```
Model:
MDTC-TA


Input dimension:
4


Hidden dimension:
64


Window size:
30


Batch size:
32

```

Optimizer:

```
AdamW
```

Learning rate:

```
1e-4
```

Weight decay:

```
1e-4
```

Learning rate scheduler:

```
CosineAnnealingLR
```

Loss function:

[
L=
Huber(y,\hat{y})
]

Early stopping:

```
patience = 20
```

Maximum epochs:

```
100
```

Random seed:

```
42
```

---

# 8. Training

Run:

```bash
python train_mdtc_ta.py
```

The program will:

1. Load S&P500 data

2. Generate sliding windows

3. Train MDTC-TA

4. Save the best model

5. Evaluate test performance

Generated files:

```
results/weights/

MDTC_TA_best.pth

MDTC_TA_pred.npy

MDTC_TA_true.npy

MDTC_TA_loss.npy

```

---

# 9. Ablation Experiments

To verify the contribution of each module:

Run:

```bash
python train_ablation.py
```

The following variants are evaluated:

| Model          | Description                       |
|----------------|-----------------------------------|
| Full           | Complete MDTC-TA                  |
| wo_Attention   | Remove Temporal Attention         |
| wo_MultiScale  | Remove Multi-scale structure      |
| wo_Transformer | Remove Feature Transformer module |

The results are saved:

```
results/ablation_metrics.csv

```

---

# 10. Evaluation Metrics

The model is evaluated using:

## Mean Absolute Error (MAE)

[
MAE=
\frac1N
\sum |y_i-\hat y_i|
]

## Root Mean Square Error (RMSE)

[
RMSE=
\sqrt{
\frac1N
\sum(y_i-\hat y_i)^2
}
]

## Mean Absolute Percentage Error (MAPE)

[
MAPE=
\frac1N
\sum
\left|
\frac{y_i-\hat y_i}{y_i}
\right|
\times100%
]

## Coefficient of Determination (R²)

[
R^2
===

1-
\frac{
\sum(y_i-\hat y_i)^2
}{
\sum(y_i-\bar y)^2
}
]

## Directional Accuracy (DA)

[
DA=
\frac{
correct\ direction
}{
N
}
\times100%
]

---

# 11. Visualization

Generate experimental figures:

```bash
python draw_all_figures.py
```

The visualization includes:

```
Prediction Curve

Training Loss Curve

Residual Analysis

Prediction Scatter Plot

Extreme Event Analysis

error_boxplot

blation_prediction_comparison

prediction_error_curve

candlestick

```

These figures are used for paper experimental analysis.

---

# 12. Main Experimental Result

The final MDTC-TA model achieves:

| Metric | Value   |
| ------ | ------- |
| MAE    | 5.4103  |
| RMSE   | 6.9257  |
| MAPE   | 0.4324% |
| R²     | 0.9897  |
| DA     | 60.31%  |

---

# 13. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the proposed model:

```bash
python train_mdtc_ta.py
```

Run ablation experiments:

```bash
python train_ablation.py
```

Generate figures:

```bash
python draw_all_figures.py
```

---

# 14. Citation

If you use this code, please cite:

```
MDTC-TA:
Stock Index Closing Price Forecasting Based on
Multi-Scale Dilated Temporal Convolutional Network with Temporal Attention
```

---

# 15. Notes

This repository is developed for academic research purposes.

The model is specifically designed for stock index closing price forecasting and does not represent financial investment advice.

```

---



