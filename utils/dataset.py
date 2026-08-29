import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import RobustScaler

class StockDataset(Dataset):

    def __init__(self,X,y):

        self.X=torch.tensor(X,dtype=torch.float32)

        self.y=torch.tensor(y,dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self,idx):

        return self.X[idx],self.y[idx]

def create_features(csv_path):

    try:
        df=pd.read_csv(csv_path,encoding="utf-8")
    except:
        df=pd.read_csv(csv_path,encoding="gbk")

    df.columns=[str(c).strip() for c in df.columns]

    open_col=None
    high_col=None
    low_col=None
    close_col=None

    for c in df.columns:

        name=str(c).lower().strip()

        if "open" in name or "开盘" in name:
            open_col=c

        elif "high" in name or "最高" in name or name=="高":
            high_col=c

        elif "low" in name or "最低" in name or name=="低":
            low_col=c

        elif "close" in name or "收盘" in name:
            close_col=c

    if None in [open_col,high_col,low_col,close_col]:

        print(df.columns.tolist())

        raise ValueError("OHLC列识别失败")

    price=df[
        [
            open_col,
            high_col,
            low_col,
            close_col
        ]
    ].copy()


    price.columns=[
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for c in price.columns:

        price[c]=(
            price[c]
            .astype(str)
            .str.replace(",","",regex=False)
            .astype(float)
        )


    price.dropna(inplace=True)


    return price

def get_financial_data_loaders(
        csv_path,
        window=30,
        batch_size=32
):

    df=create_features(csv_path)

    print("数据维度:",df.shape)

    print(df.head())

    X=df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ].values

    y=df["Close"].shift(-1).values.reshape(-1,1)

    # 删除最后一天无标签数据

    X=X[:-1]

    y=y[:-1]

    n=len(X)

    train_end=int(n*0.7)

    val_end=int(n*0.9)

    X_train=X[:train_end]

    X_val=X[train_end:val_end]

    X_test=X[val_end:]


    y_train=y[:train_end]

    y_val=y[train_end:val_end]

    y_test=y[val_end:]

    x_scaler=RobustScaler()


    X_train=x_scaler.fit_transform(X_train)

    X_val=x_scaler.transform(X_val)

    X_test=x_scaler.transform(X_test)

    y_scaler=RobustScaler()


    y_train=y_scaler.fit_transform(y_train)

    y_val=y_scaler.transform(y_val)

    y_test=y_scaler.transform(y_test)

    def make_window(X,y):

        Xs=[]

        ys=[]

        for i in range(len(X)-window+1):

            Xs.append(
                X[i:i+window]
            )

            ys.append(
                y[i+window-1]
            )


        return np.array(Xs),np.array(ys)

    X_train,y_train=make_window(
        X_train,
        y_train
    )

    X_val,y_val=make_window(
        X_val,
        y_val
    )


    X_test,y_test=make_window(
        X_test,
        y_test
    )

    print("训练输入:",X_train.shape)

    print("训练目标:",y_train.shape)

    train_loader=DataLoader(
        StockDataset(X_train,y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader=DataLoader(
        StockDataset(X_val,y_val),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader=DataLoader(
        StockDataset(X_test,y_test),
        batch_size=batch_size,
        shuffle=False
    )

   # test_closes = df["Close"].values[val_end + window:]
    test_closes = df["Close"].values[val_end + window - 1:-1]

    return (
        train_loader,
        val_loader,
        test_loader,
        x_scaler,
        y_scaler,
        test_closes
    )