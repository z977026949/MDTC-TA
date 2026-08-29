import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm_baseline import LSTMBaseline
from models.bilstm_baseline import BiLSTMBaseline
from models.cnn_baseline import CNN1DBaseline
from models.tcn_baseline import TCNBaseline
from models.transformer_baseline import TransformerBaseline

from utils.dataset import get_financial_data_loaders

# =====================================================
# Seed
# =====================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================
# Metrics
# =====================================================

def calculate_metrics(true,pred,last_close):

    mae=np.mean(np.abs(true-pred))

    rmse=np.sqrt(np.mean((true-pred)**2))

    mape=np.mean(
        np.abs(
            (true-pred)/(true+1e-8)
        )
    )*100


    ss_res=np.sum(
        (true-pred)**2
    )

    ss_tot=np.sum(
        (true-np.mean(true))**2
    )

    r2=1-ss_res/(ss_tot+1e-8)


    true_direction=np.sign(
        true-last_close
    )

    pred_direction=np.sign(
        pred-last_close
    )


    da=np.mean(
        true_direction==pred_direction
    )*100

    return {
        "MAE":mae,
        "RMSE":rmse,
        "MAPE":mape,
        "R2":r2,
        "DA":da
    }

def print_metrics(name,metrics):

    print("\n==============================")
    print(name)
    print("==============================")


    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"MAPE : {metrics['MAPE']:.4f}%"
    )

    print(
        f"R2   : {metrics['R2']:.4f}"
    )

    print(
        f"DA   : {metrics['DA']:.2f}%"
    )

# =====================================================
# Save
# =====================================================

def save_metrics(name,metrics):

    os.makedirs(
        "results",
        exist_ok=True
    )

    path="results/experiment_summary.csv"


    df_new=pd.DataFrame({

        "Model":[name],

        "MAE":[metrics["MAE"]],

        "RMSE":[metrics["RMSE"]],

        "MAPE":[metrics["MAPE"]],

        "R2":[metrics["R2"]],

        "DA":[metrics["DA"]]

    })

    if os.path.exists(path):

        df_old=pd.read_csv(path)

        df_old=df_old[
            df_old["Model"]!=name
        ]

        df=pd.concat(
            [
                df_old,
                df_new
            ],
            ignore_index=True
        )

    else:

        df=df_new

    df.to_csv(
        path,
        index=False
    )
# =====================================================
# Train
# =====================================================

def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device
):

    model.train()

    total_loss=0


    for X,y in loader:


        X=X.to(device)

        y=y.to(device)


        optimizer.zero_grad()

        pred=model(X)

        if isinstance(pred,tuple):
            pred=pred[0]

        loss=criterion(
            pred,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss+=loss.item()*X.size(0)

    return total_loss/len(loader.dataset)

# =====================================================
# Evaluate
# =====================================================

def evaluate(
        model,
        loader,
        criterion,
        device
):

    model.eval()

    preds=[]

    trues=[]

    total_loss=0

    with torch.no_grad():

        for X,y in loader:

            X=X.to(device)

            y=y.to(device)

            pred=model(X)

            if isinstance(pred,tuple):

                pred=pred[0]

            loss=criterion(
                pred,
                y
            )

            total_loss+=loss.item()*X.size(0)

            preds.append(
                pred.cpu().numpy()
            )

            trues.append(
                y.cpu().numpy()
            )

    return (

        total_loss/len(loader.dataset),

        np.concatenate(preds),

        np.concatenate(trues)

    )

# =====================================================
# Run Model
# =====================================================

def run_model(
        name,
        model,
        train_loader,
        val_loader,
        test_loader,
        y_scaler,
        test_closes,
        device
):

    print("\n==============================")
    print(name)
    print("==============================")

    model=model.to(device)

    optimizer=optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler=optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )

    criterion=nn.HuberLoss(
        delta=1.0
    )

    best_loss=float("inf")
    patience=20
    wait=0

    os.makedirs(
        "results/weights",
        exist_ok=True
    )

    save_path=f"results/weights/{name}.pth"

    loss_history=[]

    for epoch in range(100):

        train_loss=train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss,_,_=evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        loss_history.append(
            train_loss
        )

        print(
            f"Epoch {epoch+1}/100 "
            f"Train:{train_loss:.6f} "
            f"Val:{val_loss:.6f}"
        )

        if val_loss<best_loss:

            best_loss=val_loss

            wait=0

            torch.save(
                model.state_dict(),
                save_path
            )

        else:

            wait+=1


            if wait>=patience:

                print(
                    "Early stopping"
                )

                break

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device
        )
    )

    _,pred,true=evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    pred=pred.reshape(-1,1)

    true=true.reshape(-1,1)

    # ============================
    # 反归一化
    # ============================

    pred_price=y_scaler.inverse_transform(
        pred
    ).flatten()

    true_price=y_scaler.inverse_transform(
        true
    ).flatten()

    # 次日涨跌方向

    #last_close=test_closes[-len(true_price)-1:-1]
    last_close = test_closes[:len(true_price)]

    metrics=calculate_metrics(
        true_price,
        pred_price,
        last_close
    )

    print_metrics(
        name,
        metrics
    )

    save_metrics(
        name,
        metrics
    )

    np.save(
        f"results/weights/{name}_pred.npy",
        pred_price
    )

    np.save(
        f"results/weights/{name}_true.npy",
        true_price
    )

    np.save(
        f"results/weights/{name}_loss.npy",
        np.array(loss_history)
    )

# =====================================================
# Main
# =====================================================

def main():

    seed_everything(42)

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device
    )

    train_loader,val_loader,test_loader,x_scaler,y_scaler,test_closes=get_financial_data_loaders(

        "data/raw/raw_SP500.csv",

        window=30,

        batch_size=32

    )

    input_dim=4

    hidden=64

    models={

        "LSTM":

        LSTMBaseline(
            input_dim=input_dim,
            hidden_dim=hidden
        ),

        "BiLSTM":

        BiLSTMBaseline(
            input_dim=input_dim,
            hidden_dim=hidden
        ),


        "CNN1D":

        CNN1DBaseline(
            input_dim=input_dim,
            hidden=hidden
        ),

        "TCN":

        TCNBaseline(
            input_dim=input_dim,
            hidden=hidden
        ),



        "Transformer":

        TransformerBaseline(
            input_dim=input_dim,
            d_model=hidden,
            nhead=4,
            layers=2
        )

    }

    for name,model in models.items():


        run_model(

            name,

            model,

            train_loader,

            val_loader,

            test_loader,

            y_scaler,

            test_closes,

            device

        )

if __name__=="__main__":

    main()