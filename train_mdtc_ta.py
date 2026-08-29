import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models.mdtc_ta import MDTCTA
from utils.dataset import get_financial_data_loaders


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(true,pred,last_close):

    mae=np.mean(
        np.abs(true-pred)
    )

    rmse=np.sqrt(
        np.mean((true-pred)**2)
    )

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

def print_metrics(m):

    print("\n================ MDTC_TA ================")

    print(f"MAE  : {m['MAE']:.4f}")

    print(f"RMSE : {m['RMSE']:.4f}")

    print(f"MAPE : {m['MAPE']:.4f}%")

    print(f"R2   : {m['R2']:.4f}")

    print(f"DA   : {m['DA']:.2f}%")

    print("==================================================")

def save_metrics(m):

    os.makedirs(
        "results",
        exist_ok=True
    )

    path="results/experiment_summary.csv"

    row=pd.DataFrame([{
        "Model":"MDTC_TA",
        **m
    }])

    if os.path.exists(path):

        old=pd.read_csv(path)

        old=old[
            old.Model!="MDTC_TA"
        ]

        row=pd.concat(
            [old,row],
            ignore_index=True
        )

    row.to_csv(
        path,
        index=False
    )

def train_epoch(model,loader,optimizer,criterion,device):

    model.train()

    total=0

    for X,y in loader:

        X=X.to(device)

        y=y.to(device)

        optimizer.zero_grad()

        pred,_,_=model(X)

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

        total+=loss.item()*X.size(0)

    return total/len(loader.dataset)

def evaluate(model,loader,criterion,device):

    model.eval()

    total=0

    preds=[]

    trues=[]

    with torch.no_grad():

        for X,y in loader:

            X=X.to(device)

            y=y.to(device)

            pred,_,_=model(X)

            loss=criterion(
                pred,
                y
            )

            total+=loss.item()*X.size(0)


            preds.append(
                pred.cpu().numpy()
            )

            trues.append(
                y.cpu().numpy()
            )

    return (
        total/len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues)
    )

def run():

    seed_everything()

    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
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

    model=MDTCTA(
        input_dim=4,
        hidden=64
    ).to(device)

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

    best=float("inf")
    patience=20
    counter=0

    loss_history = []

    os.makedirs(
        "results/weights",
        exist_ok=True
    )

    save_path="results/weights/MDTC_TA_best.pth"

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

        print(
            f"Epoch {epoch+1}/100 Train:{train_loss:.6f} Val:{val_loss:.6f}"
        )

        loss_history.append(train_loss)

        if val_loss<best:

            best=val_loss

            counter=0

            torch.save(
                model.state_dict(),
                save_path
            )

        else:

            counter+=1

            if counter>=patience:

                print("Early stopping")

                break

    np.save(
        "results/weights/MDTC_TA_loss.npy",
        np.array(loss_history)
    )

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

    pred_price=y_scaler.inverse_transform(pred).flatten()

    true_price=y_scaler.inverse_transform(true).flatten()

   #last_close = test_closes[-len(true_price) - 1:-1]
    last_close = test_closes[:len(true_price)]

    metrics = calculate_metrics(
        true_price,
        pred_price,
        last_close
    )

    print_metrics(metrics)

    save_metrics(metrics)

    np.save(
        "results/weights/MDTC_TA_pred.npy",
        pred_price
    )

    np.save(
        "results/weights/MDTC_TA_true.npy",
        true_price
    )

if __name__=="__main__":

    run()