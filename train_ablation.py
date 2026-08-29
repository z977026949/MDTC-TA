import os
import random
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim


from models.mdtc_ta_ablation import MDTCTAAblation

from utils.dataset import get_financial_data_loaders



# ======================================================
# Seed
# ======================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

# ======================================================
# Metrics
# ======================================================

def calculate_metrics(
        true,
        pred,
        last_close
):

    mae=np.mean(
        np.abs(true-pred)
    )

    rmse=np.sqrt(
        np.mean(
            (true-pred)**2
        )
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

# ======================================================
# Train
# ======================================================

def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device
):

    model.train()

    total=0

    for X,y in loader:

        X=X.to(device)

        y=y.to(device)

        optimizer.zero_grad()

        pred, _, _ = model(X)

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

        total += loss.item()*X.size(0)

    return total/len(loader.dataset)

# ======================================================
# Evaluate
# ======================================================

def evaluate(
        model,
        loader,
        criterion,
        device
):

    model.eval()

    preds=[]

    trues=[]

    total=0

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

# ======================================================
# Single Model
# ======================================================

def run_model(
        name,
        mode,
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

    model=MDTCTAAblation(

        mode=mode,

        input_dim=4,

        hidden=64,

        dropout=0.1

    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable Parameters:",
        params
    )

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

    wait=0

    best_epoch = 0

    safe_name = (
        name
        .replace("/", "_")
        .replace(" ", "_")
        .replace("\\", "_")
    )

    save_dir = "results/ablation"

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    save_path = f"{save_dir}/{safe_name}.pth"

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

        if val_loss<best:

            best_epoch = epoch + 1

            best=val_loss

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
    print(
         f"Best Epoch: {best_epoch}, Best Val Loss: {best:.6f}"
                )

    # load best

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

    pred_price=y_scaler.inverse_transform(
        pred
    ).flatten()

    true_price=y_scaler.inverse_transform(
        true
    ).flatten()

    last_close=test_closes[:len(true_price)]

    # 保存预测

    np.save(
        f"results/ablation/{safe_name}_pred.npy",
        pred_price
    )

    np.save(
        f"results/ablation/{safe_name}_true.npy",
        true_price
    )

    np.save(
        f"results/ablation/{safe_name}_loss.npy",
        np.array(loss_history)
    )

    metrics = calculate_metrics(

        true_price,

        pred_price,

        last_close

    )

    metrics["BestEpoch"] = best_epoch

    metrics["BestValLoss"] = best

    print(metrics)

    return metrics

# ======================================================
# Main
# ======================================================

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

    train_loader,val_loader,test_loader,x_scaler,y_scaler,test_closes = get_financial_data_loaders(

        "data/raw/raw_SP500.csv",

        window=30,

        batch_size=32

    )

    experiments = {

        "Full":
            "full",

        "wo_Attention":
            "no_attention",

        "wo_MultiScale":
            "single_scale",

        "wo_Transformation":
            "no_transformation"

    }

    results=[]

    for name,mode in experiments.items():

        metrics=run_model(

            name,

            mode,

            train_loader,

            val_loader,

            test_loader,

            y_scaler,

            test_closes,

            device

        )

        results.append(

            {

                "Model":name,

                **metrics

            }

        )

    df=pd.DataFrame(results)

    os.makedirs(

        "results",

        exist_ok=True

    )

    df.to_csv(

        "results/ablation_metrics.csv",

        index=False

    )

    print("\n消融实验完成")

    print(df)

if __name__=="__main__":

    main()