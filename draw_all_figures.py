import os
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Global
# ==============================

plt.rcParams["figure.figsize"] = (12,5)
plt.rcParams["font.size"] = 12


WEIGHT_DIR = "results/weights"
ABLATION_DIR = "results/ablation"

SAVE_DIR = "results/figures"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



# ==============================
# Load Ours
# ==============================

pred = np.load(
    f"{WEIGHT_DIR}/MDTC_TA_pred.npy"
)


true = np.load(
    f"{WEIGHT_DIR}/MDTC_TA_true.npy"
)


loss = np.load(
    f"{WEIGHT_DIR}/MDTC_TA_loss.npy"
)



# =====================================================
# 01 Ours prediction
# =====================================================

def draw_01():

    plt.figure(figsize=(12,5))


    plt.plot(
        true,
        "--",
        linewidth=1.8,
        label="Actual"
    )


    plt.plot(
        pred,
        linewidth=1.5,
        label="Ours"
    )


    plt.xlabel(
        "Trading Days"
    )

    plt.ylabel(
        "Close Price"
    )


#  plt.title("Prediction Performance of MSTCN-Attention")


    plt.legend()

    plt.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        f"{SAVE_DIR}/01_Ours_prediction.png",
        dpi=300
    )


    plt.close()



# =====================================================
# 02 Local prediction
# =====================================================

def draw_02():

    start=200
    end=400


    plt.figure(figsize=(12,5))


    plt.plot(
        true[start:end],
        "--",
        linewidth=2,
        label="Actual"
    )


    plt.plot(
        pred[start:end],
        linewidth=2,
        label="Ours"
    )


    plt.xlabel(
        "Trading Days"
    )

    plt.ylabel(
        "Close Price"
    )


#    plt.title( "Local Prediction Comparison")


    plt.legend()

    plt.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        f"{SAVE_DIR}/02_Local_prediction.png",
        dpi=300
    )

    plt.close()




# =====================================================
# 03 Loss
# =====================================================

def draw_03():


    plt.figure(figsize=(8,5))


    plt.plot(
        loss,
        linewidth=2
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Huber Loss"
    )


#   plt.title( "Training Convergence")


    plt.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        f"{SAVE_DIR}/03_training_loss.png",
        dpi=300
    )


    plt.close()




# =====================================================
# 04 Residual Analysis
# =====================================================

def draw_04():


    residual = true - pred



    fig, axs = plt.subplots(

        2,

        1,

        figsize=(12,8)

    )


    # ==========================
    # Residual curve
    # ==========================


    axs[0].plot(

        residual,

        linewidth=1.5,

        label="Residual"

    )


    axs[0].axhline(

        0,

        linestyle="--",

        linewidth=1.5

    )


#  axs[0].set_title( "Prediction Residual Curve")


    axs[0].set_xlabel(

        "Trading Days"

    )


    axs[0].set_ylabel(

        "Residual"

    )


    axs[0].grid(

        alpha=0.3

    )



    # ==========================
    # Residual histogram
    # ==========================


    axs[1].hist(

        residual,

        bins=40,

    )


#  axs[1].set_title( "Residual Distribution")


    axs[1].set_xlabel(

        "Residual"

    )


    axs[1].set_ylabel(

        "Frequency"

    )


    axs[1].grid(

        alpha=0.3

    )



    plt.tight_layout()



    plt.savefig(

        f"{SAVE_DIR}/04_residual_analysis.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



# =====================================================
# 05 Scatter
# =====================================================

def draw_05():


    r2=1-np.sum(
        (true-pred)**2
    )/np.sum(
        (true-np.mean(true))**2
    )


    plt.figure(figsize=(6,6))


    plt.scatter(
        true,
        pred,
        s=12,
        alpha=0.5
    )


    min_v=min(
        true.min(),
        pred.min()
    )

    max_v=max(
        true.max(),
        pred.max()
    )


    plt.plot(
        [min_v,max_v],
        [min_v,max_v],
        "--"
    )


    plt.xlabel(
        "Actual"
    )

    plt.ylabel(
        "Prediction"
    )


# plt.title(f"Prediction Scatter (R²={r2:.4f})")


    plt.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        f"{SAVE_DIR}/05_prediction_scatter.png",
        dpi=300
    )


    plt.close()




# =====================================================
# 06 Extreme events
# =====================================================

def draw_06():

    events={

        "2008 Financial Crisis":
        (0,80),

        "2020 COVID-19 Crash":
        (250,330),

        "2022 Bear Market":
        (400,480)

    }


    fig,axs=plt.subplots(
        3,
        1,
        figsize=(12,12)
    )


    for ax,(name,(s,e)) in zip(
        axs,
        events.items()
    ):


        ax.plot(
            true[s:e],
            "--",
            label="Actual"
        )


        ax.plot(
            pred[s:e],
            label="Ours"
        )


        ax.set_title(
            name
        )


        ax.legend()

        ax.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        f"{SAVE_DIR}/06_extreme_event_analysis.png",
        dpi=300
    )


    plt.close()




# =====================================================
# 07 Model Error Boxplot
# =====================================================

def draw_07_error_boxplot():


    models = [

        "LSTM",

        "TCN",

        "Transformer",

        "Ours"

    ]


    errors=[]


    # -----------------------------
    # Load prediction error
    # -----------------------------

    files=[

        "LSTM_pred.npy",

        "TCN_pred.npy",

        "Transformer_pred.npy",

        "MDTC_TA_pred.npy"

    ]


    for file in files:


        p=np.load(
            f"{WEIGHT_DIR}/{file}"
        )


        t=true[:len(p)]


        error=np.abs(
            t-p
        )


        errors.append(error)



    # -----------------------------
    # Plot
    # -----------------------------

    plt.figure(
        figsize=(9,5)
    )


    box=plt.boxplot(

        errors,

        tick_labels=models,

        showfliers=False,

        patch_artist=True

    )


    # -----------------------------
    # Colors
    # -----------------------------

    colors=[

        "#6FA8DC",   # LSTM blue

        "#F6B26B",   # TCN orange

        "#93C47D",   # Transformer green

        "#E57373"    # MSTCN red

    ]


    for patch,color in zip(

        box["boxes"],

        colors

    ):

        patch.set_facecolor(
            color
        )

        patch.set_alpha(
            0.75
        )



    # median line

    for median in box["medians"]:

        median.set_color(
            "black"
        )

        median.set_linewidth(
            1.5
        )



    plt.ylabel(

        "Absolute Prediction Error"

    )


    plt.xlabel(

        "Models"

    )


#   plt.title( "Prediction Error Distribution")


    plt.grid(

        axis="y",

        alpha=0.3

    )


    plt.tight_layout()



    plt.savefig(

        f"{SAVE_DIR}/07_model_error_boxplot.png",

        dpi=300

    )


    plt.close()



# =====================================================
# 08 Ablation Prediction Comparison (2x2)
# =====================================================

def draw_08_ablation_prediction_comparison():


    path = "results/ablation"


    models = [

        "wo_Attention",

        "wo_MultiScale",

        "wo_Transformation",

        "Full"

    ]


    titles = [

        "w/o Temporal Attention",

        "w/o Multi-Scale TCN",

        "w/o Feature Transformation",

        "Ours"

    ]


    colors = [

        "#6FA8DC",

        "#F6B26B",

        "#93C47D",

        "#E57373"

    ]



    true = np.load(

        f"{path}/Full_true.npy"

    )



    start = 200

    end = 400



    fig,axs = plt.subplots(

        2,

        2,

        figsize=(14,8)

    )


    axs = axs.flatten()



    for ax,model,title,color in zip(

            axs,

            models,

            titles,

            colors

    ):


        pred=np.load(

            f"{path}/{model}_pred.npy"

        )


        # Actual

        ax.plot(

            true[start:end],

            "--",

            color="black",

            linewidth=1.8,

            label="Actual"

        )


        # Prediction

        ax.plot(

            pred[start:end],

            color=color,

            linewidth=1.8,

            label=title

        )


        ax.set_title(

            title,

            fontsize=13

        )


        ax.set_xlabel(

            "Trading Days"

        )


        ax.set_ylabel(

            "Close Price"

        )


        ax.legend(

            fontsize=9

        )


        ax.grid(

            alpha=0.3

        )



#  plt.suptitle("Ablation Study Prediction Comparison",fontsize=16)


    plt.tight_layout(

        rect=[0,0,1,0.96]

    )


    plt.savefig(

        f"{SAVE_DIR}/08_ablation_prediction_comparison.png",

        dpi=300

    )


    plt.close()




# =====================================================
# 09 Prediction Error Curve
# =====================================================

def draw_09_prediction_error_curve():


    models = [

        "LSTM",
        "TCN",
        "Transformer",
        "Ours"

    ]


    files=[

        "LSTM_pred.npy",

        "TCN_pred.npy",

        "Transformer_pred.npy",

        "MDTC_TA_pred.npy"

    ]


    colors=[

        "#808080",   # LSTM gray

        "#808080",   # TCN gray

        "#808080",   # Transformer gray

        "#E53935"    # MDTC_TA red

    ]


    fig,axs = plt.subplots(

        2,
        2,

        figsize=(12,8)

    )


    axs=axs.flatten()


    for ax,model,file,color in zip(

            axs,

            models,

            files,

            colors

    ):


        p=np.load(

            f"{WEIGHT_DIR}/{file}"

        )


        t=true[:len(p)]


        error=np.abs(

            t-p

        )


        ax.plot(

            error,

            color=color,

            linewidth=1.5

        )


        ax.set_title(

            model

        )


        ax.set_xlabel(

            "Trading Days"

        )


        ax.set_ylabel(

            "Absolute Error"

        )


        ax.grid(

            alpha=0.3

        )


#    plt.suptitle( "Prediction Error Curve of Different Models", fontsize=16)


    plt.tight_layout()


    plt.savefig(

        f"{SAVE_DIR}/09_prediction_error_curve.png",

        dpi=300

    )


    plt.close()




# =====================================================
# 10 Candlestick Chart
# =====================================================

def draw_10_candlestick():

    import pandas as pd
    import mplfinance as mpf


    csv_path = "data/raw/raw_SP500.csv"



    # ==========================
    # Read csv
    # ==========================

    try:

        df = pd.read_csv(
            csv_path,
            encoding="utf-8"
        )

    except:

        df = pd.read_csv(
            csv_path,
            encoding="gbk"
        )



    print(
        "Original columns:",
        df.columns.tolist()
    )



    # 去除空格

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]



    # ==========================
    # Date
    # ==========================

    df["Date"] = pd.to_datetime(
        df["日期"]
    )


    df = df.set_index(
        "Date"
    )



    # ==========================
    # Rename OHLC
    # ==========================

    rename_dict={}



    for c in df.columns:

        name=str(c).strip().lower()


        # Open

        if (
            "开盘" in name
            or "open" in name
        ):

            rename_dict[c]="Open"


        # High

        elif (
            "最高" in name
            or name=="高"
            or "high" in name
        ):

            rename_dict[c]="High"



        # Low

        elif (
            "最低" in name
            or name=="低"
            or "low" in name
        ):

            rename_dict[c]="Low"



        # Close

        elif (
            "收盘" in name
            or "close" in name
        ):

            rename_dict[c]="Close"



    df=df.rename(
        columns=rename_dict
    )


    print(
        "Renamed columns:",
        df.columns.tolist()
    )



    # ==========================
    # Keep OHLC
    # ==========================

    df=df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ]



    # ==========================
    # Remove comma
    # same as dataset.py
    # ==========================

    for c in df.columns:

        df[c]=(
            df[c]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False
            )
            .astype(float)
        )



    df.dropna(
        inplace=True
    )



    # ==========================
    # Sort date
    # CSV is descending
    # ==========================

    df=df.sort_index()



    print(
        df.head()
    )

    print(
        df.tail()
    )



    # ==========================
    # Test period
    # same as dataset.py
    # ==========================

    n=len(df)


    test_start=int(
        n*0.9
    )


    test=df.iloc[
        test_start:
    ]



    # only show last 300 days

    test=test.iloc[
        :300
    ]



    print(
        "K line data shape:",
        test.shape
    )



    # ==========================
    # Style
    # ==========================

    mc = mpf.make_marketcolors(

        up="red",

        down="green",

        edge="inherit",

        wick="inherit"

    )


    style = mpf.make_mpf_style(

        marketcolors=mc,

        gridstyle="--",

        gridcolor="lightgray",

        facecolor="white",

        figcolor="white"

    )



    # ==========================
    # Plot
    # ==========================


    mpf.plot(

        test,

        type="candle",

        style=style,

        volume=False,

        figsize=(12,5),

        title="",

        ylabel="Close Price",

        savefig=
        f"{SAVE_DIR}/10_candlestick_chart.png",

        tight_layout=True

    )




# =====================================================
# Main
# =====================================================

if __name__=="__main__":


    draw_01()

    draw_02()

    draw_03()

    draw_04()

    draw_05()

    draw_06()

    draw_07_error_boxplot()

    draw_08_ablation_prediction_comparison()

    draw_09_prediction_error_curve()

    draw_10_candlestick()


    print(
        "All figures generated."
    )