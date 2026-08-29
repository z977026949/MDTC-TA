import matplotlib.pyplot as plt
import numpy as np


def plot_all_models_comparison(true_price, models_preds_dict):
    """
    true_price: 真实绝对价格数组
    models_preds_dict: 字典，格式为 {"LSTM": pred1, "TCN": pred2, "MSTCN_GA": pred_my}
    """
    plt.figure(figsize=(14, 6), dpi=300)

    # 1. 真实值用醒目的黑色粗虚线，放在底层或半透明
    plt.plot(true_price, label="True Close", color="black", linewidth=2.5, linestyle="--", alpha=0.8)

    # 2. 各个 Baseline 用常规颜色
    colors = {"LSTM": "blue", "BiLSTM": "orange", "CNN1D": "green", "TCN": "cyan", "Transformer": "purple"}

    for name, pred in models_preds_dict.items():
        if name == "MSTCN_GA":
            # 重点：你的主模型用最醒目的亮红色、较粗的实线突出显示，防止被盖住！
            plt.plot(pred, label=f"Proposed ({name})", color="red", linewidth=2.0, alpha=0.9)
        else:
            color = colors.get(name, "gray")
            plt.plot(pred, label=name, color=color, linewidth=1.2, alpha=0.6)

    plt.title("Unified Comparison of All Models vs. True Market Prices", fontsize=14, fontweight="bold")
    plt.xlabel("Time Steps (Test Set)", fontsize=12)
    plt.ylabel("Close Price ($)", fontsize=12)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/model_comparison_fixed.png")
    plt.show()