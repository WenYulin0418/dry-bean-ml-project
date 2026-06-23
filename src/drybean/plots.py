from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from drybean.config import PATHS, ensure_artifact_dirs

COLORS = ["#2C6E63", "#D28C45", "#4C78A8", "#A65D57", "#7568A6"]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, filename: str) -> Path:
    ensure_artifact_dirs()
    path = PATHS.figures / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_model_comparison(frame: pd.DataFrame) -> None:
    configure_style()
    ordered = frame.sort_values("test_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=ordered,
        x="test_accuracy",
        y="model",
        hue="model",
        palette=COLORS,
        legend=False,
        ax=ax,
    )
    ax.set(xlabel="测试集 Accuracy", ylabel="")
    save_figure(fig, "model_accuracy.png")

    dimensions = [
        ("train_seconds", "训练时间（秒）"),
        ("inference_ms_per_sample", "单样本推理（毫秒）"),
        ("model_size_mb", "模型大小（MB）"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (column, title) in zip(axes, dimensions):
        sns.barplot(
            data=frame.sort_values(column),
            x=column,
            y="model",
            color=COLORS[0],
            ax=ax,
        )
        ax.set(xlabel=title, ylabel="")
    save_figure(fig, "model_efficiency.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=frame.sort_values("accuracy_gap", ascending=False),
        x="accuracy_gap",
        y="model",
        color=COLORS[1],
        ax=ax,
    )
    ax.set(xlabel="训练 Accuracy - 测试 Accuracy", ylabel="")
    save_figure(fig, "overfitting_gap.png")


def plot_loss_history(frame: pd.DataFrame) -> None:
    configure_style()
    for model_key, subset in frame.groupby("model_key"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(subset["epoch"], subset["train_loss"], label="训练集")
        ax.plot(subset["epoch"], subset["val_loss"], label="验证集")
        ax.set(
            xlabel="迭代轮数",
            ylabel="Log Loss",
            title=f"{model_key} Loss 曲线",
        )
        ax.legend()
        save_figure(fig, f"loss_{model_key}.png")


def plot_robustness(frame: pd.DataFrame) -> None:
    configure_style()
    names = {
        "gaussian": "高斯噪声",
        "impulse": "脉冲噪声",
        "missing": "缺失噪声",
        "label": "标签噪声",
    }
    for noise_type, subset in frame.groupby("noise_type"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=subset,
            x="strength",
            y="accuracy_drop",
            hue="model",
            marker="o",
            ax=ax,
        )
        ax.set(
            xlabel="噪声强度",
            ylabel="Accuracy 下降",
            title=names[noise_type],
        )
        save_figure(fig, f"robustness_{noise_type}.png")


def plot_ablation(frame: pd.DataFrame) -> None:
    configure_style()
    display = frame.copy()
    display["特征方案"] = display["engineered_features"].map(
        {False: "仅原始特征", True: "加入衍生特征"}
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=display,
        x="test_accuracy",
        y="model",
        hue="特征方案",
        ax=ax,
    )
    ax.set(xlabel="测试集 Accuracy", ylabel="")
    save_figure(fig, "feature_ablation.png")


def plot_confusion_and_importance() -> None:
    configure_style()
    for path in PATHS.results.glob("confusion_*.csv"):
        matrix = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
        ax.set(xlabel="预测类别", ylabel="真实类别")
        save_figure(fig, f"{path.stem}.png")
    for path in PATHS.results.glob("feature_importance_*.csv"):
        importance = pd.read_csv(path).head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=importance,
            x="importance",
            y="feature",
            color=COLORS[0],
            ax=ax,
        )
        ax.set(xlabel="特征重要性", ylabel="")
        save_figure(fig, f"{path.stem}.png")

