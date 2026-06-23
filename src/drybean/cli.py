import json

import pandas as pd
import typer

from drybean.config import PATHS, ensure_artifact_dirs
from drybean.data import load_splits, profile_frame
from drybean.experiments.ablation import run_feature_ablation
from drybean.experiments.baseline import run_baseline_experiment
from drybean.experiments.robustness import run_robustness_experiment
from drybean.plots import (
    plot_ablation,
    plot_confusion_and_importance,
    plot_loss_history,
    plot_model_comparison,
    plot_robustness,
)

app = typer.Typer(help="Dry Bean 多分类实验命令行")


@app.command()
def analyze():
    """生成三份数据的数据质量画像。"""
    ensure_artifact_dirs()
    splits = load_splits()
    report = {
        name: profile_frame(getattr(splits, name))
        for name in ("train", "val", "test")
    }
    (PATHS.results / "data_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo("已生成 data_profile.json")


@app.command()
def train():
    """训练五种算法并生成最终测试结果。"""
    ensure_artifact_dirs()
    run_baseline_experiment(engineered=True)


@app.command()
def ablation():
    """比较原始特征与衍生特征。"""
    ensure_artifact_dirs()
    run_feature_ablation()


@app.command()
def robustness():
    """执行四类多强度训练噪声实验。"""
    ensure_artifact_dirs()
    run_robustness_experiment()


@app.command()
def plot():
    """从离线结果生成全部高清图表。"""
    ensure_artifact_dirs()
    plot_model_comparison(
        pd.read_csv(PATHS.results / "model_comparison.csv")
    )
    loss_path = PATHS.results / "loss_history.csv"
    if loss_path.exists() and loss_path.stat().st_size:
        loss = pd.read_csv(loss_path)
        if not loss.empty:
            plot_loss_history(loss)
    plot_robustness(pd.read_csv(PATHS.results / "robustness.csv"))
    plot_ablation(pd.read_csv(PATHS.results / "feature_ablation.csv"))
    plot_confusion_and_importance()


@app.command(name="all")
def run_all():
    """运行数据分析、训练、消融、鲁棒性和绘图全流程。"""
    analyze()
    train()
    ablation()
    robustness()
    plot()


if __name__ == "__main__":
    app()

