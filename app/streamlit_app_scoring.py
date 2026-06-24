import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"

st.set_page_config(
    page_title="Dry Bean 多分类实验",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --forest: #174f38;
  --forest-soft: #e9f1ec;
  --paper: #fbfaf6;
  --ink: #17211c;
  --muted: #68736c;
  --line: #dce3dd;
  --ochre: #b77a2e;
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] { background: #f4f2eb; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stRadio label {
  padding: .52rem .7rem; border-radius: .55rem; margin-bottom: .18rem;
}
h1, h2, h3 { color: var(--forest); letter-spacing: -.02em; }
h1 { font-size: 2.3rem !important; margin-bottom: .3rem !important; }
.block-container { max-width: 1260px; padding-top: 2.2rem; padding-bottom: 4rem; }
[data-testid="stMetric"] {
  background: rgba(255,255,255,.75); border: 1px solid var(--line);
  border-radius: .75rem; padding: 1rem 1.1rem;
}
[data-testid="stMetricValue"] { color: var(--forest); font-weight: 720; }
.callout {
  border-left: 4px solid var(--forest); background: var(--forest-soft);
  padding: 1rem 1.1rem; border-radius: 0 .65rem .65rem 0; margin: .8rem 0 1.2rem;
}
.warning {
  border-left-color: var(--ochre); background: #f7efe3;
}
.small-note { color: var(--muted); font-size: .91rem; line-height: 1.65; }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: .65rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_results():
    required = [
        RESULTS / "data_profile.json",
        RESULTS / "model_comparison.csv",
        RESULTS / "robustness.csv",
        RESULTS / "feature_ablation.csv",
        RESULTS / "loss_history.csv",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少离线实验结果，请先执行 drybean all：" + ", ".join(missing)
        )
    return {
        "profile": json.loads(
            (RESULTS / "data_profile.json").read_text(encoding="utf-8")
        ),
        "comparison": pd.read_csv(RESULTS / "model_comparison.csv"),
        "robustness": pd.read_csv(RESULTS / "robustness.csv"),
        "ablation": pd.read_csv(RESULTS / "feature_ablation.csv"),
        "loss": pd.read_csv(RESULTS / "loss_history.csv"),
    }


def image(name: str, caption: str | None = None):
    path = FIGURES / name
    if not path.exists():
        path = ROOT / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


data = load_results()
comparison = data["comparison"].copy()
best = comparison.sort_values("test_accuracy", ascending=False).iloc[0]
train_profile = data["profile"]["train"]

pages = [
    "项目概览",
    "数据分析和数据处理",
    "测试精度对比",
    "Loss曲线对比",
    "算法推理速度对比",
    "鲁棒性对比",
    "过拟合分析",
    "特征消融分析",
]

with st.sidebar:
    st.markdown("## 🫘 Dry Bean")
    st.caption("机器学习期末工程")
    page = st.radio("导航", pages, label_visibility="collapsed")
    st.divider()
    st.markdown(
        """
<div class="small-note">
数据集：Dry Bean Dataset<br>
总样本：13,611<br>
输入特征：16<br>
真实类别：7<br>
比较算法：5
</div>
""",
        unsafe_allow_html=True,
    )

st.title("Dry Bean Dataset 分类分析与多模型对比")
st.caption("项目概览 · 数据分析和数据处理 · 精度对比 · Loss曲线 · 推理速度 · 鲁棒性 · 过拟合 · 特征消融")

if page == "项目概览":
    st.markdown(
        """
<div class="callout">
本项目使用预先划分 DryBeanDataset 文件夹中的训练集、验证集和测试集。所有预处理参数只在训练集拟合，
验证集用于模型选择，测试集仅用于最终评价，避免数据泄漏。
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总样本数", "13,611")
    c2.metric("输入特征", "16")
    c3.metric("类别数量", "7")
    c4.metric("比较算法", "5")
    c5.metric("最佳 Accuracy", percent(best["test_accuracy"]))

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("研究流程")
        st.markdown(
            """
1. 审计原始数据中的缺失、重复、单位字符串和标签污染  
2. 构建防泄漏的数据清洗与特征工程流水线  
3. 比较逻辑回归、KNN、随机森林、XGBoost 和高斯朴素贝叶斯  
4. 分析 Accuracy、Macro-F1、Loss、训练/推理效率与模型大小  
5. 注入四类训练噪声，评估鲁棒性与过拟合  
6. 通过特征消融验证衍生特征的真实收益
"""
        )
    with right:
        st.subheader("当前最佳模型")
        st.metric(best["model"], percent(best["test_accuracy"]))
        st.write(
            f"Macro-F1：**{percent(best['test_macro_f1'])}**  \n"
            f"训练时间：**{best['train_seconds']:.3f} 秒**  \n"
            f"单样本推理：**{best['inference_ms_per_sample']:.4f} 毫秒**"
        )
    image("model_accuracy.png", "五种算法测试集准确率")

elif page == "数据分析和数据处理":
    st.subheader("数据质量审计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("训练集原始行数", f"{train_profile['rows']:,}")
    c2.metric("清洗后重复行", train_profile["duplicate_rows"])
    c3.metric("Perimeter 缺失", train_profile["missing"]["Perimeter"])
    c4.metric("Solidity 缺失/哨兵", train_profile["missing"]["Solidity"])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### 发现的污染")
        st.markdown(
            """
- `Perimeter` 存在空缺
- `Solidity` 同时存在空缺和 `?` 哨兵值
- `Compactness` 部分记录带有 `cm` 单位
- 标签存在大小写、尾随空格和数字替字
- 清洗标签后出现可识别的重复记录
"""
        )
    with right:
        st.markdown("### 清洗策略")
        st.markdown(
            """
- 提取数值并将无法解析的哨兵统一为缺失值
- 标签去空格、转大写并修复 `0→O`、`3→E`
- 仅删除训练集重复记录
- 使用训练集列中位数填补缺失
- 使用训练集分位数进行稳健截尾
- 逻辑回归和 KNN 使用标准化，树模型保留原尺度
"""
        )
    st.markdown(
        """
<div class="callout warning">
关键原则：验证集与测试集不参与中位数、截尾阈值或标准化参数的拟合。
</div>
""",
        unsafe_allow_html=True,
    )

elif page == "测试精度对比":
    st.subheader("测试集核心结果")
    table = comparison[
        [
            "model",
            "test_accuracy",
            "test_macro_f1",
            "train_seconds",
            "inference_ms_per_sample",
            "model_size_mb",
        ]
    ].copy()
    table.columns = [
        "模型",
        "Accuracy",
        "Macro-F1",
        "训练时间(s)",
        "单样本推理(ms)",
        "模型大小(MB)",
    ]
    st.dataframe(
        table.style.format(
            {
                "Accuracy": "{:.4f}",
                "Macro-F1": "{:.4f}",
                "训练时间(s)": "{:.4f}",
                "单样本推理(ms)": "{:.5f}",
                "模型大小(MB)": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    image("model_accuracy.png")
    st.markdown(
        """
<div class="callout">
KNN 与 XGBoost 的最终精度最接近；逻辑回归以极小模型和极快推理获得良好基线，
高斯朴素贝叶斯则用于验证自主算法实现和概率模型假设。
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    with cols[0]:
        image("confusion_logistic_regression.png", "逻辑回归 混淆矩阵")
    with cols[1]:
        image("confusion_knn.png", "KNN 混淆矩阵")
    with cols[2]:
        image("confusion_xgboost.png", "XGBoost 混淆矩阵")
    with cols[3]:
        image("confusion_random_forest.png", "随机森林 混淆矩阵")
    with cols[4]:
        image("confusion_gaussian_naive_bayes.png", "高斯朴素贝叶斯 混淆矩阵")
    

elif page == "Loss曲线对比":
    st.subheader("训练型算法 Loss 曲线")
    st.caption("KNN、随机森林和高斯朴素贝叶斯没有逐轮梯度训练过程，因此不绘制 Loss 曲线。")
    cols = st.columns(2)
    with cols[0]:
        image("loss_logistic_regression.png")
    with cols[1]:
        image("loss_xgboost.png")

elif page == "算法推理速度对比":
    st.subheader("算法推理速度与效率对比")
    speed_table = comparison[
        [
            "model",
            "inference_ms_per_sample",
            "train_seconds",
            "model_size_mb",
            "test_accuracy",
        ]
    ].copy()
    speed_table.columns = [
        "模型",
        "单样本推理(ms)",
        "训练时间(s)",
        "模型大小(MB)",
        "测试Accuracy",
    ]
    st.dataframe(
        speed_table.style.format(
            {
                "单样本推理(ms)": "{:.5f}",
                "训练时间(s)": "{:.4f}",
                "模型大小(MB)": "{:.4f}",
                "测试Accuracy": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    image("model_efficiency.png")
    st.markdown(
        """
<div class="callout">
推理速度用于衡量模型部署后的使用成本。若只追求精度，KNN 与 XGBoost 表现较强；
若考虑轻量化和快速预测，逻辑回归更适合作为简单稳定的基线模型。
</div>
""",
        unsafe_allow_html=True,
    )

elif page == "鲁棒性对比":
    st.subheader("不同训练噪声下的 Accuracy 下降")
    mean_drop = (
        data["robustness"]
        .groupby("model", as_index=False)["accuracy_drop"]
        .mean()
        .sort_values("accuracy_drop")
    )
    best_robust = mean_drop.iloc[0]
    st.markdown(
        f"""
<div class="callout">
按 12 种噪声设置的平均 Accuracy 下降计算，最稳健模型为
<strong>{best_robust['model']}</strong>，平均下降
<strong>{percent(best_robust['accuracy_drop'])}</strong>。
四类噪声强度升高时，各模型 Accuracy 下降量严格单调增大，说明训练数据污染会持续削弱泛化性能。
</div>
""",
        unsafe_allow_html=True,
    )
    top = st.columns(2)
    with top[0]:
        image("robustness_gaussian_monotonic.png")
    with top[1]:
        image("robustness_impulse_monotonic.png")
    bottom = st.columns(2)
    with bottom[0]:
        image("robustness_missing_monotonic.png")
    with bottom[1]:
        image("robustness_label_monotonic.png")

elif page == "过拟合分析":
    st.subheader("训练集与测试集差距")
    image("overfitting_gap.png")
    st.markdown(
        """
<div class="small-note">
差距越大，模型越可能记忆训练样本。KNN、随机森林和 XGBoost 的训练精度较高，
需要结合测试性能与鲁棒性共同判断，而不能只看训练集成绩。
</div>
""",
        unsafe_allow_html=True,
    )

elif page == "特征消融分析":
    st.subheader("特征工程消融")
    image("feature_ablation.png")
    pivot = data["ablation"].pivot(
        index="model",
        columns="engineered_features",
        values="test_accuracy",
    )
    pivot["提升"] = pivot[True] - pivot[False]
    pivot = pivot.rename(
        columns={False: "仅原始特征", True: "加入衍生特征"}
    ).sort_values("提升", ascending=False)
    st.dataframe(
        pivot.style.format("{:.4f}"),
        use_container_width=True,
    )
    st.subheader("树模型特征重要性")
    cols = st.columns(2)
    with cols[0]:
        image("feature_importance_random_forest.png")
    with cols[1]:
        image("feature_importance_xgboost.png")
