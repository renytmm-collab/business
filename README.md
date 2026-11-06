# 促销与广告的因果效果估计、可靠性分析与策略评估（题目 10）

本仓库实现**题目 10（促销的个体处理效应：Uplift 建模与双重稳健估计）**中的**“因果效果估计与可靠性分析”**模块，并与队友的 [uplift_training](file:///D:/WX_files/xwechat_files/wxid_4n8a0yrbh44r22_f387/msg/file/2026-11/uplift_training/uplift_training)（Uplift 模型训练与营销策略）模块实现无缝接口对接。

---

## 1. 核心研究内容与文件对应关系

| 任务模块 | 核心科学问题 | 对应代码与输出文件 |
| :--- | :--- | :--- |
| **0. 问题定义与可证伪假设** | 形式化定义 ATE / CATE / 策略价值、双重稳健乘积误差界、半参数方差下界及 H1–H5 五项可证伪假设 | [`问题定义与可证伪假设.md`](file:///d:/code/business/问题定义与可证伪假设.md) |
| **1. 三步递进 ATE 估计流程** | 朴素均值差 $\to$ 无交叉拟合样本内 AIPW $\to$ K 折交叉拟合 AIPW / DML-IRM $\to$ 真值已知的 Oracle 参照 | [`estimators.py`](file:///d:/code/business/estimators.py)、[`results/criteo_ate_comparison.csv`](file:///d:/code/business/results/criteo_ate_comparison.csv) |
| **2. 真值已知的 Monte Carlo 可靠性实验** | 在 200 次重复生成数据下，检验**小效应量、样本量、重叠度不足、四象限模型误设**对偏误（Bias）、RMSE、95% 区间覆盖率与区间宽度的影响 | [`dgp.py`](file:///d:/code/business/dgp.py)、[`run_simulations.py`](file:///d:/code/business/run_simulations.py)、[`results/simulation_core_metrics.csv`](file:///d:/code/business/results/simulation_core_metrics.csv)、[`results/simulation_misspecification_metrics.csv`](file:///d:/code/business/results/simulation_misspecification_metrics.csv) |
| **3. 处理后特征偏差量化实验** | 验证 $\text{广告 } T \to \text{处理后访问 } V_{\text{post}} \to \text{购买 } Y$ 及碰撞偏差下，**“预测 AUC 提升但因果估计失真”**的悖论 | [`results/post_treatment_metrics.csv`](file:///d:/code/business/results/post_treatment_metrics.csv)、[`results/figures/post_treatment_bias_paradox.png`](file:///d:/code/business/results/figures/post_treatment_bias_paradox.png) |
| **4. 不可识别 vs 不可学习边界分析** | 区分**未观测混淆、缺乏重叠、小效应稀有事件**，以及**全员零效应 vs 正负效应抵消（$\text{ATE}=0$ 但个体 Uplift 可学习）** | [`results/identifiability_metrics.csv`](file:///d:/code/business/results/identifiability_metrics.csv)、[`results/sure_things_vs_persuadables_metrics.csv`](file:///d:/code/business/results/sure_things_vs_persuadables_metrics.csv) |
| **5. 队友 Uplift 策略独立评估与成本收益** | 基于 200,000 条 Criteo 独立测试集，对比随机投放、基础购买率 $\hat\mu_0(X)$ 排序、T-Learner 与 DR-Learner 排序在触达成本与核销补贴模型下的增量效果与净收益 | [`run_criteo_analysis.py`](file:///d:/code/business/run_criteo_analysis.py)、[`results/policy_evaluation_table.csv`](file:///d:/code/business/results/policy_evaluation_table.csv)、[`results/teammate_minimal_evaluation_interface.csv`](file:///d:/code/business/results/teammate_minimal_evaluation_interface.csv) |
| **6. 完整研究报告** | 汇总所有真实数据结论、模拟实验表格、可视化图谱与方法论边界说明 | [`因果效果估计与可靠性分析报告.md`](file:///d:/code/business/因果效果估计与可靠性分析报告.md) |

---

## 2. 环境配置与一键复现命令

在本目录（`d:\code\business`）下运行：

```powershell
# 1. 创建虚拟环境并安装依赖
uv venv .venv
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 运行单元测试（验证双重稳健恒等式、DGP 概率校准、交叉拟合去偏、处理后偏差、策略评估尺度）
.\.venv\Scripts\python.exe -m unittest -v test_causal.py

# 3. 运行 Criteo 真实数据 ATE 对比与队友 Uplift 策略评估
.\.venv\Scripts\python.exe run_criteo_analysis.py

# 4. 运行 200 次重复 Monte Carlo 可靠性、覆盖率、处理后特征偏差与可学习性边界模拟实验
.\.venv\Scripts\python.exe run_simulations.py --replications 200 --jobs 12
```

---

## 3. 开放数据来源与合规许可说明

- **数据集名称**：Criteo Uplift Prediction Dataset v2.1（修正旧版广告主混杂泄漏的官方版本）
- **官方获取链接**：
  - Criteo AI Lab: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
  - Hugging Face 仓库: https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz
- **文件 SHA-256 校验值**：`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- **许可协议**：CC BY-NC-SA 4.0（遵守非商业学术研究共享规范，不重新分发原始 300MB 压缩包，通过行号清单 `split_manifest.csv` 与 `test_predictions.csv` 实现精确复现）。
- **原始文献引用**：Diemert E, Betlei A, Renaudin C, Amini M-R. *A Large Scale Benchmark for Uplift Modeling*. KDD 2018 Workshop on AdKDD & TargetAd.
