# ===================== 1. 导入依赖包 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 固定全局随机种子，保证交叉验证分折结果可复现
np.random.seed(123)

# 自定义C-index评分器，适配GridSearchCV接口
def c_index_score(y_true, y_pred):
    """包装C-index函数，仅返回单值评分，适配sklearn框架"""
    return concordance_index_censored(y_true, y_pred)[0]
# 转为sklearn兼容的评分对象，C-index越大模型越好
c_index_scorer = make_scorer(c_index_score, greater_is_better=True)


# ===================== 2. 数据读取与预处理 =====================
# ========== 核心修改：读取本地CSV，拆分特征与生存响应 ==========
# 替换为你的CSV文件实际路径
df = pd.read_csv("C://cox_survival_data.csv")

# 校验必要列是否存在，避免列名错误导致报错
assert "time" in df.columns and "status" in df.columns, "CSV必须包含time和status两列"

# 拆分生存响应与特征矩阵
y_df = df[["time", "status"]]  # 提取生存响应
x_df = df.drop(["time", "status"], axis=1)  # 提取特征矩阵

# 自动获取样本量与特征数，适配任意大小数据集
n_samples, n_features = x_df.shape

# ========== 格式转换：适配模型输入 ==========
# X：DataFrame可直接传入模型，无需额外转换
x = x_df
# Y：两列DataFrame转标准Surv对象（必须步骤，等价于R的Surv(time, status)）
y = Surv.from_arrays(
    event = y_df["status"].astype(bool),  # 事件列转布尔值：True=事件发生，False=右删失
    time = y_df["time"]                   # 生存时间列
)

# 验证数据格式
print(f"数据集加载完成：样本量{n_samples}，特征数{n_features}")
print(f"前5个特征名：{list(x.columns[:5])}...")
print(f"生存响应前5行：\n{y_df.head()}\n")


# ===================== 3. 拟合 Lasso Cox 全路径模型 =====================
lasso_cox_fit = CoxnetSurvivalAnalysis(
    l1_ratio=1.0,          # 纯Lasso(L1惩罚)，设为0则为Ridge，0-1为弹性网
    n_alphas=100,          # 自动生成100个惩罚强度λ的完整路径
    normalize=True         # 自动标准化特征，与R glmnet默认行为一致
)

lasso_cox_fit.fit(x, y)

lambda_path = lasso_cox_fit.alphas_  # λ序列：降序排列（惩罚从强到弱）
coef_path = lasso_cox_fit.coef_      # 全路径系数矩阵：形状为(特征数, λ个数)

print(f"模型拟合完成，共生成 {len(lambda_path)} 个λ值")
print(f"系数路径矩阵维度：{coef_path.shape}（特征数 × λ个数）\n")


# ===================== 4. 10折交叉验证：选择最优惩罚参数λ =====================
# 构建参数网格：每个模型对应单个λ值
param_grid = {"alphas": [[a] for a in lambda_path]}

cv_lasso_cox = GridSearchCV(
    estimator=CoxnetSurvivalAnalysis(l1_ratio=1.0, normalize=True),
    param_grid=param_grid,
    cv=10,                  # 10折交叉验证
    scoring=c_index_scorer, # 评价指标：Harrell C-index
    n_jobs=-1               # 并行加速
)

cv_lasso_cox.fit(x, y)

# 修正版1se规则计算，避免空数组报错
cv_results = pd.DataFrame(cv_lasso_cox.cv_results_)
cv_mean = cv_results["mean_test_score"].values
cv_se = cv_results["std_test_score"].values / np.sqrt(10)

best_idx = np.argmax(cv_mean)
best_score = cv_mean[best_idx]
lambda_min = lambda_path[best_idx]  # C-index最高的最优λ

# 计算1倍标准误阈值，筛选满足条件的模型
threshold = best_score - cv_se[best_idx]
valid_idx = np.where(cv_mean >= threshold)[0]

# 1se规则：选满足条件中惩罚最强（lambda最大、索引最小）的最精简模型
lambda_1se = lambda_path[valid_idx[0]] if len(valid_idx) > 0 else lambda_min

print(f"最优lambda.min：{round(lambda_min, 4)}（C-index最高）")
print(f"精简模型lambda.1se：{round(lambda_1se, 4)}（1倍标准误内最精简）\n")


# ===================== 5. 提取最优模型系数与核心变量 =====================
# 提取lambda.min对应的特征系数
coef_min = coef_path[:, best_idx]

# 整理为带特征名的系数表
coef_min_df = pd.DataFrame({
    "variable": x.columns,
    "coefficient": coef_min
})

# 筛选非零核心变量，按系数绝对值降序排序
selected_vars_min = coef_min_df[coef_min_df["coefficient"] != 0].copy()
selected_vars_min = selected_vars_min.sort_values(
    by="coefficient", key=abs, ascending=False
).reset_index(drop=True)

print(f"lambda.min下筛选出的非零核心变量（共{len(selected_vars_min)}个）：")
print(selected_vars_min.to_string(index=False))


# ===================== 6. 结果可视化 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 图1：Lasso Cox系数路径图 ----------
plt.figure(figsize=(10, 6))
log_lambda = -np.log(lambda_path)

# 动态遍历所有特征绘制系数路径，适配任意特征数
for i in range(n_features):
    plt.plot(log_lambda, coef_path[i, :], linewidth=1)

# 标注两个最优λ切点
plt.axvline(x=-np.log(lambda_min), color="red", linestyle="--", label="lambda.min")
plt.axvline(x=-np.log(lambda_1se), color="blue", linestyle="--", label="lambda.1se")

plt.xlabel("-Log(λ)")
plt.ylabel("Coefficients")
plt.title("Lasso Cox 系数路径图")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.show()

# ---------- 图2：10折交叉验证 C-index 曲线 ----------
plt.figure(figsize=(10, 6))
plt.errorbar(log_lambda, cv_mean, yerr=cv_se, fmt="o", color="crimson", capsize=3)

plt.axvline(x=-np.log(lambda_min), color="black", linestyle=":")
plt.axvline(x=-np.log(lambda_1se), color="black", linestyle=":")

plt.xlabel("-Log(λ)")
plt.ylabel("C-index")
plt.title("10折交叉验证 C-index 曲线")
plt.grid(alpha=0.3)
plt.show()


# ===================== 7. 绘制最优模型平均生存曲线 =====================
# 重新拟合最优模型，开启基线生存函数计算（绘制生存曲线必填）
best_model = CoxnetSurvivalAnalysis(
    l1_ratio=1.0,
    alphas=[lambda_min],
    normalize=True,
    fit_baseline_model=True
)
best_model.fit(x, y)

# 计算协变量均值对应的基准样本生存曲线（与R默认输出一致）
mean_x = x.mean(axis=0).values.reshape(1, -1)
surv_func = best_model.predict_survival_function(mean_x)[0]

plt.figure(figsize=(10, 6))
plt.step(surv_func.x, surv_func.y, where="post", linewidth=2, color="steelblue")
plt.xlabel("生存时间")
plt.ylabel("生存率")
plt.title("Lasso Cox 最优模型平均生存曲线")
plt.grid(alpha=0.3)
plt.show()