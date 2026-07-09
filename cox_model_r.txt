# ===================== 1. 安装与加载依赖包 =====================
# 判断是否已安装glmnet包，未安装则自动从CRAN安装
if (!require("glmnet", quietly = TRUE)) {
  install.packages("glmnet")
}
# 判断是否已安装survival包，未安装则自动从CRAN安装
if (!require("survival", quietly = TRUE)) {
  install.packages("survival")
}

# 加载glmnet包：核心工具，实现正则化Cox模型（Lasso/弹性网/Ridge）
library(glmnet)
# 加载survival包：生成标准生存响应对象、提供生存分析基础函数
library(survival)


# ===================== 2. 数据准备（内置示例，可替换自有数据） =====================
# 加载glmnet内置的Cox模型示例数据集，无需额外导入文件
data(CoxExample)

# 提取协变量矩阵x：n行（样本数）× p列（特征数），每行对应1个样本的所有特征
x <- CoxExample$x
# 提取生存响应y：Surv格式两列矩阵，第一列为生存时间，第二列为事件状态（1=事件发生，0=右删失）
y <- CoxExample$y

# 查看前5行响应数据，确认时间+事件的标准格式
head(y, 5)
# 查看协变量维度：示例为100个样本、30个特征（典型高维场景，适配Lasso变量筛选）
dim(x)

y <- Surv(time = CoxExample$y[, "time"], event = CoxExample$y[, "status"])

# 验证y的类型：输出 "Surv" 即格式正确
class(y)

#combined_data <- cbind(y, x)
# 3. 转为数据框格式（write.csv的标准输入格式）
#combined_df <- as.data.frame(combined_data)
# 4. 导出为CSV文件，row.names = FALSE 表示不导出行号
#write.csv(combined_df, file = "C://cox_survival_data.csv", row.names = FALSE)
# ===================== 3. 拟合 Lasso Cox 全路径模型 =====================
# 固定随机种子，保证模型拟合结果可复现
set.seed(123)

# 拟合Lasso Cox回归全路径模型，核心参数逐行说明：
lasso_cox_fit <- glmnet(
  x = x,                  # 输入协变量矩阵
  y = y,                  # 输入生存响应对象
  family = "cox",         # 指定模型族为Cox比例风险模型
  alpha = 1,              # 惩罚类型：1=纯Lasso(L1惩罚)，0=Ridge(L2惩罚)，0-1为弹性网
  cox.ties = "efron",     # 生存时间结点处理方法，与coxph默认一致，消除版本过渡警告
  nlambda = 100           # 自动生成100个不同惩罚强度λ的模型，形成完整系数路径
)

# 打印模型基础信息：λ数量、每个λ对应的非零系数个数、偏似然偏差
print(lasso_cox_fit)


# ===================== 4. K折交叉验证：选择最优惩罚参数λ =====================
# 固定随机种子，保证交叉验证的样本分折可复现
set.seed(123)

# 执行10折交叉验证，用于选择最优惩罚强度λ，核心参数逐行说明：
cv_lasso_cox <- cv.glmnet(
  x = x,                  # 输入协变量矩阵
  y = y,                  # 输入生存响应对象
  family = "cox",         # 指定模型族为Cox比例风险模型
  alpha = 1,              # 保持Lasso惩罚，与基础模型一致
  cox.ties = "efron",     # 结点处理方法与基础模型保持一致
  type.measure = "C",     # 模型评价指标：Harrell C-index（一致性指数，越高越好）
  nfolds = 10,            # 10折交叉验证，行业通用标准
  grouped = TRUE          # 默认值：更高效的风险集分组计算方式，适配小样本场景
)

# 提取两个行业通用的最优λ值：
# lambda.min：交叉验证误差最小（C-index最高）对应的λ，模型拟合效果最优
lambda_min <- cv_lasso_cox$lambda.min
# lambda.1se：最小误差1倍标准误范围内，惩罚最强的λ，模型更简洁、泛化性更强
lambda_1se <- cv_lasso_cox$lambda.1se

# 打印两个最优λ的数值
cat("最优lambda.min：", round(lambda_min, 4), "\n")
cat("简洁模型lambda.1se：", round(lambda_1se, 4), "\n")


# ===================== 5. 提取最优模型系数与核心变量 =====================
# 提取lambda.min对应的模型系数，返回稀疏矩阵格式
coef_min <- coef(cv_lasso_cox, s = "lambda.min")
# 提取lambda.1se对应的模型系数
coef_1se <- coef(cv_lasso_cox, s = "lambda.1se")

# 将系数转为数据框格式，方便筛选与查看
coef_min_df <- data.frame(
  variable = rownames(coef_min),  # 变量名
  coefficient = as.numeric(coef_min)  # 对应系数值
)

# 筛选非零系数的变量：Lasso核心特性，系数为0的变量被自动剔除
selected_vars_min <- coef_min_df[coef_min_df$coefficient != 0, ]
# 按系数绝对值降序排序，直观展示对生存风险影响最大的因子
selected_vars_min <- selected_vars_min[order(-abs(selected_vars_min$coefficient)), ]

# 打印lambda.min下筛选出的核心变量及其系数
cat("\nlambda.min下筛选出的非零变量（共", nrow(selected_vars_min), "个）：\n")
print(selected_vars_min, row.names = FALSE)


# ===================== 6. 结果可视化 =====================
# 图1：Lasso Cox系数路径图
# 横轴为-log(λ)，纵轴为特征系数，每条线对应一个特征
# 规律：λ越小（惩罚越弱），越多变量的系数变为非零
plot(lasso_cox_fit, xvar = "lambda", label = TRUE, main = "Lasso Cox 系数路径图")
# 添加红色虚线，标记lambda.min的位置
abline(v = log(lambda_min), lty = 2, col = "red")
# 添加蓝色虚线，标记lambda.1se的位置
abline(v = log(lambda_1se), lty = 2, col = "blue")
# 添加图例
legend("topleft", legend = c("lambda.min", "lambda.1se"), lty = 2, col = c("red", "blue"))

# 图2：交叉验证误差曲线
# 横轴为-log(λ)，纵轴为C-index，点为折内均值，误差线为标准误
# 两条竖线分别对应lambda.min和lambda.1se
plot(cv_lasso_cox, main = "10折交叉验证 C-index 曲线")


# ===================== 7. 拓展：绘制最优模型生存曲线 =====================
# 基于lambda.min的最优模型，计算基线生存函数
# 必须传入拟合模型、原始x/y数据和指定的λ值
surv_fit <- survfit(
  cv_lasso_cox,
  s = "lambda.min",
  x = x,
  y = y
)

# 绘制整体平均生存曲线（协变量取均值的基准样本）
plot(surv_fit, main = "Lasso Cox 最优模型平均生存曲线",
     xlab = "生存时间", ylab = "生存率")
