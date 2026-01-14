import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# ==========================================
# 1. 大规模数据模拟
# ==========================================
np.random.seed(42) # 保证结果可复现

# 设定时间范围：5年数据
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
n = len(dates)

# 构造数据成分
# A. 趋势项 (Trend): 线性增长，模拟业务扩张
trend = np.linspace(50, 120, n) 

# B. 季节项 (Seasonality): 
#    - 周度循环 (7天): 周末销量高
weekly_season = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
#    - 年度循环 (365天): 模拟淡旺季
yearly_season = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)

# C. 随机噪声 (Noise): 模拟不确定性
noise = np.random.normal(0, 12, n)  # 均值为0，标准差12

# D. 合成总销量 (需保证非负)
sales = trend + weekly_season + yearly_season + noise
sales = np.maximum(sales, 0) # 修正负值

# 创建DataFrame
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df.set_index('Date', inplace=True)

# 划分训练集和测试集 (最后30天作为验证/预测对象)
train_data = df.iloc[:-30]
test_data = df.iloc[-30:]

print(f"数据总条数: {n} 条")
print(f"训练集时间范围: {train_data.index.min().date()} 到 {train_data.index.max().date()}")
print(f"测试集时间范围: {test_data.index.min().date()} 到 {test_data.index.max().date()}")

# 绘图：展示历史销售全貌
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Sales'], label='历史销售数据', color='steelblue', alpha=0.7)
plt.title('图1：某短生命周期产品5年历史销售趋势图', fontsize=16)
plt.ylabel('销量 (单位)', fontsize=12)
plt.legend()
plt.show()
# ==========================================
# 2. 需求预测
# ==========================================

# 建立模型：加法模型 (Trend='add', Seasonal='add')，周期为7 (周度季节性)
# 注：对于长周期季节性，statsmodels会自动捕捉或通过更复杂参数设置
model = ExponentialSmoothing(
    train_data['Sales'], 
    trend='add', 
    seasonal='add', 
    seasonal_periods=7  # 显式指定周度季节性
).fit()

# 进行预测：预测未来30天
forecast = model.forecast(30)
forecast.index = test_data.index

# 计算预测误差指标 (MSE, RMSE)
mse = ((test_data['Sales'] - forecast) ** 2).mean()
rmse = np.sqrt(mse)
mae = np.abs(test_data['Sales'] - forecast).mean()

# 此外，我们需要计算"预测残差的标准差"，这是库存模型中计算安全库存的关键
# 使用训练集的残差来估计未来的不确定性 sigma
residuals = train_data['Sales'] - model.fittedvalues
sigma_forecast = residuals.std()

print("="*30)
print("模型评估结果")
print("="*30)
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"预测分布标准差 (Sigma): {sigma_forecast:.2f}")

# 绘图：预测结果对比
plt.figure(figsize=(15, 6))
# 为了看清细节，只画最后3个月
display_start = '2023-10-01'
plt.plot(df.loc[display_start:].index, df.loc[display_start:]['Sales'], label='实际销量', color='gray')
plt.plot(forecast.index, forecast, label='Holt-Winters预测值', color='red', linestyle='--', linewidth=2)
plt.fill_between(forecast.index, 
                 forecast - 1.96*sigma_forecast, 
                 forecast + 1.96*sigma_forecast, 
                 color='pink', alpha=0.3, label='95% 置信区间')
plt.title('图2：Holt-Winters 模型预测对比 (局部放大)', fontsize=16)
plt.legend()
plt.show()
# ==========================================
# 3. 库存决策优化
# ==========================================

# 设定商业参数 (单位：元)
price = 60      # 销售单价 (P)
cost = 20       # 进货成本 (C)
salvage = 10    # 残值/处理价 (S) - 卖不掉只能低价处理

# 计算边际损益
# Cu (Underage Cost): 少订货导致的损失 (机会成本) = 售价 - 成本
Cu = price - cost 
# Co (Overage Cost): 多订货导致的损失 (超储成本) = 成本 - 残值
Co = cost - salvage

# 计算临界分位数 (Critical Ratio / Service Level)
# 公式来源：课件《库存》P.1041 -> P(r <= Q*) = k / (k + h) 即 Cu / (Cu + Co)
critical_ratio = Cu / (Cu + Co)

print("="*30)
print("报童模型参数")
print("="*30)
print(f"单位缺货机会成本 (Cu): {Cu}")
print(f"单位库存积压成本 (Co): {Co}")
print(f"最优服务水平 (Critical Ratio): {critical_ratio:.4f}")

# 针对未来某一天(例如测试集的第一天)进行决策
target_date = test_data.index[0]
mu_demand = forecast[target_date]  # 预测的均值
sigma_demand = sigma_forecast      # 预测的标准差

# 计算最优订货量 Q*
# 使用正态分布的逆函数 (ppf) 求解
# Q* = mu + Z * sigma
Q_optimal = norm.ppf(critical_ratio, loc=mu_demand, scale=sigma_demand)

print(f"\n针对日期 {target_date.date()} 的决策:")
print(f"预测需求均值: {mu_demand:.2f}")
print(f"最优订货量 (Q*): {Q_optimal:.2f} (向上取整: {np.ceil(Q_optimal)})")
# ==========================================
# 4. 蒙特卡洛仿真与策略对比
# ==========================================

n_simulations = 10000  # 模拟1万次

# 模拟真实的随机需求：服从 N(预测均值, 预测标准差)
simulated_demands = np.random.normal(mu_demand, sigma_demand, n_simulations)
simulated_demands = np.maximum(simulated_demands, 0) # 需求不能为负

# 定义利润计算函数
def calculate_profit(order_qty, demand):
    sold = np.minimum(order_qty, demand)       # 卖出量
    unsold = np.maximum(order_qty - demand, 0) # 滞销/积压量
    
    revenue = sold * price
    salvage_revenue = unsold * salvage
    total_cost = order_qty * cost
    
    return revenue + salvage_revenue - total_cost

# 策略对比
# 策略A: 订货量 = 预测均值 (经验主义)
Q_mean = mu_demand
profits_mean = calculate_profit(Q_mean, simulated_demands)

# 策略B: 订货量 = 最优订货量 Q* (数据驱动)
Q_opt = Q_optimal
profits_opt = calculate_profit(Q_opt, simulated_demands)

# 策略C: 激进策略 (订货量 = 预测均值 + 2倍标准差)
Q_high = mu_demand + 2 * sigma_demand
profits_high = calculate_profit(Q_high, simulated_demands)

# 输出结果对比
df_res = pd.DataFrame({
    'Strategy': ['策略A: 均值订货', '策略B: 最优订货(Q*)', '策略C: 激进订货'],
    'Order_Qty': [Q_mean, Q_opt, Q_high],
    'Avg_Profit': [profits_mean.mean(), profits_opt.mean(), profits_high.mean()],
    'Risk (Std)': [profits_mean.std(), profits_opt.std(), profits_high.std()]
})

print("="*30)
print("仿真结果对比 (10,000次模拟)")
print("="*30)
print(df_res)

# 绘图：利润分布直方图
plt.figure(figsize=(12, 6))
sns.histplot(profits_mean, color='blue', alpha=0.3, label=f'策略A: 均值 (Q={Q_mean:.0f})', kde=True)
sns.histplot(profits_opt, color='red', alpha=0.3, label=f'策略B: 最优 (Q={Q_opt:.0f})', kde=True)
plt.axvline(profits_mean.mean(), color='blue', linestyle='--')
plt.axvline(profits_opt.mean(), color='red', linestyle='--')
plt.title(f'图3：不同订货策略下的利润分布模拟 (Critical Ratio={critical_ratio:.2f})', fontsize=16)
plt.xlabel('利润', fontsize=12)
plt.legend()
plt.show()

# 灵敏度分析图：不同订货量对应的期望利润
q_range = np.linspace(mu_demand - 3*sigma_demand, mu_demand + 3*sigma_demand, 100)
expected_profits = []
for q in q_range:
    p = calculate_profit(q, simulated_demands).mean()
    expected_profits.append(p)

plt.figure(figsize=(10, 5))
plt.plot(q_range, expected_profits, linewidth=2, color='green')
plt.axvline(Q_opt, color='red', linestyle='--', label=f'最优订货量 Q*={Q_opt:.1f}')
plt.scatter(Q_opt, profits_opt.mean(), color='red', s=100, zorder=5)
plt.title('图4：订货量与期望利润的关系 (灵敏度分析)', fontsize=16)
plt.xlabel('订货量', fontsize=12)
plt.ylabel('期望利润', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
