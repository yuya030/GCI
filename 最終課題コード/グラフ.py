import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import japanize_matplotlib

increments = [i for i in range(0, 51)]  # 給与上昇率 (%) 0%から50%
attrition_rates = [0.16 - (0.16 * (i / 100) * 0.25) for i in increments]  # 仮の離職率計算

# グラフ1: 離職の分布
data = pd.DataFrame({'Attrition': ['Yes', 'No', 'Yes', 'No', 'No']})
plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=data, palette='pastel')
plt.title('離職の分布')
plt.ylabel('件数')
plt.xlabel('離職 (はい/いいえ)')
plt.show()

# # グラフ2: 特徴量と離職の相関
# correlation_data = pd.Series({
#     'MonthlyIncome': 0.15,
#     'TotalWorkingYears': 0.12,
#     'OverTime': 0.18,
#     'Age': -0.05,
#     'YearsAtCompany': 0.07
# }).sort_values(ascending=False)
#
# plt.figure(figsize=(8, 10))
# sns.barplot(x=correlation_data.values, y=correlation_data.index, palette="coolwarm")
# plt.title('特徴量と離職の相関')
# plt.xlabel('相関係数')
# plt.ylabel('特徴量')
# plt.show()

correlation_data = pd.Series({
    'MonthlyIncome': 0.15,
    'TotalWorkingYears': 0.12,
    'OverTime': 0.18,
    'Age': -0.05,
    'YearsAtCompany': 0.07,
    'DistanceFromHome': 0.03,
    'JobSatisfaction': -0.1,
    'WorkLifeBalance': -0.08,
    'YearsInCurrentRole': -0.09,
    'JobLevel': 0.05
}).sort_values(ascending=False)

# 特徴量と離職の相関を可視化
plt.figure(figsize=(8, 10))
sns.barplot(x=correlation_data.values, y=correlation_data.index, palette="coolwarm")
plt.title('特徴量と離職の相関')
plt.xlabel('相関係数')
plt.ylabel('特徴量')
plt.show()

# グラフ3: 給与上昇率と離職率の関係
plt.figure(figsize=(8, 6))
plt.plot(increments, attrition_rates, marker='o', linestyle='-', color='blue')
plt.title('給与上昇率と離職率の関係')
plt.xlabel('給与上昇率 (%)')
plt.ylabel('離職率')
plt.grid()
plt.show()
