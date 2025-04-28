import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from matplotlib import ticker

# csvファイルを読み込む
df = pd.read_csv("C:\GCI\data.csv")

# Attritionをラベルエンコーディング
df["Attrition"] = df["Attrition"].replace("No", 0)
df["Attrition"] = df["Attrition"].replace("Yes", 1)

# # BusinessTravelをラベルエンコーディング
# df["BusinessTravel"] = df["BusinessTravel"].replace("Non-Travel", 0)
# df["BusinessTravel"] = df["BusinessTravel"].replace("Travel_Rarely", 1)
# df["BusinessTravel"] = df["BusinessTravel"].replace("Travel_Frequently", 2)
#
# # Departmentをラベルエンコーディング
# df["Department"] = df["Department"].replace("Sales", 0)
# df["Department"] = df["Department"].replace("Research & Development", 1)
# df["Department"] = df["Department"].replace("Human Resources", 2)
#
# # EducationFieldをラベルエンコーディング
# df["EducationField"] = df["EducationField"].replace("Life Sciences", 0)
# df["EducationField"] = df["EducationField"].replace("Medical", 1)
# df["EducationField"] = df["EducationField"].replace("Marketing", 2)
# df["EducationField"] = df["EducationField"].replace("Technical Degree", 3)
# df["EducationField"] = df["EducationField"].replace("Human Resources", 4)
# df["EducationField"] = df["EducationField"].replace("Other", 5)
#
# # Genderをラベルエンコーディング
# df["Gender"] = df["Gender"].replace("Male", 0)
# df["Gender"] = df["Gender"].replace("Female", 1)
#
# # JobRoleをラベルエンコーディング
# df["JobRole"] = df["JobRole"].replace("Sales Executive", 0)
# df["JobRole"] = df["JobRole"].replace("Research Scientist", 1)
# df["JobRole"] = df["JobRole"].replace("Laboratory Technician", 2)
# df["JobRole"] = df["JobRole"].replace("Manufacturing Director", 3)
# df["JobRole"] = df["JobRole"].replace("Healthcare Representative", 4)
# df["JobRole"] = df["JobRole"].replace("Manager", 5)
# df["JobRole"] = df["JobRole"].replace("Sales Representative", 6)
# df["JobRole"] = df["JobRole"].replace("Research Director", 7)
# df["JobRole"] = df["JobRole"].replace("Human Resources", 8)
#
# # MaritalStatusをラベルエンコーディング
# df["MaritalStatus"] = df["MaritalStatus"].replace("Single", 0)
# df["MaritalStatus"] = df["MaritalStatus"].replace("Married", 1)
# df["MaritalStatus"] = df["MaritalStatus"].replace("Divorced", 2)

# Over18を削除
df = df.drop("Over18", axis=1)

# EnployeeNumberを削除
df = df.drop("EmployeeNumber", axis=1)

# EnployeeCountを削除
df = df.drop("EmployeeCount", axis=1)

# StandardHoursを削除
df = df.drop("StandardHours", axis=1)
#
# # OverTimeをラベルエンコーディング
# df["OverTime"] = df["OverTime"].replace("No", 0)
# df["OverTime"] = df["OverTime"].replace("Yes", 1)

# Attritionをヒストグラムで可視化
# sns.countplot(data=df, x="Attrition")
# # グラフのラベルを設定
# plt.xlabel("離職したかどうか")
# plt.ylabel("人数")
# plt.show()

target = df["Attrition"]

# dfからAttritionを削除
# df = df.drop("Attrition", axis=1)

# 数値型変数のカラムとAttritionの関係を可視化
# int_columns = [column for column in df.columns if df[column].dtype == "int64"]
# fig, axes = plt.subplots((len(int_columns) + 4) // 5, 5, figsize=(20, 20))
# for i, column in enumerate(int_columns):
#     sns.barplot(x=target, y=column, data=df, palette='Set2', ax=axes[i//5, i%5])
# plt.tight_layout()
# plt.show()

# カテゴリ変数のカラムとAttritionの関係をcountplotで可視化し、2行4列のグラフを表示
# object_columns = [column for column in df.columns if df[column].dtype == "object"]
# fig, axes = plt.subplots(2, 4, figsize=(20, 20))
# for i, column in enumerate(object_columns):
#     sns.countplot(x=column, hue=target, data=df, ax=axes[i//4, i%4])
#     # xラベルを回転
#     axes[i//4, i%4].tick_params(axis='x', rotation=70)
# plt.tight_layout()
# plt.show()


# カテゴリ変数の要素とAttritionのYes/Noの割合を計算
# for column in object_columns:
#     print(df.groupby(column)["Attrition"].value_counts(normalize=True))

# print(df.info())
#
# fig, axes = plt.subplots(3, 3, figsize=(20, 20))
# 量的データを3枚に分けて可視化する
# # スライド1枚目
# sns.countplot(data=df, x="Age", ax=axes[0, 0])
# axes[0, 0].set_title('年齢')
# # x軸の目盛りを5刻みにする
# axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
#
# sns.countplot(data=df, x="DailyAchievement", ax=axes[0, 1])
# axes[0, 1].set_title('日々の業績')
# axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(100))
#
# sns.countplot(data=df, x="DistanceFromHome", ax=axes[0, 2])
# axes[0, 2].set_title('自宅からの距離')
# axes[0, 2].xaxis.set_major_locator(ticker.MultipleLocator(2))
#
# sns.countplot(data=df, x="Education", ax=axes[1, 0])
# axes[1, 0].set_title('教育')
#
# sns.countplot(data=df, x="EnvironmentSatisfaction", ax=axes[1, 1])
# axes[1, 1].set_title('環境満足度')
#
# sns.countplot(data=df, x="HourlyAchievement", ax=axes[1, 2])
# axes[1, 2].set_title('時間毎の業績')
# axes[1, 2].xaxis.set_major_locator(ticker.MultipleLocator(10))
#
# sns.countplot(data=df, x="JobInvolvement", ax=axes[2, 0])
# axes[2, 0].set_title('仕事への関与')
#
# sns.countplot(data=df, x="JobLevel", ax=axes[2, 1])
# axes[2, 1].set_title('職位')
#
# sns.countplot(data=df, x="JobSatisfaction", ax=axes[2, 2])
# axes[2, 2].set_title('仕事満足度')

# スライド2枚目
#
# sns.countplot(data=df, x="MonthlyIncome", ax=axes[0, 0])
# axes[0, 0].set_title('月収')
# axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(200))
#
# sns.countplot(data=df, x="MonthlyAchievement", ax=axes[0, 1])
# axes[0, 1].set_title('月々の業績')
# axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(200))
#
# sns.countplot(data=df, x="NumCompaniesWorked", ax=axes[0, 2])
# axes[0, 2].set_title('勤務した会社の数')
#
# sns.countplot(data=df, x="PercentSalaryHike", ax=axes[1, 0])
# axes[1, 0].set_title('給与の上昇率')
#
# sns.countplot(data=df, x="PerformanceRating", ax=axes[1, 1])
# axes[1, 1].set_title('パフォーマンス評価')
#
# sns.countplot(data=df, x="RelationshipSatisfaction", ax=axes[1, 2])
# axes[1, 2].set_title('関係満足度')
#
# sns.countplot(data=df, x="StockOptionLevel", ax=axes[2, 0])
# axes[2, 0].set_title('株式オプションレベル')
#
# sns.countplot(data=df, x="TotalWorkingYears", ax=axes[2, 1])
# axes[2, 1].set_title('総勤務年数')
# axes[2, 1].xaxis.set_major_locator(ticker.MultipleLocator(2))

# スライド3枚目
# sns.countplot(data=df, x="TrainingTimesLastYear", ax=axes[0, 0])
# axes[0, 0].set_title('昨年の研修回数')
#
# sns.countplot(data=df, x="WorkLifeBalance", ax=axes[0, 1])
# axes[0, 1].set_title('ワークライフバランス')
#
# sns.countplot(data=df, x="YearsAtCompany", ax=axes[0, 2])
# axes[0, 2].set_title('会社での勤続年数')
# # x軸の目盛りを2刻みにする
# axes[0, 2].xaxis.set_major_locator(ticker.MultipleLocator(2))
#
#
# sns.countplot(data=df, x="YearsInCurrentRole", ax=axes[1, 0])
# axes[1, 0].set_title('現在の役職での年数')
#
# sns.countplot(data=df, x="YearsSinceLastPromotion", ax=axes[1, 1])
# axes[1, 1].set_title('最後の昇進からの年数')
#
# sns.countplot(data=df, x="YearsWithCurrManager", ax=axes[1, 2])
# axes[1, 2].set_title('現在のマネージャーとの年数')
#
# sns.countplot(data=df, x="Incentive", ax=axes[2, 0])
# axes[2, 0].set_title('インセンティブ')
# # x軸の目盛りを100刻みにする
# axes[2, 0].xaxis.set_major_locator(ticker.MultipleLocator(100))
# # y軸の上限を6にする
# axes[2, 0].set_ylim(0, 6)
#
#
# sns.countplot(data=df, x="RemoteWork", ax=axes[2, 1])
# axes[2, 1].set_title('リモートワーク')
#
# plt.tight_layout()
# plt.show()
#
# corr = df.corr(numeric_only=True)
# print(corr['Attrition'].sort_values(ascending=False))

# 年齢を20~60までの5刻みで分割
# 5刻みの年齢ごとに離職率を計算
# この結果を棒グラフで可視化
# age = [f"{i}~{i+4}" for i in range(20, 61, 5)]
# attrition_rate = [df[(df['Age'] >= i) & (df['Age'] <= i+4)]['Attrition'].mean() for i in range(20, 61, 5)]
# plt.bar(age, attrition_rate)
# plt.xlabel("年齢")
# plt.ylabel("離職率")
# plt.show()

#　リモートワークの程度で離職率を計算
# remote_work = df.groupby('RemoteWork')['Attrition'].mean()
# print(remote_work)
# remote_work.plot(kind='bar')
# plt.xlabel("リモートワークの程度")
# plt.ylabel("離職率")
# plt.show()

#　環境満足度で離職率を計算
# environment_satisfaction = df.groupby('EnvironmentSatisfaction')['Attrition'].mean()
# print(environment_satisfaction)
# environment_satisfaction.plot(kind='bar')
# plt.xlabel("環境満足度")
# plt.ylabel("離職率")
# plt.show()

# JobLevelで離職率を計算　これは使えなさそう
# job_level = df.groupby('JobLevel')['Attrition'].mean()
# print(job_level)
# job_level.plot(kind='bar')
# plt.xlabel("職位")
# plt.ylabel("離職率")
# plt.show()

# JobInvolvementで離職率を計算
# job_involvement = df.groupby('JobInvolvement')['Attrition'].mean()
# print(job_involvement)
# job_involvement.plot(kind='bar')
# plt.xlabel("仕事への関与度")
# plt.ylabel("離職率")
# plt.show()

#　RelationshipSatisfactionで離職率を計算
# relationship_satisfaction = df.groupby('RelationshipSatisfaction')['Attrition'].mean()
# print(relationship_satisfaction)
# relationship_satisfaction.plot(kind='bar')
# plt.xlabel("人間関係の満足度")
# plt.ylabel("離職率")
# plt.show()

# MoneyIncomeの範囲を1000刻みで分割して範囲ごとの離職率を計算
# money_income = [f"{i}" for i in range(1000, 20001, 1000)]
# attrition_rate = [df[(df['MonthlyIncome'] >= i) & (df['MonthlyIncome'] <= i+999)]['Attrition'].mean() for i in range(1000, 20001, 1000)]
# plt.bar(money_income, attrition_rate)
# plt.xlabel("月収")
# plt.ylabel("離職率")
# plt.xticks(np.arange(0, 21, 2))
# plt.show()

# PercentSalaryHikeの数値ごとの離職率を計算
# percent_salary_hike = df.groupby('PercentSalaryHike')['Attrition'].mean()
# print(percent_salary_hike)
# percent_salary_hike.plot(kind='bar')
# plt.xlabel("給与の上昇率(%)")
# plt.ylabel("離職率")
# plt.show()

# StockOptionLevelの数値ごとの離職率を計算
# stock_option_level = df.groupby('StockOptionLevel')['Attrition'].mean()
# print(stock_option_level)
# stock_option_level.plot(kind='bar')
# plt.xlabel("ストックオプションの度合い")
# plt.ylabel("離職率")
# plt.show()

# DistanceFromHomeの数値ごとの離職率を計算
# distance_from_home = df.groupby('DistanceFromHome')['Attrition'].mean()
# print(distance_from_home)
# distance_from_home.plot(kind='bar')
# plt.xlabel("自宅からの距離")
# plt.ylabel("離職率")
# plt.show()

# x軸をDistanceFromHome、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="DistanceFromHome", hue="Attrition")
# plt.xlabel("自宅からの距離")
# plt.ylabel("人数")
# plt.show()

# WorkLifeBalanceの数値ごとの離職率を計算
# work_life_balance = df.groupby('WorkLifeBalance')['Attrition'].mean()
# print(work_life_balance)
# work_life_balance.plot(kind='bar')
# plt.xlabel("ワークライフバランス")
# plt.ylabel("離職率")
# plt.show()

# x軸をWorkLifeBalance、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="WorkLifeBalance", hue="Attrition")
# plt.xlabel("ワークライフバランス")
# plt.ylabel("人数")
# plt.show()

# WorkLifeBalanceの数値ごとの離職率を計算
# work_life_balance = df.groupby('WorkLifeBalance')['Attrition'].mean()
# print(work_life_balance)

# x軸をStockOptionLevel、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="StockOptionLevel", hue="Attrition")
# plt.xlabel("ストックオプションの度合い")
# plt.ylabel("人数")
# plt.show()

# x軸をEnvironmentSatisfaction、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="EnvironmentSatisfaction", hue="Attrition")
# plt.xlabel("環境満足度")
# plt.ylabel("人数")
# plt.show()

# x軸をRelationshipSatisfaction、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="RelationshipSatisfaction", hue="Attrition")
# plt.xlabel("人間関係の満足度")
# plt.ylabel("人数")
# plt.show()

# x軸をWorkLifeBalance、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="WorkLifeBalance", hue="Attrition")
# plt.xlabel("ワークライフバランス")
# plt.ylabel("人数")
# plt.show()

# x軸をMonthlyIncome、y軸をカウントとして、Attiritionの割合を棒グラフで可視化。
# sns.countplot(data=df, x="MonthlyIncome", hue="Attrition")
# plt.xlabel("月収")
# plt.ylabel("人数")
# plt.show()

# 職種別の給料の分布を箱ひげ図で可視化
# sns.boxplot(data=df, x="JobRole", y="MonthlyIncome")
# plt.xticks(rotation=90)
# plt.xlabel("職種")
# plt.ylabel("月収")
# plt.show()



