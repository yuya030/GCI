import pandas as pd

df = pd.read_csv("C:\GCI\data.csv")

# 離職した人と離職してない人でDistancesFromHomeの平均値を出してグラフを作成
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
#
# df["Attrition"] = df["Attrition"].replace({"Yes": 1, "No": 0})
# sns.barplot(x="Attrition", y="DistanceFromHome", data=df)
# plt.show()

# 離職した人のOverTimeの割合を出して円グラフを作成
over_time = df[df["Attrition"] == "Yes"]["OverTime"].value_counts()
plt.pie(over_time, labels=over_time.index, autopct="%1.1f%%", colors=["blue", "orange"])
plt.title("離職した人のOverTimeの割合")
plt.show()

# 離職してない人のOverTimeの割合を出して円グラフを作成
over_time = df[df["Attrition"] == "No"]["OverTime"].value_counts()
plt.pie(over_time, labels=over_time.index, autopct="%1.1f%%", colors=["orange", "blue"])
plt.title("離職してない人のOverTimeの割合")
plt.show()
