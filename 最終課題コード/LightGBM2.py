import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# データの読み込み
file_path = r'C:\GCI\data.csv'
data = pd.read_csv(file_path)

# Over18を削除
data = data.drop("Over18", axis=1)

# EnployeeNumberを削除
data = data.drop("EmployeeNumber", axis=1)

# EnployeeCountを削除
data = data.drop("EmployeeCount", axis=1)

# StandardHoursを削除
data = data.drop("StandardHours", axis=1)

target_column = 'Attrition'
X = data.drop(columns=[target_column])
y = data[target_column]

# カテゴリ変数をラベルエンコーディング
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# ターゲット変数のエンコーディング
y = LabelEncoder().fit_transform(y)

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LGBMデータセット作成
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# パラメータ設定
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

# コールバックを使用してearly_stoppingとログを設定
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.log_evaluation(period=50),           # 50イテレーションごとに評価結果をログ出力
        lgb.early_stopping(stopping_rounds=50)   # 早期終了
    ]
)

# 予測と評価
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
# 再現率、適合率、F1値を表示
print(classification_report(y_test, y_pred))

# 混同行列を表示
print(confusion_matrix(y_test, y_pred))


# # 給与を5％上昇させた場合の離職率を予測
# # 元データのコピー
# data_increase = data.copy()
#
# # 既に Over18, EmployeeNumber, EmployeeCount, StandardHours は削除済みならOK
# # Attrition も削除して学習時と同じ構成にする
# data_increase = data_increase.drop("Attrition", axis=1)
#
# # 給与を5％上昇
# data_increase['MonthlyIncome'] *= 1.05
#
# # カテゴリ列をエンコード
# for column in data_increase.select_dtypes(include='object').columns:
#     data_increase[column] = label_encoders[column].transform(data_increase[column])
#
# # 予測
# y_pred_prob_increase = model.predict(data_increase, num_iteration=model.best_iteration)
# y_pred_increase = np.where(y_pred_prob_increase > 0.5, 1, 0)
#
# # 離職率
# attrition_rate = y_pred_increase.mean()
# print(f"Attrition rate up5％: {attrition_rate:.4f}")
#
# # 元データをコピー(「Attrition」は落とすがAgeは元のまま)
# data_increase_20_34_only = data.copy()
# data_increase_20_34_only.drop("Attrition", axis=1, inplace=True)
#
# # 20～34歳の行に対してのみ給与を5%上昇
# mask_20_34 = (data_increase_20_34_only["Age"] >= 20) & (data_increase_20_34_only["Age"] <= 34)
# data_increase_20_34_only.loc[mask_20_34, "MonthlyIncome"] *= 1.05
#
# # 変数をエンコード
# for column in data_increase_20_34_only.select_dtypes(include='object').columns:
#     data_increase_20_34_only[column] = label_encoders[column].transform(data_increase_20_34_only[column])
#
# # 予測
# y_pred_prob_increase_20_34_only = model.predict(data_increase_20_34_only, num_iteration=model.best_iteration)
# y_pred_increase_20_34_only = np.where(y_pred_prob_increase_20_34_only > 0.5, 1, 0)
#
# # 全社員(=全行)を母数とした離職率
# attrition_rate_20_34_only = y_pred_increase_20_34_only.mean()
# print(f"Attrition rate (only 20-34 raised): {attrition_rate_20_34_only:.4f}")
#
# # 例: 部分的に給与を上げたシナリオで「20～34歳の離職率」だけを計算
# attrition_rate_20_34_subgroup = y_pred_increase_20_34_only[mask_20_34].mean()
# print(f"Attrition rate in 20-34 age subgroup: {attrition_rate_20_34_subgroup:.4f}")
#
# # 元データをコピー(「Attrition」は落とすがAgeは元のまま)
# data_increase_55_60_only = data.copy()
# data_increase_55_60_only.drop("Attrition", axis=1, inplace=True)
#
# # 55～60歳の行に対してのみ給与を5%上昇
# mask_55_60 = (data_increase_55_60_only["Age"] >= 55) & (data_increase_55_60_only["Age"] <= 60)
# data_increase_55_60_only.loc[mask_55_60, "MonthlyIncome"] *= 1.05
#
# # 変数をエンコード
# for column in data_increase_55_60_only.select_dtypes(include='object').columns:
#     data_increase_55_60_only[column] = label_encoders[column].transform(data_increase_55_60_only[column])
#
# # 予測
# y_pred_prob_increase_55_60_only = model.predict(data_increase_55_60_only, num_iteration=model.best_iteration)
# y_pred_increase_55_60_only = np.where(y_pred_prob_increase_55_60_only > 0.5, 1, 0)
#
# # 全社員(=全行)を母数とした離職率
# attrition_rate_55_60_only = y_pred_increase_55_60_only.mean()
# print(f"Attrition rate (only 55-60 raised): {attrition_rate_55_60_only:.4f}")
#
# # 例: 部分的に給与を上げたシナリオで「20～34歳の離職率」だけを計算
# attrition_rate_55_60_only = y_pred_increase_55_60_only[mask_55_60].mean()
# print(f"Attrition rate in 55-60 age subgroup: {attrition_rate_55_60_only:.4f}")
#
# # 元データをコピー（Attrition列を落とす）
# data_increase_20_34_55_60 = data.copy()
# data_increase_20_34_55_60 = data_increase_20_34_55_60.drop("Attrition", axis=1)
#
# # 20～34歳または55～60歳に該当する行だけ給与を5%上昇
# mask_20_34_55_60 = ((data_increase_20_34_55_60["Age"] >= 20) & (data_increase_20_34_55_60["Age"] <= 34)) | \
#                    ((data_increase_20_34_55_60["Age"] >= 55) & (data_increase_20_34_55_60["Age"] <= 60))
#
# data_increase_20_34_55_60.loc[mask_20_34_55_60, "MonthlyIncome"] *= 1.05
#
# # 予測用にカテゴリ列をエンコード
# for column in data_increase_20_34_55_60.select_dtypes(include='object').columns:
#     data_increase_20_34_55_60[column] = label_encoders[column].transform(data_increase_20_34_55_60[column])
#
# # モデルで予測
# y_pred_prob_increase_20_34_55_60 = model.predict(data_increase_20_34_55_60, num_iteration=model.best_iteration)
# y_pred_increase_20_34_55_60 = np.where(y_pred_prob_increase_20_34_55_60 > 0.5, 1, 0)
#
# # 離職率:全社員を母数として、予測が1(離職)になった人の割合
# attrition_rate_20_34_55_60 = y_pred_increase_20_34_55_60.mean()
# print(f"Attrition rate (5% up for Age 20-34 & 55-60, all employees as denominator): {attrition_rate_20_34_55_60:.4f}")


# X のコピーを作成し、DistanceFromHomeを0に上書き
X_distance = X.copy()
X_distance['DistanceFromHome'] = 0

# 予測
y_pred_prob_distance = model.predict(X_distance, num_iteration=model.best_iteration)
y_pred_distance = np.where(y_pred_prob_distance > 0.5, 1, 0)

# 離職率
attrition_rate_distance = y_pred_distance.mean()
print(f"Attrition rate (DistanceFromHome=0): {attrition_rate_distance:.4f}")

