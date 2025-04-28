from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pandas as pd

# データの読み込み
file_path = 'C:\GCI\data.csv'
data = pd.read_csv(file_path)

# データの基本情報を表示
# data_info = {
#     "head": data.head(),
#     "info": data.info(),
#     "missing_values": data.isnull().sum()
# }

# ターゲット変数と特徴量を設定
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBMデータセット
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# モデルパラメータ
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

# モデル学習
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=50)

# 予測と評価
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, conf_matrix, classification_rep

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# ラベルエンコーディング
encoded_data = data.copy()
label_encoders = {}

for column in encoded_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(encoded_data[column])
    label_encoders[column] = le

# ターゲット変数と特徴量の分割
X = encoded_data.drop(columns=[target_column])
y = encoded_data[target_column]

# ランダムフォレストモデルの訓練
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# 特徴量重要度の取得
feature_importances = rf_model.feature_importances_
features = X.columns

# 特徴量重要度をデータフレームに変換
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 上位10特徴量をプロット
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Top 10 Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

importance_df.head(10)

# # ターゲット変数の再定義
# target_column = 'Attrition'
#
# # 特徴量とターゲットの分割を再設定
# X = encoded_data.drop(columns=[target_column])
# y = encoded_data[target_column]
#
# # ランダムフォレストモデルの訓練
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X, y)
#
# # 特徴量重要度の取得
# feature_importances = rf_model.feature_importances_
# features = X.columns
#
# # 特徴量重要度をデータフレームに変換
# importance_df = pd.DataFrame({
#     'Feature': features,
#     'Importance': feature_importances
# }).sort_values(by='Importance', ascending=False)
#
# # 上位10特徴量をプロット
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
# plt.gca().invert_yaxis()
# plt.title('Top 10 Important Features')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()
#
# importance_df.head(10)
#
