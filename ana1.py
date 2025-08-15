import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# sns.set_style('whitegrid')
# print(train_data.head())
#
# print(train_data.info())
# print("-" * 40)
# test_data.info()
#
# train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
#
# plt.show()

# print(train_data.Embarked.dropna().mode().values)
# print(train_data.Embarked)

# 处理缺失值
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data['Cabin'] = train_data.Cabin.fillna('U0') # train_data.Cabin[train_data.Cabin.isnull()]='U0'

# print(train_data.info())

# 使用随机森林预测缺失值
from sklearn.ensemble import RandomForestRegressor

#choose training data to predict age
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
# print(age_df.info())
# print("-" * 40)
# print(age_df_isnull.info())
# print("-" * 40)
# print(age_df_notnull.info())
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
print(train_data.info())


# 分析数据关系