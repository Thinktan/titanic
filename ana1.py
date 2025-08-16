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
# (1) 性别与是否生存的关系 Sex
print(train_data.groupby(['Sex','Survived'])['Survived'].count())
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# (2) 船舱等级和生存与否的关系 Pclass
print(train_data.groupby(['Pclass','Survived']).groups.keys())
print(train_data.groupby(['Pclass','Survived'])['Pclass'].count())
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
print(train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())


# (3) 年龄与存活与否的关系 Age

# 分别分析不同等级船舱和不同性别下的年龄分布和生存的关系：
fig, ax = plt.subplots(1, 2, figsize = (18, 8))
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train_data, split=False, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

# 分析总体的年龄分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age', showfliers=True)


# 不同年龄下的生存和非生存的分布情况
# plt.figure(figsize=(12,5))
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# 不同年龄下的平均生存率
# plt.figure(figsize=(12,5))
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data["Age_int"] = train_data["Age"].astype(int)
average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)

# 按照年龄，将乘客划分为儿童、少年、成年和老年，分析四个群体的生还情况
plt.figure(figsize=(12,5))
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
by_age.plot(kind='bar')


# (4) 称呼与存活与否的关系 Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_data['Title'], train_data['Sex']))

# 不同称呼与生存率的关系
# plt.figure(figsize=(12,5))
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

# 观察名字长度和生存率之间存在关系的可能
plt.figure(figsize=(12,5))
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)

# (5) 有无兄弟姐妹和存活与否的关系 SibSp
# 将数据分为有兄弟姐妹的和没有兄弟姐妹的两组：
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')


# (6) 有无父母子女和存活与否的关系 Parch
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')


# (7) 亲友的人数和存活与否的关系 SibSp & Parch
# plt.figure(figsize=(10,5))
fig,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()

# (8) 票价分布和存活与否的关系 Fare
plt.figure(figsize=(10,5))
train_data['Fare'].hist(bins = 70)

train_data.boxplot(column='Fare', by='Pclass', showfliers=False)


# 生存与否与票价均值和方差的关系
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
print(average_fare.info)
print(std_fare.info)
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

# (9) 船舱类型和存活与否的关系 Cabin
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()

# 不同类型的船舱
# create feature for the alphabetical part of the cabin number
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# convert the distinct cabin letters with incremental integer values
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()


# (10) 港口和存活与否的关系 Embarked
plt.figure(figsize=(10,5))
sns.countplot(x='Embarked', hue='Survived', data=train_data)
plt.title('Embarked and Survived')

# sns.factorplot(x='Embarked', y='Survived', data=train_data, size=3, aspect=2)
# plt.title('Embarked and Survived rate')
# plt.show()

plt.show()



# 4. 变量转换
print(train_data.info())
print(train_data.info)































