#### KUTUPHANELERIN IMPORT EDILMESI VE DATALARIN YUKLENMESI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print(train.head())

print(train.shape, test.shape)

print(train.info())
print(train.dtypes.value_counts(), test.dtypes.value_counts())


print(train.describe())


#### GENEL BAKIS ICIN BAZI GRAFIKLER

train.plot(kind="scatter", x="GrLivArea", y="SalePrice")
train.plot(kind="scatter", x="TotalBsmtSF", y="SalePrice")

corr_matrix = train.corr()
print(corr_matrix["SalePrice"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
scatter_matrix(train[cols], figsize=(12, 8))


#### OUTLIARS VE EKSIK VERILERIN HALLEDILMESI
# Outliars

#Outliars Before
sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True,
            color = 'red', marker = 'o', scatter_kws={'s':2} )
plt.show()

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train.drop(train[(train['GrLivArea']>5000) | (train['SalePrice']>500000)].index)

#Outliars After
sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True,
            color = 'red', marker = 'o', scatter_kws={'s':2})
plt.show()

# Missing Data

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)


# ###Normallik testi
## Distribution of Target variable (SalePrice)
plt.figure(figsize=(8,6))
sns.distplot(train['SalePrice'], hist_kws={"edgecolor": '#aaff00'},
             kde_kws={'color': 'k', 'lw':3})
plt.show()


#Skew and kurtosis for SalePrice
print(f"Skewness: {train['SalePrice'].skew()}")
print(f"Kurtosis: {train['SalePrice'].kurt()}")


train['SalePrice'] = np.log1p(train['SalePrice'])

plt.figure(figsize=(8,6))
sns.distplot(train['SalePrice'], hist_kws={"edgecolor": '#aaff00'},
             kde_kws={'color': 'k', 'lw':3})
plt.show()

print(f"Skewness: {train['SalePrice'].skew()}")
print(f"Kurtosis: {train['SalePrice'].kurt()}")


####Bazı Manipulasyonlar


train_y = train['SalePrice']
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data = all_data.drop(['SalePrice'], axis=1)

all_data = pd.get_dummies(all_data)



train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]



# Model Kurma
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


xgb_m = xgb.XGBRegressor(learning_rate=0.05, max_depth=3,
                             min_child_weight=2, n_estimators=1500,
                             subsample=0.5,
                             random_state =1, nthread = 1)

train_X, test_X, train_y, test_y = train_test_split(train, train_y)


xgb_m.fit(train_X, train_y, early_stopping_rounds=5,
             eval_set=[(test_X, test_y)], verbose=False)

pred_xgb_m = xgb_m.predict(test_X)
print ("RMSE XGBoost :", np.sqrt(mean_absolute_error(test_y, pred_xgb_m)))

pred_xgb_m = xgb_m.predict(test)


submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = np.expm1(pred_xgb_m)
submission.to_csv('submission.csv',index=False)
print(len(submission))





