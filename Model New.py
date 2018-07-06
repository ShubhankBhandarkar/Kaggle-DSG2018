# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:15:25 2018

@author: shubh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
import gc

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


full = pd.read_csv('input/generated/trade_challenge_new.csv')
full.drop('Unnamed: 0', axis=1, inplace=True)
full = reduce_mem_usage(full)
full = full[full.BuySell == 'Buy']
print(full.shape)
full.head()

# Bond Features
bond_week_feat = pd.read_csv('input/generated/bond_week_feat.csv')
bond_week_feat = reduce_mem_usage(bond_week_feat)
full = full.merge(bond_week_feat.drop_duplicates(), how='left')
del bond_week_feat; gc.collect()

bonds_feat = pd.read_csv('input/generated/bonds_updated.csv')
bonds_feat = reduce_mem_usage(bonds_feat)
full = full.merge(bonds_feat.drop_duplicates(), how='left')
del bonds_feat; gc.collect()

# Customer Features
cust_week_feat = pd.read_csv('input/generated/cust_week_feat.csv')
cust_week_feat = reduce_mem_usage(cust_week_feat)
full = full.merge(cust_week_feat.drop_duplicates(), how='left')
del cust_week_feat; gc.collect()

customer_feat = pd.read_csv('input/generated/customer_feat.csv')
customer_feat = reduce_mem_usage(customer_feat)
full = full.merge(customer_feat.drop_duplicates(), how='left')
del customer_feat; gc.collect()

print('Size of the data is: {0:.2f} GB'.format(sys.getsizeof(full)/1024**3))





full = pd.read_csv('../input/tradechallenge/trade_challenge_new.csv')
full.drop('Unnamed: 0', axis=1, inplace=True)
full = reduce_mem_usage(full)
full = full[full.BuySell == 'Buy']
print(full.shape)
full.head()

# Bond Features
bond_week_feat = pd.read_csv('input/generated/bond_week_feat.csv')
bond_week_feat = reduce_mem_usage(bond_week_feat)
full = full.merge(bond_week_feat.drop_duplicates(), how='left')
del bond_week_feat; gc.collect()

bonds_feat = pd.read_csv('input/generated/bonds_updated.csv')
fdrop = ['IndustrySubgroup', 'Owner']
bonds_feat.drop(fdrop, axis=1, inplace=True)

categorical_feats = list(pd.DataFrame(bonds_feat.dtypes[bonds_feat.dtypes == 'object']).index)
for feature in categorical_feats:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(bonds_feat[feature].astype(str))
    #encoder.fit(full[feature].append(challenge[feature]).astype(str))
    
    bonds_feat[feature] = encoder.transform(bonds_feat[feature].astype(str))

bonds_feat = reduce_mem_usage(bonds_feat)
full = full.merge(bonds_feat.drop_duplicates(), how='left')
del bonds_feat; gc.collect()

print(full.shape)
full.head()

# Customer Features
cust_week_feat = pd.read_csv('input/generated/cust_week_feat.csv')
cust_week_feat = reduce_mem_usage(cust_week_feat)
full = full.merge(cust_week_feat.drop_duplicates(), how='left')
del cust_week_feat; gc.collect()
print(full.shape)

customer_feat = pd.read_csv('input/generated/customer_feat.csv')
fdrop = ['Customer_Country', 'Customer_Subsector']
customer_feat.drop(fdrop, axis=1, inplace=True)

temp_feats = list(pd.DataFrame(customer_feat.dtypes[customer_feat.dtypes == 'object']).index)
for feature in temp_feats:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(customer_feat[feature].astype(str))
    #encoder.fit(full[feature].append(challenge[feature]).astype(str))
    
    customer_feat[feature] = encoder.transform(customer_feat[feature].astype(str))

customer_feat = reduce_mem_usage(customer_feat)
full = full.merge(customer_feat.drop_duplicates(), how='left')
categorical_feats.extend(temp_feats)
print(categorical_feats)
del customer_feat; gc.collect()

print(full.shape)
full.head()

print('Size of the data is: {0:.2f} GB'.format(sys.getsizeof(full)/1024**3))

gc.collect()

'''
bonds = reduce_mem_usage(bonds)
fdrop = ['IndustrySubgroup', 'Owner', 'Customer_Country', 'Customer_Subsector']
bonds.drop(fdrop, axis=1, inplace=True)
'''

temp_feats = list(pd.DataFrame(full.dtypes[full.dtypes == 'object']).index)
temp_feats.remove('PredictionIdx')
for feature in temp_feats:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(full[feature].astype(str))
    #encoder.fit(full[feature].append(challenge[feature]).astype(str))
    
    full[feature] = encoder.transform(full[feature].astype(str))
    
categorical_feats.extend(temp_feats)
print(categorical_feats)

print('Size of the data is: {0:.2f} GB'.format(sys.getsizeof(full)/1024**3))
full = reduce_mem_usage(full)
print('Size of the data is: {0:.2f} GB'.format(sys.getsizeof(full)/1024**3))

categorical_feats.remove('BuySell')

# Splitting Data
train = full[full.PredictionIdx == 'trade'].copy()
train = train[(train.WeekID != 139) & (train.WeekID != 140)]
valid1 = full[(full.WeekID == 139)].copy()
valid2 = full[(full.WeekID == 140)].copy()
test = full[full.WeekID == 141].copy()

print(train.shape)
print(valid1.shape)
print(valid2.shape)
print(test.shape)

fdrop = ['BuySell', 'CustomerIdx', 'IsinIdx', 'WeekID', 'Cust_Previous_WeekID', 'PredictionIdx']
train.drop(fdrop, axis=1, inplace=True)
valid1.drop(fdrop, axis=1, inplace=True)
valid2.drop(fdrop, axis=1, inplace=True)

fdrop = ['BuySell', 'CustomerIdx', 'IsinIdx', 'WeekID', 'Cust_Previous_WeekID', 'CustomerInterest']
test.drop(fdrop, axis=1, inplace=True)

del full; gc.collect()

#===================================================================================================
# Modelling Phase

# LGBM Dataset Formatting 
feat = list(train.columns)
feat.remove('CustomerInterest')
lgtrain = lgb.Dataset(train.loc[:, train.columns != 'CustomerInterest'], train.CustomerInterest,
                      feature_name=feat,
                      categorical_feature = categorical_feats)
lgvalid1 = lgb.Dataset(valid1.loc[:, valid1.columns != 'CustomerInterest'], valid1.CustomerInterest,
                       feature_name=feat,
                       categorical_feature = categorical_feats)
lgvalid2 = lgb.Dataset(valid2.loc[:, valid2.columns != 'CustomerInterest'], valid2.CustomerInterest,
                       feature_name=feat,
                       categorical_feature = categorical_feats)

lgbm_params = {
    'objective' : 'binary',
    'metric': {'binary_logloss','auc'},
    'num_leaves' : 32,
    'max_depth': 12,
    'learning_rate' : 0.018,
    'feature_fraction' : 0.6,
    'bagging_fraction': 0.75,
    'verbosity' : -1,
}

lgb_clf_buy = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=3000,
    valid_sets=[lgtrain, lgvalid1, lgvalid2],
    valid_names=['train','valid1','valid2'],
    early_stopping_rounds=100,
    verbose_eval=50
)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf_buy, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('LGB_feature_import.png')


test['CustomerInterest'] = lgb_clf_buy.predict(test[feat])
test.head()

test.CustomerInterest.describe()

challenge = pd.read_csv('input/original/Challenge_20180423.csv')

submission = test.loc[:,['PredictionIdx','CustomerInterest']]
print(submission.shape)
print(submission.CustomerInterest.isnull().sum())
submission.head()
submission.to_csv('shubh_lgb_2406_1stTrial_bond&custfeat.csv', index=False)

import pickle
filename = 'lgbm_model_2406_1st trial.sav'
pickle.dump(lgb_clf_buy, open(filename, 'wb'))

'''
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
'''

sub1 = pd.read_csv('output/lgb_1706.csv')
sub2 = submission.merge(sub1, on='PredictionIdx')
sub2['CustomerInterest'] = (sub2.CustomerInterest_x + sub2.CustomerInterest_y)/2
fdrop = ['CustomerInterest_x', 'CustomerInterest_y']
sub2.drop(fdrop, axis=1, inplace=True)

sub2.to_csv('shubh_blend_lgb1706_and_lgb_2406_1stTrial_bond&custfeat.csv', index=False)