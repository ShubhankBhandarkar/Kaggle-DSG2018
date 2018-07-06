# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:26:57 2018

@author: shubh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:22:49 2018

@author: shubh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
import gc

path_input = 'input/'
#path_input = ''

'''
trade = pd.read_csv(path_input + 'Trade.csv', parse_dates=['TradeDateKey'])
trade.rename(columns = {'TradeDateKey':'DateKey'},inplace = True)
trade.head()


import datetime
import calendar
def week_of_month(tgtdate):
    tgtdate = tgtdate.to_pydatetime()

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1

trade['Year'] = trade['DateKey'].dt.year
trade['Month'] = trade['DateKey'].dt.month
trade['Week_of_Month'] = trade['DateKey'].apply(week_of_month)
trade['TimeHash'] = trade.Year*1000 + trade.Month*10 + trade.Week_of_Month
timehash_weekid = pd.read_csv(path_input + 'TimeHash-WeekID.csv', usecols=['TimeHash', 'WeekID'])
trade = trade.merge(timehash_weekid)
trade['Prev_WeekID'] = trade.WeekID-1
trade.head()

#trade['Hash'] = trade.CustomerIdx.astype(str) + trade.IsinIdx.astype(str) + trade.BuySell + trade.WeekID.astype(str)
trade = trade[trade.TradeStatus != 'Holding']
trade.shape
trade.to_csv(path_input+'positive_trade_cleaned.csv', index=False)
'''

trade = pd.read_csv('input/intermediate/positive_trade_cleaned.csv', 
                    usecols = ['WeekID', 'CustomerIdx', 'IsinIdx', 'BuySell'])

# Challenge Data
challenge = pd.read_csv('input/original/Challenge_20180423.csv', parse_dates = ['DateKey'])
challenge.head()

trade_buy = trade[trade.BuySell == 'Buy'].copy()
trade_sell = trade[trade.BuySell == 'Sell'].copy()

unique_cust_bond = challenge.drop_duplicates(['CustomerIdx', 'IsinIdx'])[['CustomerIdx', 'IsinIdx']]
print(unique_cust_bond.shape)
unique_cust_bond.head()
del challenge; gc.collect()

a = pd.DataFrame({'WeekID': range(1, 141)})
a['key'] = 1
unique_cust_bond['key'] = 1

extended_trade_buy = pd.merge(unique_cust_bond, a, on='key')[['CustomerIdx', 'IsinIdx', 'WeekID']]
extended_trade_sell = extended_trade_buy.copy()

extended_trade_buy = extended_trade_buy.merge(trade_buy, how='left')
extended_trade_sell = extended_trade_sell.merge(trade_sell, how='left')
print(extended_trade_buy.shape)
extended_trade_buy['BuySell'] = 'Buy'
extended_trade_sell['BuySell'] = 'Sell'

extended_trade_buy = extended_trade_buy[extended_trade_buy.WeekID >= 100]
extended_trade_sell = extended_trade_sell[extended_trade_sell.WeekID >= 100]

generated_keys = pd.read_csv('input/generated/generated_keys.csv')
generated_keys = generated_keys[generated_keys.WeekID < 100]
generated_keys.drop('eval', axis=1,inplace=True)

generated_keys = pd.concat([generated_keys, pd.concat([extended_trade_buy, extended_trade_sell])])
generated_keys.drop_duplicates(inplace=True)

trade_new = pd.concat([trade, generated_keys])
trade_new.drop_duplicates(inplace=True)
trade_new.to_csv('input/generated/trade_challenge_keys.csv', index=False)













trade = pd.read_csv('input/intermediate/positive_trade_cleaned.csv')

# Challenge Data
challenge = pd.read_csv('input/original/Challenge_20180423.csv', parse_dates = ['DateKey'])
challenge['Year'] = 2018
challenge['Month'] = 4
challenge['Week_of_Month'] = 4
#challenge['TimeHash'] = 2018044
challenge['WeekID'] = 141
#challenge['Prev_WeekID'] = 140
challenge.drop(['DateKey'], axis=1, inplace=True)
challenge.head()


full = pd.read_csv('input/generated/trade_challenge_keys.csv')

timehash_weekid = pd.read_csv('input/generated/TimeHash-WeekID.csv')
timehash_weekid.drop('TimeHash', axis=1, inplace=True)
full = full.merge(timehash_weekid, how='left', on='WeekID')

feat = ['CustomerIdx', 'IsinIdx', 'BuySell', 'CustomerInterest', 'WeekID']
full = full.merge(trade[feat].drop_duplicates(), how='left')
full.CustomerInterest.fillna(0, inplace=True)


trade_challenge = pd.concat([full, challenge])
trade_challenge['PredictionIdx'].fillna('trade', inplace=True)

trade_challenge.to_csv('input/generated/trade_challenge_new.csv')


temp = extended_trade_buy.groupby(['WeekID', 'CustomerIdx']).count()
temp = temp.reset_index()[['WeekID', 'CustomerIdx', 'IsinIdx', 'BuySell']]
temp.columns = ['WeekID', 'CustomerIdx', 'nTotalBond', 'nActualBond']
print(temp.shape)
temp.head()


from tqdm import tqdm
count = []
for i in tqdm(range(temp.shape[0])):
    if temp.loc[i,'nActualBond'] == 0:
        if temp.loc[i,'nTotalBond'] > 4:
            count.append(4)
        else: count.append(temp.loc[i,'nTotalBond'])
    elif temp.loc[i,'nActualBond'] < 20:
        if temp.loc[i,'nTotalBond'] >= 3*temp.loc[i,'nActualBond']:
            count.append(3*temp.loc[i,'nActualBond'])
        else: count.append(temp.loc[i,'nTotalBond'])
    else:
        if temp.loc[i,'nTotalBond'] >= 2*temp.loc[i,'nActualBond']:
            count.append(np.round(2*temp.loc[i,'nActualBond']).astype(int))
        else: count.append(temp.loc[i,'nTotalBond'])
            
temp['NewBondNum'] = count
temp.NewBondNum.sum()

temp['TempHash'] = temp.CustomerIdx*10000000 + temp.WeekID
extended_trade_buy['TempHash'] = extended_trade_buy.CustomerIdx*10000000 + extended_trade_buy.WeekID
temp.head()

gp = extended_trade_buy.groupby('TempHash', as_index = False)
new_gen_df = pd.DataFrame(columns=extended_trade_buy.columns)

for group in tqdm(gp):
    a = group[1]['TempHash'].head(1)
    a.reset_index(drop=True,inplace=True)
    
    b = temp[temp.TempHash == a[0]].NewBondNum.reset_index(drop=True)
    
    c = group[1].sample(b[0], replace=True)
    new_gen_df = new_gen_df.append(c, ignore_index=True)
    
del temp, gen_CI_0, group, i, a,b,c,count; gc.collect()    

import pickle
# Writing pickle files
with open('NewGenerated1.pkl', 'wb') as f:
    pickle.dump(new_gen_df, f, pickle.HIGHEST_PROTOCOL)
    
new_gen_df.to_csv(path_input+'negative_samples_raw.csv', index=False)

new_gen_df.drop_duplicates(inplace=True)

def splityear(integer):
    a = int(str(integer)[0:4])
    return a

def splitmonth(integer):
    a = int(str(integer)[4:6])
    return a

def splitweek(integer):
    a = int(str(integer)[6:7])
    return a

new_gen_df['Week_of_Month'] = new_gen_df['TimeHash'].apply(splitweek)
new_gen_df['Year'] = new_gen_df['TimeHash'].apply(splityear)
new_gen_df['Month'] = new_gen_df['TimeHash'].apply(splitmonth)
new_gen_df['TimeHash'] = new_gen_df.Year*1000 + new_gen_df.Month*10 + new_gen_df.Week_of_Month
timehash_weekid = pd.read_csv(path_input + 'TimeHash-WeekID.csv', usecols=['TimeHash', 'WeekID'])
new_gen_df = new_gen_df.merge(timehash_weekid)
new_gen_df['Prev_WeekID'] = new_gen_df.WeekID-1
new_gen_df['CustomerInterest'] = 0
new_gen_df.drop('TempHash', axis=1, inplace=True)
new_gen_df.head()
new_gen_df.to_csv(path_input+'negative_samples_cleaned.csv', index=False)

trade = pd.read_csv(path_input+'positive_trade_cleaned.csv', parse_dates=['DateKey'])
trade.shape
trade.drop_duplicates(inplace=True)
trade.head()

new_trade = pd.concat([trade, new_gen_df])
new_trade.to_csv(path_input+ 'new_trade_uncleaned.csv', index=False)

new_trade.head()
new_trade.isnull().sum()


# Challenge Data
challenge = pd.read_csv(path_input + 'Challenge_20180423.csv', parse_dates = ['DateKey'])
challenge['Year'] = 2018
challenge['Month'] = 4
challenge['Week_of_Month'] = 4
challenge['TimeHash'] = 2018044
challenge['WeekID'] = 141
challenge['Prev_WeekID'] = 140
#challenge['Hash'] = challenge.CustomerIdx.astype(str) + challenge.IsinIdx.astype(str) + challenge.BuySell + challenge.TimeHash.astype(str)
#challenge.drop(['DateKey', 'CustomerInterest'], axis=1, inplace=True)
challenge.head()

challenge.columns
new_trade.columns

trade_challenge = pd.concat([new_trade, challenge])
trade_challenge.drop(['TradeStatus'], axis=1, inplace=True)
trade_challenge.PredictionIdx.fillna('trade', inplace=True)

#trade_challenge[(trade_challenge.PredictionIdx.isnull()) & trade_challenge.CustomerInterest == 1].PredictionIdx.fillna('trade_positive').head()
trade_challenge.to_csv(path_input + 'trade_challenge.csv' ,index=False)






# Removing Repeated rows in positive samples and taking average NotionalEUR and Price
trade = pd.read_csv('input/generated/trade_challenge.csv')
trade.shape

clist = ['BuySell', 'CustomerIdx', 'CustomerInterest', 'IsinIdx',
       'Month', 'PredictionIdx', 'Prev_WeekID',
       'TimeHash', 'WeekID', 'Week_of_Month', 'Year']

trade[clist].drop_duplicates().shape
trade[['BuySell', 'CustomerIdx', 'IsinIdx', 'WeekID']].drop_duplicates().shape
trade.columns
fdrop = ['DateKey', 'TimeHash']
trade.drop(fdrop, axis=1, inplace=True)
keys = ['BuySell', 'CustomerIdx', 'IsinIdx', 'WeekID']
temp = trade.groupby(keys)['NotionalEUR', 'Price'].mean()
temp.reset_index(inplace=True)
temp.head()
trade.drop(['NotionalEUR', 'Price'], axis=1, inplace=True)
trade.drop_duplicates(inplace=True)
trade = trade.merge(temp, how='left')
trade.head()
trade.to_csv('input/generated/trade_challenge.csv', index=False)


# Creating Validation Set
outer_joined = pd.read_csv('input/intermediate/extensive_trade_29463420shape.csv')
temp = trade[trade.WeekID == 141].copy()
temp = temp[['BuySell', 'CustomerIdx', 'IsinIdx', 'WeekID']]
temp.WeekID = 140
temp = temp.merge(trade, how='left')



'''
temp1 = trade.groupby(keys)['NotionalEUR', 'Price'].min()
temp1.reset_index(inplace=True)
temp['Min_TradePrice'] = temp1.Price
temp['Min_NotionalEUR'] = temp1.NotionalEUR
temp1 = trade.groupby(keys)['NotionalEUR', 'Price'].max()
temp1.reset_index(inplace=True)
temp['Max_TradePrice'] = temp1.Price
temp['Max_NotionalEUR'] = temp1.NotionalEUR
temp.head()

plt.title('Correlation between different fearures')
sns.heatmap(temp.corr(), annot = True, cmap = 'RdYlGn', vmax = 1,
            linewidths=0.2, fmt = '.2f', xticklabels=True, yticklabels=True) 
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
'''













# Sell
trade = pd.read_csv('input/intermediate/positive_trade_cleaned.csv', 
                    usecols = ['WeekID', 'CustomerIdx', 'IsinIdx', 'BuySell'])

# Challenge Data
challenge = pd.read_csv('input/original/Challenge_20180423.csv', parse_dates = ['DateKey'])
challenge.head()

trade_sell = trade[trade.BuySell == 'Sell'].copy()

unique_cust_bond = challenge.drop_duplicates(['CustomerIdx', 'IsinIdx'])[['CustomerIdx', 'IsinIdx']]
print(unique_cust_bond.shape)
unique_cust_bond.head()
del trade, challenge; gc.collect()

a = pd.DataFrame({'WeekID': range(1, 141)})
a['key'] = 1
unique_cust_bond['key'] = 1

extended_trade_sell = pd.merge(unique_cust_bond, a, on='key')[['CustomerIdx', 'IsinIdx', 'WeekID']]
extended_trade_sell = extended_trade_sell.merge(trade_sell, how='left')
print(extended_trade_sell.shape)
del a, unique_cust_bond; gc.collect()

temp = extended_trade_sell.groupby(['WeekID', 'CustomerIdx']).count()
temp = temp.reset_index()[['WeekID', 'CustomerIdx', 'IsinIdx', 'BuySell']]
temp.columns = ['WeekID', 'CustomerIdx', 'nTotalBond', 'nActualBond']
print(temp.shape)
temp.head()


from tqdm import tqdm
count = []
for i in tqdm(range(temp.shape[0])):
    if temp.loc[i,'nActualBond'] == 0:
        if temp.loc[i,'nTotalBond'] > 4:
            count.append(4)
        else: count.append(temp.loc[i,'nTotalBond'])
    elif temp.loc[i,'nActualBond'] < 20:
        if temp.loc[i,'nTotalBond'] >= 3*temp.loc[i,'nActualBond']:
            count.append(3*temp.loc[i,'nActualBond'])
        else: count.append(temp.loc[i,'nTotalBond'])
    else:
        if temp.loc[i,'nTotalBond'] >= 2*temp.loc[i,'nActualBond']:
            count.append(np.round(2*temp.loc[i,'nActualBond']).astype(int))
        else: count.append(temp.loc[i,'nTotalBond'])
            
temp['NewBondNum'] = count
temp.NewBondNum.sum()

temp['TempHash'] = temp.CustomerIdx*10000000 + temp.WeekID
extended_trade_sell['TempHash'] = extended_trade_sell.CustomerIdx*10000000 + extended_trade_sell.WeekID
temp.head()

gp = extended_trade_sell.groupby('TempHash', as_index = False)
new_gen_df = pd.DataFrame(columns=extended_trade_sell.columns)

for group in tqdm(gp):
    a = group[1]['TempHash'].head(1)
    a.reset_index(drop=True,inplace=True)
    
    b = temp[temp.TempHash == a[0]].NewBondNum.reset_index(drop=True)
    
    c = group[1].sample(b[0], replace=True)
    new_gen_df = new_gen_df.append(c, ignore_index=True)



    
del temp, gen_CI_0, group, i, a,b,c,count; gc.collect()    

import pickle
# Writing pickle files
with open('NewGenerated1.pkl', 'wb') as f:
    pickle.dump(new_gen_df, f, pickle.HIGHEST_PROTOCOL)
    
new_gen_df.to_csv('negative_samples_sell.csv', index=False)

new_gen_df.drop_duplicates(inplace=True)
new_gen_df.drop('TempHash', axis = 1, inplace=True)
new_gen_df['BuySell'] = 'Sell'

new_gen_df.to_csv('negative_samples_sell.csv', index=False)
    
trade_sell.head()

generated_keys = pd.concat([trade_sell, new_gen_df])

generated_keys.to_csv('input/intermediate/generated_keys_sell.csv', index=False)

valid_sell = extended_trade_sell[(extended_trade_sell.WeekID == 140) | (extended_trade_sell.WeekID == 139)]

generated_keys['eval'] = 'train'


valid_sell.drop_duplicates(inplace=True)
valid_sell.drop('TempHash', axis = 1, inplace=True)
valid_sell['BuySell'] = 'Sell'
valid_sell['eval'] = 'valid'

generated_keys = pd.concat([generated_keys, valid_sell])

generated_keys_buy = pd.read_csv('input/intermediate/generated_keys_buy.csv')
generated_keys = pd.concat([generated_keys, generated_keys_buy])

generated_keys.to_csv('input/intermediate/generated_keys.csv')




