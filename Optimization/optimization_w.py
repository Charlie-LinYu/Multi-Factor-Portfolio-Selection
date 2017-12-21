import os
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime

%matplotlib qt5

path_os="C:\\Academic\\Georgia Tech\\Financial System-C++\\final project\\project\\"
os.chdir(path_os)
from Security_class import security
from Strategy_class import strategy
from Portfolio_class import portfolio

securities_dict={}
starttime_insample = '2011-01-01'
endtime_insample = '2014-10-31'
starttime_outsample = '2014-11-01'
endtime_outsample = '2015-07-31'
criteria=np.asarray([1/7]*7)

hs300=pd.read_csv('hs300.csv').set_index('Date')
price_data=pd.read_hdf('price_data.h5', 'key')
factor_data=pd.read_hdf('factor_data.h5', 'key')
trading_date = hs300.index
insample=[d for d in trading_date if d>=starttime_insample and d<=endtime_insample]
outsample=[d for d in trading_date if d>=starttime_outsample and d<=endtime_outsample]
#Read tickers from csv file
with open('ticker_universe.csv','r') as fin:
    csvin = csv.DictReader(fin)
    security_info = [line for line in csvin]

#use information from csv file to initialize security classes, and store the
#security classes in a dictionary with tickers as the key
i=1
for item in security_info:
    if item['ticker'][-1] == 'H':
        item['ticker'] = item['ticker'][0:-1]+'S'
    securities_dict[item['ticker']]=security(item)
    securities_dict[item['ticker']].read_price(price_data)
    securities_dict[item['ticker']].read_factor(factor_data)
    securities_dict[item['ticker']].calculate_pm(5)
    securities_dict[item['ticker']].calculate_pr(20)
    securities_dict[item['ticker']].calculate_vol(30)
    print(i)
    i=i+1

#u=15
u=30

#calculate the u-period future cumulative returns on a rolling basis of each
#stock and merge together with the seven factors to generate cross-sectional
#data for linear regression or logistic regression
subset=[(k,v) for k,v in securities_dict.items() if v.validity]
for i in range(len(subset)):
    test=subset[i][1]
    predicted_ret=(pd.DataFrame(((test.price['RET']+1).cumprod()/(test.price['RET']+1).
        cumprod().shift(u)).shift(-u)).rename(columns={'RET':'predicted ret'}))
    tmp_reg=predicted_ret.merge(test.price[['PM_5','PR_20','Volatility_30']],
                                left_index=True, right_index=True,how='inner')
    tmp_reg=tmp_reg.merge(test.factor,left_index=True, right_index=True,how='inner').dropna()
    if i==0:
        df_reg=tmp_reg
    else:
        df_reg=df_reg.append(tmp_reg)
    print(i)

df_reg=df_reg.loc[insample].reset_index(drop=True)
#df_reg.to_csv('data for regression.csv')
independent_para=['PB', 'PCF', 'PE', 'PS','PM_5','PR_20','Volatility_30']
#linear regression
res=sm.OLS(df_reg['predicted ret'], df_reg[independent_para]).fit()
res.summary()

#logistic regression to recognize stocks with top 10% cumulative returns
top_10=(df_reg.sort_values('predicted ret',ascending=False)).iloc[0:int(0.1*len(df_reg))]
top_10['mark']=1
df_reg2=df_reg.merge(pd.DataFrame(top_10['mark']),left_index=True,
                      right_index=True,how='left').fillna(0)

res2=sm.Logit(df_reg2['mark'],df_reg2[independent_para]).fit()
res2.summary()

#logistic regression to recognize stocks with top 5% cumulative returns
top_5=(df_reg.sort_values('predicted ret',ascending=False)).iloc[0:int(0.05*len(df_reg))]
top_5['mark']=1
df_reg3=df_reg.merge(pd.DataFrame(top_5['mark']),left_index=True,
                      right_index=True,how='left').fillna(0)

res3=sm.Logit(df_reg3['mark'],df_reg3[independent_para]).fit()
res3.summary()
