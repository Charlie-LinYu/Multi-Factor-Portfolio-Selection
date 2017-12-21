import os
import csv
import pandas as pd
import numpy as np
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
criteria=np.asarray([10,2,1,4,10000,100000,200])
initial=10000000

#criteria=np.asarray([1/7]*7)

hs300=pd.read_csv('hs300.csv').set_index('Date')
trading_date = hs300.index
price_data=pd.read_hdf('price_data.h5', 'key')
factor_data=pd.read_hdf('factor_data.h5', 'key')

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

#res=[-0.0056]  #sharp ratio when u = 1 and w=[1/7]*7
#construct the portfolio regarding different rebalance intervals and
#calculate sharp ratio against different Us
res_opt=[]
for u in range(5,91,5):
    days=0
    while(days<len(insample)-1):
        if days==0:
            tmp_date=insample[days]
            print('Loading '+tmp_date+' ...')
            strat=strategy(securities_dict,criteria)
            strat.calculate_score(tmp_date)
            tmp_res=strat.get_signal(tmp_date)

            p=portfolio(tmp_date,initial,securities_dict,hs300,tmp_res)
            p.get_beta(tmp_date)
            print('...Completed!')
            days=days+u
        else:
            tmp_date=insample[days]
            print('Loading '+tmp_date+' ...')
            strat.calculate_score(tmp_date)
            tmp_res=strat.get_signal(tmp_date)

            p.update(tmp_date,tmp_res)
            p.get_beta(tmp_date)
            p.set_beta_neutral()
            print('...Completed!')
            days=days+u

    p.close_portfolio(insample[-1])
    p.set_beta_neutral()
    p.get_statistics()
    res_opt.append(p.statistics['Sharp ratio'])

#plot the sharp ratios along with different rebalance intervals
index0=[i for i in range(5,91,5)]
u_optimal=pd.DataFrame({'sharp ratio':res_opt},index=index0)
u_optimal.to_csv('u_optimal.csv')
fig=plt.figure()
ax= fig.add_subplot(1,1,1)
ax.plot(u_optimal.index,u_optimal['sharp ratio'])
ax.set_xlabel('Rebalance interval')
ax.set_ylabel('Sharp ratio')
plt.show()
plt.close(fig)

#decided u=30
