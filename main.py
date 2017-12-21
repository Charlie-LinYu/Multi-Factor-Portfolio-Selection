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
#criteria=np.asarray([-0.0003,-1.458e-06,-4.283e-06,-7.203e-05,0,-0.1413, -4.9825])
u=30

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

#insample test
days=0
while(days<len(insample)-1):
    if days==0:
        tmp_date=insample[days]
        print('Loading '+tmp_date+' ...')
        strat_in=strategy(securities_dict,criteria)
        strat_in.calculate_score(tmp_date)
        tmp_res=strat_in.get_signal(tmp_date)

        p_in=portfolio(tmp_date,initial,securities_dict,hs300,tmp_res)
        p_in.get_beta(tmp_date)
        print('...Completed!')
        days=days+u
    else:
        tmp_date=insample[days]
        print('Loading '+tmp_date+' ...')
        strat_in.calculate_score(tmp_date)
        tmp_res=strat_in.get_signal(tmp_date)

        p_in.update(tmp_date,tmp_res)
        p_in.get_beta(tmp_date)
        p_in.set_beta_neutral()
        print('...Completed!')
        days=days+u
p_in.close_portfolio(insample[-1])
p_in.set_beta_neutral()

#outsample test
days=0
while(days<len(outsample)-1):
    if days==0:
        tmp_date=outsample[days]
        print('Loading '+tmp_date+' ...')
        strat_out=strategy(securities_dict,criteria)
        strat_out.calculate_score(tmp_date)
        tmp_res=strat_out.get_signal(tmp_date)

        p_out=portfolio(tmp_date,initial,securities_dict,hs300,tmp_res)
        p_out.get_beta(tmp_date)
        print('...Completed!')
        days=days+u
    else:
        tmp_date=outsample[days]
        print('Loading '+tmp_date+' ...')
        strat_out.calculate_score(tmp_date)
        tmp_res=strat_out.get_signal(tmp_date)

        p_out.update(tmp_date,tmp_res)
        p_out.get_beta(tmp_date)
        p_out.set_beta_neutral()
        print('...Completed!')
        days=days+u
p_out.close_portfolio(outsample[-1])
p_out.set_beta_neutral()

#show results
res_in1=p_in.get_statistics('Insample','original')
res_in2=p_in.get_statistics('Insample','beta neutral')
p_in.get_plot_netvalue('Insample','original')
p_in.get_plot_netvalue('Insample','beta neutral')
p_in.get_plot_distribution('Insample','original')
p_in.get_plot_distribution('Insample','beta neutral')

res_out1=p_out.get_statistics('Outsample','original')
res_out2=p_out.get_statistics('Outsample','beta neutral')
p_out.get_plot_netvalue('Outsample','original')
p_out.get_plot_netvalue('Outsample','beta neutral')
p_out.get_plot_distribution('Outsample','original')
p_out.get_plot_distribution('Outsample','beta neutral')

performance_res=res_in1.merge(res_in2,left_index=True, right_index=True).merge(
        res_out1,left_index=True, right_index=True).merge(
                res_out2,left_index=True, right_index=True)
performance_res.to_csv('performance result.csv')
