import os
import csv
import pandas as pd
import fix_yahoo_finance as yf


path_os="C:\\Academic\\Georgia Tech\\Financial System-C++\\final project\\project\\"
path_price = "C:\\Academic\\Georgia Tech\\Financial System-C++\\final project\\project\\price_data\\"
os.chdir(path_os)
from Security_class import security

securities_dict={}
starttime = '2010-01-01'
endtime = '2016-12-31'
datasource = 'yahoo'

#Read tickers from csv file
with open('ticker_universe.csv','r') as fin:
    csvin = csv.DictReader(fin)
    security_info = [line for line in csvin]

#use information from csv file to initialize security classes, and store the
#security classes in a dictionary with tickers as the key
for item in security_info:
    if item['ticker'][-1] == 'H':
        item['ticker'] = item['ticker'][0:-1]+'S'
    securities_dict[item['ticker']]=security(item)
i=1
for stock in securities_dict.values():
    stock.download_data(starttime,endtime)
    stock.save_data(path_price)
    print(i)
    i=i+1

hs_300=yf.download('000300.SS',starttime,endtime)
hs_300.to_csv(path_os+'hs300.csv')

