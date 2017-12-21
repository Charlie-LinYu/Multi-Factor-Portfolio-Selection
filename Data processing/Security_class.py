import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from datetime import datetime

class security:
    '''
    This class is used to store information of a security,
    and it can also be used to download market data
    '''
    def __init__(self,info):
        self.ticker = info['ticker']
        self.marketshare = int(float(info['mktshare'].replace(',','')))
        self.ipo_date = datetime.strptime(info['ipo_date'],'%m/%d/%Y')
        self.validity = True

    def download_data(self, starttime, endtime):
        self.price = yf.download(self.ticker,starttime,endtime)

    def save_data(self,path):
        self.price.to_csv(path+self.ticker+'.csv')

    def read_price(self,panel):
        '''
        read price data from the panel data
        '''
        if self.ticker in panel.items:
            self.price = panel.loc[self.ticker].dropna()
            self.price = self.price.apply(pd.to_numeric)
            if self.price.empty:
                self.validity = False
            else:
                self.price['Log Ret']=np.log(self.price['Adj Close']).diff(1)
                self.price['RET']=self.price['Adj Close']/self.price['Adj Close'].shift(1) - 1
        else:
            self.validity = False

    def calculate_pm(self,n):
        '''
        calculate n period momentum factor
        '''
        if self.validity:
            self.price['PM_'+str(n)]=np.log(self.price['Adj Close'].shift(1)).diff(n)

    def calculate_pr(self,n):
        '''
        calculate n period reversion factor
        '''
        if self.validity:
            self.price['PR_'+str(n)]=-np.log(self.price['Adj Close'].shift(1)).diff(n)

    def calculate_vol(self,n):
        '''
        calculate n period log-return volatility
        '''
        if self.validity:
            self.price['Volatility_'+str(n)]=self.price['Log Ret'].shift(1).rolling(window=n).std()*(250**0.5)

    def read_factor(self,panel):
        '''
        read factor data from the panel data
        '''
        if self.ticker in panel.items:
            self.factor = panel.loc[self.ticker].dropna()
            self.factor = self.factor.apply(pd.to_numeric)
            if self.price.empty:
                self.validity = False
        else:
            self.validity = False

    def get_mkt_cap(self,date):
        '''
        calculate the market capital
        '''
        if self.validity and (date in self.price.index):
            return self.marketshare*self.price.loc[date,'Adj Close']
        else:
            return -1

    def get_ave_vol(self,date):
        '''
        calculate the average volume over the past 15 days
        '''
        if self.validity and (date in self.price.index):
            pos=self.price.index.get_loc(date)
            return self.price.iloc[pos-15:pos]['Volume'].mean()
        else:
            return -1
