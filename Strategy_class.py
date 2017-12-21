import pandas as pd
import numpy as np
import math

class strategy:
    '''
    This class is used to generate trading signals based on some strategies
    or criteria
    '''
    def __init__(self, stock_dict, criteria, pm=5, pr=20, volatility=30):
        self.stock=stock_dict
        self.criteria=criteria
        self.pm='PM_'+str(pm)
        self.pr='PR_'+str(pr)
        self.volatility='Volatility_'+str(volatility)

    def calculate_score(self,date):
        '''
        select stocks with required market capital and average volume, and
        then calculate the M-score
        '''
        self.date=date
        self.signal = pd.DataFrame(0,index=[date],columns=self.stock.keys())
        self.subset=[(k,v) for k,v in self.stock.items() if v.validity and
                     v.get_mkt_cap(date)>=500000000 and
                     v.get_ave_vol(date)>=1000000
                     and (date in v.factor.index)]
        self.score=[(k,v.factor.loc[date].append(v.price.loc[date,
                     [self.pm,self.pr,self.volatility]])) for k,v in self.subset]
        self.score=pd.DataFrame(dict(self.score)).T.dropna()
        #linear format
        self.score['Score']=self.score.apply(lambda x: np.inner(np.asarray(x),
                  self.criteria),axis=1)
        #logistic regression
        #self.score['Score']=self.score.apply(lambda x: 1/(1-math.exp(
        #       -np.inner(np.asarray(x),self.criteria))),axis=1)

    def get_signal(self,date):
        '''
        Select 100 stocks with the highest M-scores and generate trading signals.
        This step must be done after calculating M-score with the same date
        '''
        if date==self.date:
            self.selected=self.score.sort_values('Score',ascending=False)[0:100].index
            self.signal[self.selected]=0.01
            return self.signal
        else:
            return print('Error! You should calculate M-scores first.')

