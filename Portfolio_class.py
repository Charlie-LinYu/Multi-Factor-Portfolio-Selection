import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

class portfolio:
    '''
    This class is used to process data and store information regrading the portfolio
    based on some strategies. And it also contains functions to evaluate performance
    '''
    def __init__(self, date, value, stock_dict, mkt_index, signal, cost=0.001):
        self.ini_date=date
        self.ini_value=value
        self.cost=cost
        valid_set=[(k,v) for k,v in stock_dict.items() if v.validity]
        self.stock_ret=[(k,v.price['RET']) for k,v in valid_set]
        self.stock_ret=pd.DataFrame(dict(self.stock_ret),index=mkt_index.index).fillna(0)
        self.stock_pool=self.stock_ret.columns
        self.mkt_index=mkt_index
        self.mkt_ret=mkt_index['Adj Close']/mkt_index['Adj Close'].shift(1) - 1
        self.trading_date=mkt_index.index
        self.position=signal
        self.ini_value=self.ini_value*(1-cost)
        self.cum_value=pd.DataFrame([self.ini_value],columns=['cumulative value'],
                                    index=[self.ini_date])
        self.cum_value_betaneutral=self.cum_value.rename(columns=
                                {'cumulative value':'Cumulative Beta-Neutral Value'})
        self.beta_stock={}
        self.beta_portfolio={}
        self.beta_neutral=True
        self.closed=True

    def update(self, new_date, new_signal):
        '''
        Update the portfolio with a new trading signal at a new date. The cumulative
        value between last rebalance day and this new date will be calculated.
        And trading cost is taken into consideration
        '''
        old_date=self.cum_value.index[-1]
        old_signal=self.position.loc[old_date]
        old_value=self.cum_value.loc[old_date]
        self.position=self.position.append(new_signal)

        period=[d for d in self.trading_date if d>old_date and d<=new_date]
        selected_stock=old_signal[old_signal==0.01].T.dropna().index
        selected_ret=self.stock_ret[selected_stock].loc[period]
        self.selected_ret=selected_ret
        selected_cumret=(selected_ret+1).cumprod()

        stock_out=new_signal[(new_signal-old_signal)<0].T.dropna().index
        stock_in=new_signal[(new_signal-old_signal)>0].T.dropna().index
        stock_remain=[ticker for ticker in selected_stock if ticker not in stock_out]

        new_cum_values=pd.DataFrame(selected_cumret.apply(lambda x: np.inner(x,
            old_signal[selected_stock]*float(old_value)),axis=1)).rename(
            columns={0:'cumulative value'})
        self.cum_value=self.cum_value.append(new_cum_values)

        tmp_cost=0
        #cost for selling stocks that are not selected
        tmp_cost=tmp_cost+(selected_cumret.loc[new_date]*
                           float(old_value)*0.01)[stock_out].sum()*self.cost
        #cost for change of weight of stocks remaining as selected
        tmp_cost=tmp_cost+abs((selected_cumret.loc[new_date]*float(old_value)*
                               0.01)[stock_remain]-float(new_cum_values.loc[new_date]
                               -tmp_cost)/100).sum()*self.cost
        #cost for buying stocks that are newly selected
        tmp_cost=tmp_cost+(new_cum_values.loc[new_date]-tmp_cost)/100 * \
            len(stock_in) * self.cost

        self.cum_value.loc[new_date]=self.cum_value.loc[new_date]-tmp_cost

        self.beta_neutral=False

    def get_beta(self, date):
        '''
        calculate the portfolio beta on the input date
        '''
        selected_stock=self.position.loc[date][self.position.loc[date]==0.01] \
            .T.dropna().index
        pos1=self.stock_ret.index.get_loc(date)
        pos2=self.mkt_ret.index.get_loc(date)
        tmp_stock_ret=self.stock_ret[selected_stock].iloc[pos1-100:pos1]
        tmp_mkt_ret=self.mkt_ret.iloc[pos2-100:pos2]

        tmp_beta=[pd.DataFrame(sm.OLS(tmp_stock_ret[k],tmp_mkt_ret).fit().
            params).rename(columns={0:k}).T for k in selected_stock]
        tmp_beta=pd.concat(tmp_beta).rename(columns={'Adj Close':'Beta'})['Beta']
        self.beta_stock[date]=tmp_beta
        self.beta_portfolio[date]=np.inner(tmp_beta,
                                       self.position[selected_stock].loc[date])

    def set_beta_neutral(self):
        '''
        Calculate the cumulative value of the beta neutral portfolio between the
        lastest two rebalance days. This function can only be called once per
        rebalance interval
        '''
        if self.beta_neutral:
            print('You have already set the portfolio to be beta neutral!')
        else:
            old_date=self.position.index[-2]
            new_date=self.position.index[-1]
            old_value=self.cum_value.loc[old_date]
            period=[d for d in self.trading_date if d>old_date and d<=new_date]
            tmp_mkt_ret=pd.DataFrame(self.mkt_ret.loc[period]). \
                rename(columns={'Adj Close':'Cumulative Beta-Neutral Value'})
            tmp_beta=self.beta_portfolio[old_date]
            tmp_ret_beta_neutral=-1*tmp_mkt_ret*tmp_beta+ \
                np.asarray(((self.cum_value/self.cum_value.shift(1))-1).loc[period])
            tmp_value_beta_neutral=pd.DataFrame(tmp_ret_beta_neutral+1).cumprod()* \
                float(old_value)
            self.cum_value_betaneutral=self.cum_value_betaneutral. \
                append(tmp_value_beta_neutral)

            self.beta_neutral=True

    def close_portfolio(self, new_date):
        '''
        Close the portfolio, so that the cumulative value between last rebalance
        day and the close day will be calculated
        '''
        old_date=self.cum_value.index[-1]
        old_signal=self.position.loc[old_date]
        old_value=self.cum_value.loc[old_date]
        self.position.loc[new_date]=self.position.iloc[0]
        self.position.loc[new_date]=0

        period=[d for d in self.trading_date if d>old_date and d<=new_date]
        selected_stock=old_signal[old_signal==0.01].T.dropna().index
        selected_ret=self.stock_ret[selected_stock].loc[period]
        selected_cumret=(selected_ret+1).cumprod()

        new_cum_values=pd.DataFrame(selected_cumret.apply(lambda x: np.inner(x,
            old_signal[selected_stock]*float(old_value)),axis=1)).rename(
            columns={0:'cumulative value'})
        self.cum_value=self.cum_value.append(new_cum_values)
        self.cum_value.iloc[-1]=self.cum_value.iloc[-1]*(1-self.cost)
        self.beta_neutral=False
        self.closed=True

    def get_plot_netvalue(self,title, type_='original'):
        '''
        Plot the cumulateive value of the portfolio with initial value of 1
        '''
        if not self.closed :
            print('Warning: the portfolio is not closed. So the plot may be not intact.')
        fig=plt.figure(figsize=(15,9))
        ax = fig.add_subplot(1,1,1)
        period=[datetime.strptime(t,'%Y-%m-%d') for t in self.cum_value.index]
        period2=[datetime.strptime(t,'%Y-%m-%d') for t in self.mkt_index.index]
        self.mkt_index['date']=period2
        self.mkt_index_new=self.mkt_index.set_index('date')
        if type_=='original':
            ax.plot(period,self.cum_value/self.ini_value,
                    label='cumulative value')
            ax.plot(period,self.mkt_index_new['Adj Close'].loc[period]/
                    float(self.mkt_index_new['Adj Close'].loc[period[0]]),
                    label='market index')
        elif type_=='beta neutral':
            ax.plot(period,self.cum_value_betaneutral/self.ini_value,
                    label='cumulative value')
        else:
            plt.close(fig)
            return print('Wrong type!')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative growth rate')
        ax.set_title(title+' cumulative net value of the '+type_)
        ax.legend()
        plt.show()

    def get_plot_distribution(self,title, type_='original'):
        '''
        Plot the histogram of the distribution of the daily returns
        '''
        if type_=='original':
            tmp_ret=np.asarray((self.cum_value/self.cum_value.shift(1)-1).iloc[1:])
        elif type_=='beta neutral':
            tmp_ret=np.asarray((self.cum_value_betaneutral/
                                self.cum_value_betaneutral.shift(1)-1).iloc[1:])
        else:
            return print('Wrong type!')
        fig=plt.figure(figsize=(15,9))
        ax= fig.add_subplot(1,1,1)
        tmp_info=ax.hist(tmp_ret, 50, density=1, edgecolor='black',facecolor='skyblue', alpha=0.75)
        ax.set_xlabel('Daily return')
        ax.set_ylabel('Density')
        ax.set_title(title+' distribution of daily return of the '+type_)
        plt.show()

    def get_statistics(self,title,type_='original'):
        '''
        Get the descriptive statistics of the portfolio.
        '''
        if type_=='original':
            cumvalue=self.cum_value
        elif type_=='beta neutral':
            cumvalue=self.cum_value_betaneutral
        else:
            return print('Wrong type!')
        self.statistics={}
        self.date_traded=self.position.index
        self.statistics['Total PnL']=float(cumvalue.iloc[-1]-self.ini_value)
        self.pnl=cumvalue.loc[self.date_traded].diff(1)
        self.pnl.iloc[0]=-self.ini_value*self.cost
        self.statistics['Average PnL']=float(self.pnl.mean())
        self.statistics['Percentage of trades being winning']= \
            len(self.pnl[self.pnl>=0].dropna())/len(self.pnl)
        self.portfolio_ret=cumvalue/cumvalue.shift(1)-1
        self.portfolio_ret.iloc[0]=-self.cost
        self.statistics['Annualized average return']=float(self.portfolio_ret.mean())*250
        self.statistics['Annualized standard deviation']=float(self.portfolio_ret.std())*(250**0.5)
        self.statistics['Sharp ratio']=self.statistics['Annualized average return'] \
            / self.statistics['Annualized standard deviation']
        mdd_end_idx = np.argmax(np.maximum.accumulate(np.asarray(cumvalue))
            - np.asarray(cumvalue))
        mdd_start_idx = np.argmax(np.asarray(cumvalue.iloc[:mdd_end_idx]))
        self.statistics['Maximum drawdown'] = float(cumvalue.iloc[mdd_end_idx]
            - cumvalue.iloc[mdd_start_idx])
        self.statistics['Initial value']=self.ini_value
        self.perf_table = pd.DataFrame(data=list(self.statistics.values()),
                                       index=list(self.statistics.keys())).reindex(
                                       ['Initial value','Total PnL',
                                        'Average PnL','Percentage of trades being winning',
                                        'Annualized average return',
                                        'Annualized standard deviation',
                                        'Sharp ratio','Maximum drawdown']).rename(
                                        columns={0:title+' performance results of the '+type_})
        self.perf_table.iloc[0]='%.2f' % self.perf_table.iloc[0]
        self.perf_table.iloc[1]='%.2f' % self.perf_table.iloc[1]
        self.perf_table.iloc[2]='%.2f' % self.perf_table.iloc[2]
        self.perf_table.iloc[3]='%.2f' % (self.perf_table.iloc[3]*100) +'%'
        self.perf_table.iloc[4]='%.2f' % (self.perf_table.iloc[4]*100) +'%'
        self.perf_table.iloc[5]='%.2f' % (self.perf_table.iloc[5]*100) +'%'
        self.perf_table.iloc[6]='%.2f' % self.perf_table.iloc[6]
        self.perf_table.iloc[7]='%.2f' % self.perf_table.iloc[7]
        print(self.perf_table)
        return self.perf_table
