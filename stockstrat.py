# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 08:01:32 2015

@author: djunh
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pandas.io.data as web
import random; random.seed(0)
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def get_px(stock, start, end):
    '''
    Takes a stock ticker, start and end date and will return time delimited
    price data.  
    
    Parameters:
    stock- Stock listing, e.g MSFT
    start-  Start date for data collection
    end -End date for data collection
    '''
    return web.get_data_yahoo(stock,start,end)['Adj Close']
    
def calc_drawdown(stock,window,option='day_dd'):
    '''
    Takes a stock, and a time window - calculates the daily drawdown and the 
    maximum daily drawdowns within specified time bandwidth. 
    
    Parameters:
    stock-  The stock listing, e.g XOM
    window- a time period denoted in days
    option- returns the daily drawdown percentage, or maximum
    '''    
    roll_max = pd.rolling_max(px[stock], window, min_periods=1)
    daily_drawdown = px[stock]/roll_max - 1.0
    max_daily_drawdown = pd.rolling_min(daily_drawdown, window, min_periods=1)
    if option=='day_dd':
        return daily_drawdown
    if option=='max_dd':
        return max_daily_drawdown
        
def calc_drawdown_length(stock):
    '''
    Calculates how long a drawdown occurs. Looks at difference between rolling 
    max, and current rolling minimum values and determines the total length.
    
    
    Parameters:
    stock-  The stock listing, e.g XOM
    window- a time period denoted in days
    option- returns the daily drawdown percentage, or maximum
    '''
    px_dd=get_px(stock,'1/1/2012','11/24/2013')
    
    highwatermark=[0]
    days_dd=[]
    
    idx=px_dd.index
    drawdown = pd.Series(index = idx)
    drawdown_time = pd.Series(index = idx)
    
    for time in range(1,len(idx)):
        highwatermark.append(max(highwatermark[time-1],px_dd[time]))
        drawdown[time]=(highwatermark[time]-px_dd[time])
        drawdown_time[time]=(0 if drawdown[time]==0 else drawdown_time[time-1]+1)
        
        if drawdown_time[time] ==0 and drawdown_time[time-1]!=0:
           days_dd.append(drawdown_time[time-1]) 
    
    return drawdown_time,days_dd
    
def calc_drawdown_local(stock,window):
    '''
    Function calculates the duration of the current rolling drawdown. Returns 
    the values of the number of days a drawdown exists.  
    
    Parameters:
    stock-  The stock listing, e.g XOM
    window- a time period denoted in days
    '''
    
    px_dd_max=calc_drawdown(stock,window,option='max_dd')
    days_dd=[]
    
    idx=px_dd_max.index
    drawdown = pd.Series(index = idx)
    drawdown_time = pd.Series(index = idx)
    
    for t in range(1,len(idx)):

        if abs(px_dd_max[t-1])==abs(px_dd_max[t]):
           drawdown[t]=1
        else:
           drawdown[t]=0
         
        drawdown_time[t]=(0 if drawdown[t]==0 else drawdown_time[t-1]+1)          
                    
        if drawdown_time[t] ==0 and drawdown_time[t-1]!=0:
           days_dd.append(drawdown_time[t-1]) 
    
    return days_dd        
        
def create_dataframes(window):
    '''
    Creates and concatinates the required data frames based on the number of time
    windows the analysis requires.
    
    Parameters:
    window-a list of time windows for the analysis.
    '''    
    arr_df=[]
    
    for idx,win in enumerate(window):
        df_string='px_drawdown'+str(win)
        df_string=DataFrame({tkr:calc_drawdown(tkr,win,'day_dd') for tkr in tickers})      
        arr_df.append(df_string)
        
    return pd.concat(arr_df,axis=1,keys=window,names=['Number_days','stock'])
    
if __name__ == "__main__":
    #Set portfolio
    tickers=['NOV','AAPL','WAT','COH']    
    
    #Read the data into a dataFrame
    px=DataFrame({n:get_px(n,'1/1/2009','11/24/2015') for n in tickers})
    
    '''
    Here, we will analyze drawdowns for 3 distinct periods, Yearly, Quarterly
    and monthly.  This data will be subsequently analyzed
    '''
    
    #Set the desired time periods
    windows=[21,63,252]
    
    #Drawdowns should be positive and expressed as a percentage.
    df_drawdowns=create_dataframes(windows)*-100
    
    #For each stock, provide a statistical overview of the data
    df_stat_desc=df_drawdowns.groupby(level='stock', axis=1).describe()
    
    #Drawdowns on yearly basis
    yr_mean=q_mean = df_drawdowns.resample('A-DEC', how='mean', kind='period')
    dd_yearly_mean=df_drawdowns.groupby(df_drawdowns.index.year).mean()
    dd_yearly_std=df_drawdowns.groupby(df_drawdowns.index.year).std()
    
    #Drawdowns on quarterly basis
    qtr_mean = df_drawdowns.resample('Q-NOV', how='mean', kind='period')
    dd_qtr_mean=df_drawdowns.groupby(df_drawdowns.index.quarter).mean()
    dd_qtr_std=df_drawdowns.groupby(df_drawdowns.index.quarter).std()
     
    #Look at drawdowns on a monthly basis
    mth_mean=df_drawdowns.resample('M', how='mean',kind='period')
    dd_monthly_mean=df_drawdowns.groupby(df_drawdowns.index.month).mean()
    dd_monthly_std=df_drawdowns.groupby(df_drawdowns.index.month).std()
    
    #Look at one year-2014
    dd_2014=df_drawdowns['2014-01-01':'2014-12-31']
    dd_2014_ri=dd_2014.mean().reset_index(name='Average Drawdown in 2014')
    
    #Creates histograms based on drawdown magnitudes
    bins_dd = np.linspace(0,30,61)
    dd_hist=df_drawdowns
    dd_hist.hist(bins=bins_dd, alpha=0.75,color='green',normed=True)
    dd_hist.plot(kind='kde',style='k--')

    '''
    Drawdown analysis-This code plots a histogram of a stocks drawdown length
    characteristics.  Ensure that stock ticker used in this function has already
    been placed in the ticker list. 
    '''

    stock_dd_length=calc_drawdown_local('WAT',63)
    dd_len_hist=DataFrame(stock_dd_length)
    bins_len_dd=np.linspace(0,30,31)
    dd_len_hist.hist(bins=bins_len_dd, alpha=0.55, color='purple',normed=True)
    plt.title('Drawdown lengths - WAT') 
    dd_len_hist.describe()
    
    