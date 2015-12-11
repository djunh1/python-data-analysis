# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:37:08 2015
t
@author: djunh
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas.io.data as web
import pprint
import statsmodels.tsa.stattools as ts

from numpy import cumsum, log, polyfit, sqrt, std, subtract 
from numpy.random import randn
from pandas.io.data import DataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from pandas.stats.api import ols

def plot_scatter(df, ts1, ts2):
    '''
    Creates scatter plot with two time series.  
    '''
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    T=np.arctan2(df[ts1], df[ts2])
    
    plt.scatter(df[ts1], df[ts2],s=25,c=T,alpha=0.7)
    
    #Best Fit line
    x=df[ts1]
    y=df[ts2]
    m,b=np.polyfit(x,y,1)
    plt.plot(x, m*x + b, '-')
    
    plt.show()

def plot_price(df,ts1,ts2):
    '''
    time series plotter
    '''
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2014, 9, 1), datetime.datetime(2015, 12, 9))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()
    
def plot_residuals(df):
    '''
    Plots residuals
    '''
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2014, 9, 1), datetime.datetime(2015, 12, 9))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()
    
def hurst(ts):
    '''
    Returns the hurst exponent of a time series. 
    '''
    lags=range(2,100)
    tau=[sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    
    return poly[0]*2.0
    
    
def create_lagged_series(symbol1, symbol2, start_date, end_date, lags=2):
    """
    This creates a pandas DataFrame that stores the 
    percentage returns of the adjusted closing value of 
    a stock obtained from Yahoo Finance, along with a 
    number of lagged returns from the prior trading days 
    (lags defaults to 5 days). Trading volume, as well as 
    the Direction from the previous day, are also included.
    """

    # Obtain stock information from Yahoo Finance
    ts1 = DataReader(
    	symbol1, "yahoo", 
    	start_date-datetime.timedelta(days=365), 
    	end_date
    )
    
    ts2= DataReader(
    	symbol2, "yahoo", 
    	start_date-datetime.timedelta(days=365), 
    	end_date
    )

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts1.index)
    tslag["Today"] = ts1["Adj Close"]
    tslag["Volume"] = ts1["Volume"]

    # Create the shifted lag series of prior trading period close values
    tslag["Lag%s" % str(1)] = ts1["Adj Close"].shift(1)
    tslag["Lag%s" % str(2)] = ts2["Adj Close"].shift(2)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = \
        tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret   
    
def create_drawdowns(pnl):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the 
    pnl_returns is a pandas Series.

    Parameters:
    pnl - A pandas Series representing period percentage returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Looking at high water mark
    hwm = [0]

    # Create the drawdown and duration series
    idx = pnl.index
    drawdown = pd.Series(index = idx)
    duration = pd.Series(index = idx)

    # Loop over the index range
    for t in range(1, len(idx)):
        hwm.append(max(hwm[t-1], pnl[t]))
        drawdown[t]= (hwm[t]-pnl[t])
        duration[t]= (0 if drawdown[t] == 0 else duration[t-1]+1)
        
    return drawdown, drawdown.max(), duration.max()
    
if __name__ == "__main__":
    start = datetime.datetime(2014, 9, 1)
    end = datetime.datetime(2015, 12, 1)

    arex = web.DataReader("NOV", "yahoo", start, end)
    wll = web.DataReader("XOM", "yahoo", start, end)

    df = pd.DataFrame(index=arex.index)
    df["NOV"] = arex["Adj Close"]
    df["XOM"] = wll["Adj Close"]

    # Plot a scatter plot and a time series
    plot_price(df, "NOV", "XOM")
    plot_scatter(df, "NOV", "XOM")

    # Calculate Beta
    res = ols(y=df['XOM'], x=df["NOV"])
    beta_hr = res.beta.x

    # Calculate the residuals of the linear combination
    df["res"] = df["XOM"] - beta_hr*df["NOV"]

    # Plot the residuals
    plot_residuals(df)

    # Calculate and output the CADF test on the residuals
    #====================================================
    cadf = ts.adfuller(df["res"])
    pprint.pprint(cadf)
    
    #Output Hurst exponent
    #=====================
    print("Hurst of (insertstockhere):  %s" %hurst(df["res"]))
    
    #Machine learning
    # Create a lagged series of the S&P500 US stock market index
    #===========================================================
    snpret = create_lagged_series(
    	"TGA","NOV", datetime.datetime(2010,1,10), 
    	datetime.datetime(2015,12,10), lags=2
    )

    # Use the prior two days of returns as predictor 
    # values, with direction as the response
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,1,1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]
   
    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()), 
              ("LDA", LDA()), 
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
              	C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
              )]

    # Iterate through the models
    for m in models:
        
        # Each model is trained
        m[1].fit(X_train, y_train)

        #Use test set to get array of predictions
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(pred, y_test))