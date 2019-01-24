#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

####################################Read the Excel File########################################################################
df = pd.read_csv('518DataIn.csv')
pd.read_csv('518DataIn.csv', header=None)


#########################Compute the mean vector and the Covariance Matrix#####################################################

a = df.mean(axis = 0)
b = df.cov()
Covariance = np.matrix(np.array(b))
d1 = np.array(a)
#print(d1)
d3 = np.delete(d1, 0)
#print(d3)
#d2 = np.delete(d1, 0)
d2 = np.matrix(d3)
mu = d2.T

stockcorr = df.corr()
print(stockcorr)


number = 30

z = np.ones((number,1))
Voo = np.asmatrix(z) 
targetreturn = 0.1

#print(a)
#############################Plots of Histogram Returns of the stocks###################################################################### 

StockPrices = pd.read_csv('518DataIn.csv', parse_dates=['Date'])

StockPrices = StockPrices.sort_values(by='Date')

StockPrices.set_index('Date', inplace=True)


plt.hist((StockPrices["NFLX"]).dropna(), bins=80, density=False)
plt.show()

#covariance = Stockreturns.cov()
#Covariance = np.matrix('2 2; 2 4')
#########################################Mean Annulaised and Sigma annulaised###################################################
mean_return_daily = np.mean(StockPrices['NFLX'])
mean_return_annualized = ((1 + mean_return_daily)**252)-1
print('Annulized mean return is',mean_return_annualized)


sigma_daily = np.std(StockPrices['NFLX'])
sigma_annualized = sigma_daily*math.sqrt(252)
print('Annulized sigma is', sigma_annualized)






#############################Compute the Optimal weights using Markowitz method##################################################



A1 = np.dot(mu.T, np.dot(Covariance.I, Voo))
B1 = np.dot(mu.T, np.dot(Covariance.I, mu))
C1 = np.dot(Voo.T, np.dot(Covariance.I, Voo))
A = A1.item()
B = B1.item()
C = C1.item()
D = B*C - A**2
g = (B/D)*(np.dot(Covariance.I, Voo)) - (A/D)*np.dot(Covariance.I, mu)
h = (C/D)*(np.dot(Covariance.I, mu)) - (A/D)*np.dot(Covariance.I, Voo)
w = g + h*targetreturn
#print(Covariance)
#print(mu)
print(w.real)
# print(w[0].item())

print(abs(w.sum()))
# print((np.dot(w.T, mu)).real)
r = 0.03
Variance = np.dot(w.T, np.dot(Covariance.I, w))
#print('V', Variance)
portfolio_volatility = math.sqrt(Variance.real)
print('The Portfolio Volatility is: ', portfolio_volatility)
Variance_eff = ((C/D)*(targetreturn - (A/C))**2) + 1/C
#print(Variance_p)
risk_measure = (Variance_eff.real)**0.5
print('The risk measure (Volatility_effective) is: ', risk_measure.real)




##################################################Plots######################################################################


df['Result1'] =  df.MSFT*(-1.93338492) + df.AMZN*(4.27128905) + df.JNJ*(-7.00048559) + df.JPM*(15.20218959)   
#df
df['Result2'] = df.XOM*(-21.2996285) + df.BAC*(-11.64672372) + df.WMT*(-24.53403233) + df.WFC*(-1.9948698) 
df['Result3'] = df.V*(8.85791018) + df.PG*(-26.86255558) + df.BUD*(-0.2818632) + df['T']*(-5.39288205)
df['Result4'] = df.CVX*(5.87335946) + df.UNH*(19.10929214) + df.PFE*(-4.47432973) + df.RHHBY*(4.01798144)
df['Result5'] = df.CHL*(-14.13657931) + df.HD*(27.01877708) + df.TSM*(7.12156938) +df.VZ*(6.26905743) 
df['Result6'] = df.ORCL*(-14.36780326) + df.C*(-1.8280523) + df.NVDA*(7.31633299) + df.NVS*(-16.82941599) 
df['Result7'] = df.AAPL*(12.82040103) + df.INTC*(-7.15037373) + df.NFLX*(6.80972672) + df.SONC*(0.58827945)
df['Result8'] = df.MO*(44.3829126) + df.GOOGL*(-8.92609853)
df['MyResult1'] = df['Result1'] + df['Result2'] + df['Result3'] + df['Result4'] + df['Result5'] + df['Result6'] + df['Result7']
df['MyResult2'] = df['Result8']

df['MyResult'] = df['MyResult1'] + df['MyResult2']
df['EqualWeight'] = df.MSFT*(1/30) + df.AMZN*(1/30) + df.JNJ*(1/30) + df.JPM*(1/30) + df.XOM*(1/30) + df.BAC*(1/30) + df.WMT*(1/30) + df.WFC*(1/30) + df.V*(1/30) + df.PG*(1/30) + df.BUD*(1/30) + df['T']*(1/30) + df.CVX*(1/30) + df.UNH*(1/30) + df.PFE*(1/30) + df.RHHBY*(1/30) + df.CHL*(1/30) + df.HD*(1/30) + df.TSM*(1/30) +df.VZ*(1/30) + df.ORCL*(1/30) + df.C*(1/30) + df.NVDA*(1/30) + df.NVS*(1/30) + df.AAPL*(1/30) + df.INTC*(1/30) + df.NFLX*(1/30) + df.SONC*(1/30) + df.MO*(1/30) + df.GOOGL*(1/30)

#df.plot(x = 'Date', y = 'MyResult' - 'EqualWeight')
df['NetReturn'] =  df['MyResult'] - df['EqualWeight']             #df.MSFT*(1.83975008e-02) + df.AMZN*(5.65136002e+00) + df.JNJ*(3.42294630e-02) + df.JPM*(1.63945202e+01) + df.XOM*(-3.29493362e+01) + df.BAC*(-1.00283093e+01) + df.WMT*(-1.15477775e+01) + df.WFC*(-5.86615197e+00) + df.V*(1.09166774e+01) + df.PG*(-2.43660349e+01) + df.BUD*(-2.20023653e+00) + df['T']*(-1.47889237e+01) + df.CVX*(1.08963550e+01) + df.UNH*(2.08591943e+01) + df.PFE*(-5.51032091e+00) + df.RHHBY*(1.70675958e+00) + df.CHL*(-1.88955441e+01) + df.HD*(2.89857352e+01) + df.TSM*(9.20896610e+00) +df.VZ*(6.75952907e+00) + df.ORCL*(-1.29630979e+01) + df.C*(-2.08396881e+00) + df.NVDA*(7.86116739e+00) + df.NVS*(-1.20676655e+01) + df.AAPL*(1.34678182e+01) + df.INTC*(-6.27068119e+00) + df.NFLX*(6.39946459e+00) + df.SONC*(-1.10194082e+00) + df.MO*(3.18913482e+01) + df.GOOGL*(-9.41153289e+00)- (df.MSFT*(1/30) + df.AMZN*(1/30) + df.JNJ*(1/30) + df.JPM*(1/30) + df.XOM*(1/30) + df.BAC*(1/30) + df.WMT*(1/30) + df.WFC*(1/30) + df.V*(1/30) + df.PG*(1/30) + df.BUD*(1/30) + df['T']*(1/30) + df.CVX*(1/30) + df.UNH*(1/30) + df.PFE*(1/30) + df.RHHBY*(1/30) + df.CHL*(1/30) + df.HD*(1/30) + df.TSM*(1/30) +df.VZ*(1/30) + df.ORCL*(1/30) + df.C*(1/30) + df.NVDA*(1/30) + df.NVS*(1/30) + df.AAPL*(1/30) + df.INTC*(1/30) + df.NFLX*(1/30) + df.SONC*(1/30) + df.MO*(1/30) + df.GOOGL*(1/30))


df.plot(x = 'Date', y = 'MyResult')
df.plot(x = 'Date', y = 'EqualWeight')
df.plot(x = 'Date', y = 'NetReturn')


###############################################Results########################################################################
Port_Vola = np.dot(w.T, np.dot(Covariance.I, w))
NetReturn = np.sum(df['NetReturn'])
EqualWeight = np.sum(df['EqualWeight'])
MyResult = np.sum(df['MyResult'])
#df['Sharpe'] = (df["MyResult"]-r)*(1/Port_Vola)

print('NetReturn',NetReturn)
print('MyResult', MyResult)
print('EqualWeight', EqualWeight)

########################################Test for Normality####################################################################
############################Using Skew, Excess Kurtosis and Shapiro - Wilk Test###############################################

print('Skew for NFLX is: ', skew(StockPrices["NFLX"].dropna()))
print('Excess Kurtosis for NFLX is: ', kurtosis(StockPrices["NFLX"].dropna()))

p_value_NFLX = stats.shapiro(StockPrices["NFLX"].dropna())[1]
if p_value_NFLX <= 0.05:
    print("Reject Null hypothesis of normality for NFLX.")
else:
    print("Failed to reject Null hypothesis of normality for NFLX.")
    
    
print('Skew for MyResult is: ', skew(df["MyResult"].dropna()))
print('Excess Kurtosis for MyResult is: ', kurtosis(df["MyResult"].dropna()))

p_value_MyResult = stats.shapiro(df["MyResult"].dropna())[1]
if p_value_MyResult <= 0.05:
    print("Reject Null hypothesis of normality for MyResult.")
else:
    print("Failed to reject Null hypothesis of normality for MyResult.")    
    
###############################################################################################################################
#df['MyResult'] =  df.MFST*(1.83975008e-02) + df.AMZN*(5.65136002e+00) + df.JNJ*(3.42294630e-02) + df.JPM*(1.63945202e+01) + df.XOM*(-3.29493362e+01) + df.BAC*(-1.00283093e+01) + df.WMT(-1.15477775e+01) + df.WFC*(-5.86615197e+00) + df.V*(1.09166774e+01) + df.PG*(-2.43660349e+01) + df.BUD*(-2.20023653e+00) + df.T*(-1.47889237e+01) + df.CVX*(1.08963550e+01) + df.UNH*(2.08591943e+01) + df.PFE*(-5.51032091e+00) + df.RHHBY*(1.70675958e+00) + df.CHL*(-1.88955441e+01) + df.HD*(2.89857352e+01) + df.TSM*(9.20896610e+00) +df.VZ*(6.75952907e+00) + df.ORCL*(-1.29630979e+01) + df.C*(-2.08396881e+00) + df.NVDA*(7.86116739e+00) + df.NVS*(-1.20676655e+01) + df.AAPL*(1.34678182e+01) + df.INTC*(-6.27068119e+00) + df.NFLX*(6.39946459e+00) + df.SONC*(-1.10194082e+00) + df.MO*(3.18913482e+01) + df.GOOGL*(-9.41153289e+00) 


#df['EqualWeight'] = df.MFST*(1/30) + df.AMZN*(1/30) + df.JNJ*(1/30) + df.JPM*(1/30) + df.XOM*(1/30) + df.BAC*(1/30) + df.WMT(1/30) + df.WFC*(1/30) + df.V*(1/30) + df.PG*(1/30) + df.BUD*(1/30) + df.T*(1/30) + df.CVX*(1/30) + df.UNH*(1/30) + df.PFE*(1/30) + df.RHHBY*(1/30) + df.CHL*(1/30) + df.HD*(1/30) + df.TSM*(1/30) +df.VZ*(1/30) + df.ORCL*(1/30) + df.C*(1/30) + df.NVDA*(1/30) + df.NVS*(1/30) + df.AAPL*(1/30) + df.INTC*(1/30) + df.NFLX*(1/30) + df.SONC*(1/30) + df.MO*(1/30) + df.GOOGL*(1/30)
##############################################AR(1) Regression######################################################################
ar1 = np.array([1, 0.7])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1762)
print(simulated_data_1)
sd = [x for x in simulated_data_1] 
#plt.plot(simulated_data_1)
df['AutoReg1'] =  pd.Series(sd, index=df.index)
df.columns
result = sm.ols(formula = "AutoReg1 ~ AMZN + JNJ + JPM + XOM + BAC + WMT + WFC + V + PG + BUD + T + CVX + UNH + PFE + RHHBY + CHL + HD + TSM + VZ + ORCL + C + NVDA + NVS + AAPL + NFLX + INTC + SONC + MO + GOOGL", data=df).fit()
print(result.params)
print(result.summary())
#df

# 'MSFT', 'AMZN', 'JNJ', 'JPM', 'XOM', 'BAC', 'WMT', 'WFC', 'V',
#        'PG', 'BUD', 'T', 'CVX', 'UNH', 'PFE', 'RHHBY', 'CHL', 'HD', 'TSM',
#        'VZ', 'ORCL', 'C', 'NVDA', 'NVS', 'AAPL', 'INTC', 'NFLX', 'SONC', 'MO',
#        'GOOGL', 'Result1', 'Result2', 'Result3', 'Result4', 'Result5',
#        'Result6', 'Result7', 'Result8', 'MyResult1', 'MyResult2', 'MyResult',
#        'EqualWeight', 'NetReturn', 'AR(1)']

