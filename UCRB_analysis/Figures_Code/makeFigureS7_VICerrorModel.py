import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss
import pandas as pd
import seaborn as sns

def makeFigureS7_VICerrorModel():

    AnnualQ_h = np.loadtxt('../Qgen/AnnualQ.csv',delimiter=',',skiprows=1,usecols=[208])*1233.48 # convert to m^3
    
    # ignore first 41 years (1909-1949)
    AnnualQ_h = AnnualQ_h[41::]
    
    CMIPscenarios = np.loadtxt('../Qgen/CMIP/CMIP3_CMIP5_singletrace/CMIP_monthlyQ_m3.csv',delimiter=',',skiprows=1,usecols=[*range(1,99)])
    
    # compute annual flows under CMIP scenarios
    CMIP_annual = np.zeros([64,98])
    for i in range(64):
        CMIP_annual[i,:] = np.sum(CMIPscenarios[i*12:(i+1)*12,:],0)
        
    sns.set()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(4,2,5)
    ax4 = fig.add_subplot(4,2,7)
    ax5 = fig.add_subplot(2,2,4)
    
    # plot CMIP historical vs. true historical
    l1,= ax1.plot(range(1950,2014),np.log(AnnualQ_h))
    l2, = ax1.plot(range(1950,2014),np.log(CMIP_annual[:,0]))
    ax1.legend([l1,l2],['Observed Flow at State Line','VIC-simulated Flow at State Line'],loc='lower left',fontsize=14)
    ax1.set_xlabel('Year',fontsize=16)
    ax1.set_ylabel('log(Annual Flow in m' + r'$^3$' + ')',fontsize=16)
    ax1.set_ylim([21.25,23.5])
    ax1.tick_params(axis='both',labelsize=14)
    
    # log-space residuals divided by mean log-space prediction
    log_resid = np.log(AnnualQ_h) - np.log(CMIP_annual[:,0])
    log_resid_frac = log_resid/np.mean(np.log(CMIP_annual[:,0]))
    
    # residuals vs. fitted
    ax2.scatter(np.log(CMIP_annual[:,0]),log_resid_frac)
    ax2.plot([0.99*np.min(np.log(CMIP_annual[:,0])),1.01*np.max(np.log(CMIP_annual[:,0]))],[0,0],c='k')
    ax2.set_xlim([0.99*np.min(np.log(CMIP_annual[:,0])),1.01*np.max(np.log(CMIP_annual[:,0]))])
    ax2.set_xlabel('Log-space Predictions',fontsize=16)
    ax2.set_ylabel('Log-space Residuals /\nLog-space Predictions', fontsize=16)
    ax2.tick_params(axis='both',labelsize=14)
    
    # acf of residuals
    sm.graphics.tsa.plot_acf(log_resid_frac,ax=ax3)
    ax3.set_xlim([-0.5,8.5])
    ax3.set_title('')
    ax3.set_ylabel('Autocorrelation',fontsize=16)
    ax3.tick_params(axis='both',labelsize=14)
    
    # pacf of residuals
    sm.graphics.tsa.plot_pacf(log_resid_frac,ax=ax4)
    ax4.set_xlim([-0.5,8.5])
    ax4.set_title('')
    ax4.set_xlabel('Lag',fontsize=16)
    ax4.set_ylabel('Partial\nAutocorrelation',fontsize=16)
    ax4.tick_params(axis='both',labelsize=14)
    
    # normal QQ of log-space residuals divded by mean log-space prediction
    x_sorted = np.sort(log_resid_frac)
    p_observed = np.arange(1,len(log_resid_frac)+1,1)/(len(log_resid_frac)+1)
    x_fitted = ss.norm.ppf(p_observed, np.mean(x_sorted), np.std(x_sorted,ddof=1))
    
    ax5.scatter(x_sorted,x_fitted,color='b')
    ax5.plot([1.02*np.min(x_sorted),3*np.max(x_sorted)],[1.02*np.min(x_sorted),3*np.max(x_sorted)],color='r')
    ax5.set_xlim([1.02*np.min(x_sorted),3*np.max(x_sorted)])
    ax5.set_ylim([1.02*np.min(x_sorted),3*np.max(x_sorted)])
    ax5.set_xlabel('Log-space Residuals /\nLog-space Fitted Values',fontsize=16)
    ax5.set_ylabel('Theoretical Normal Quantiles',fontsize=16)
    ax5.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(hspace=0.3,wspace=0.25)
    fig.set_size_inches([15.6,9.6])
    fig.savefig('FigureS7_VICerrorModel.pdf')
    fig.clf()
    
    return None