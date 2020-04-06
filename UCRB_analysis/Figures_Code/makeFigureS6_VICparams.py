import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt
from utils import fitParams

def makeFigureS6_VICparams():

    AnnualQ_h = np.loadtxt('../Qgen/AnnualQ.csv',delimiter=',',skiprows=1,usecols=[208])*1233.48 # convert to m^3
    
    # ignore first 41 years (1909-1949)
    AnnualQ_h = AnnualQ_h[41::]
    
    CMIPscenarios = np.loadtxt('../Qgen/CMIP/CMIP3_CMIP5_singletrace/CMIP_monthlyQ_m3.csv',delimiter=',',skiprows=1,usecols=[*range(1,99)])
    
    # compute annual flows under CMIP scenarios
    CMIP_annual = np.zeros([64,98])
    for i in range(64):
        CMIP_annual[i,:] = np.sum(CMIPscenarios[i*12:(i+1)*12,:],0)
        
    # log-space residuals divided by mean log-space prediction
    log_resid = np.log(AnnualQ_h) - np.log(CMIP_annual[:,0])
    log_resid_frac = log_resid/np.mean(np.log(CMIP_annual[:,0]))
    
    mu_resid = np.mean(log_resid_frac)
    std_resid = np.std(log_resid_frac)
    rho1_resid = np.corrcoef(log_resid_frac[1:],log_resid_frac[0:-1])[0,1]
    
    # 1) take log-space CMIP simulations
    # 2) take mean and variance of normal distribution fitted above
    # 3) multiply by mean log-space CMIP prediction
    # 4) generate normal AR1 noise from that distribution and add it to log-space CMIP simulations
    # 5) fit HMM 
    # 6) repeat 100 times and find mean parameter estimates
        
    # find mean parameter estimates over nsims of historical CMIP runs 
    # and compare parameter with true parameter estimates over historical data
    nsims = 100
    simParams = np.zeros([nsims,6])
    
    logFlows = np.log(CMIP_annual[:,0])
    noise = np.zeros(len(logFlows))
    mu_noise = mu_resid*np.mean(logFlows)
    std_noise = std_resid*np.mean(logFlows)
    for j in range(nsims): # add noise and fit HMM 100x
        noise[0] = ss.norm.rvs(mu_noise,std_noise)
        for k in range(len(logFlows)-1):
            noise[k+1] = mu_noise + rho1_resid*(noise[k] - mu_noise) + \
                            ss.norm.rvs(0,std_noise)*np.sqrt(1-rho1_resid**2)
        noisyLogFlows = np.exp((logFlows + noise))/1233.48 # convert to real-space and then acre-ft
        simParams[j,:] = fitParams(noisyLogFlows)
        
    baseParams = np.array([fitParams(AnnualQ_h/1233.48)])
    baseParams = pd.DataFrame({'mu0':baseParams[:,0],'sigma0':baseParams[:,1],\
                              'mu1':baseParams[:,2],'sigma1':baseParams[:,3],\
                              'p00':baseParams[:,4],'p11':baseParams[:,5]})
    baseParams['Ensemble'] = 'Historical'
    
    noNoiseParams = np.array([fitParams(CMIP_annual[:,0]/1233.48)])
    noNoiseParams = pd.DataFrame({'mu0':noNoiseParams[:,0],'sigma0':noNoiseParams[:,1],\
                              'mu1':noNoiseParams[:,2],'sigma1':noNoiseParams[:,3],\
                              'p00':noNoiseParams[:,4],'p11':noNoiseParams[:,5]})
    noNoiseParams['Ensemble'] = 'VIC Predictions'
    
    meanParams = np.array([np.mean(simParams,0)])
    meanParams = pd.DataFrame({'mu0':meanParams[:,0],'sigma0':meanParams[:,1],\
                              'mu1':meanParams[:,2],'sigma1':meanParams[:,3],\
                              'p00':meanParams[:,4],'p11':meanParams[:,5]})
    meanParams['Ensemble'] = 'Mean VIC Predictions + Noise'
        
    simParams = pd.DataFrame({'mu0':simParams[:,0],'sigma0':simParams[:,1],\
                              'mu1':simParams[:,2],'sigma1':simParams[:,3],\
                              'p00':simParams[:,4],'p11':simParams[:,5]})
    simParams['Ensemble'] = 'VIC Predictions + Noise'
    
    allSamples = pd.concat([simParams,meanParams,noNoiseParams,baseParams])
    
    sns.set_style("dark")
    
    colors=['#ff7f00','#377eb8','#4daf4a','#000000']
    sns.set_palette(sns.color_palette(colors))
    
    sns.pairplot(allSamples,hue='Ensemble',corner=True)
    plt.savefig('FigureS6_VICparams.pdf')
    plt.clf()
    
    return None