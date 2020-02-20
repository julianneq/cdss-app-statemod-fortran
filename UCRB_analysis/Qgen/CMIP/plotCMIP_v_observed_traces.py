import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import seaborn as sns

AnnualQ_h = np.loadtxt('../AnnualQ.csv',delimiter=',',skiprows=1,usecols=[208])*1233.48 # convert to m^3
MonthlyQ_h = np.loadtxt('../MonthlyQ.csv',delimiter=',',skiprows=1,usecols=[208])*1233.48 # convert to m^3

# ignore first 41 years (1909-1949)
AnnualQ_h = AnnualQ_h[41::]
MonthlyQ_h = MonthlyQ_h[(41*12)::]

CMIPscenarios = np.loadtxt('CMIP3_CMIP5_singletrace/CMIP_monthlyQ_m3.csv',delimiter=',',skiprows=1,usecols=[*range(1,99)])

# compute annual flows under CMIP scenarios
CMIP_annual = np.zeros([64,98])
for i in range(64):
    CMIP_annual[i,:] = np.sum(CMIPscenarios[i*12:(i+1)*12,:],0)

# plot CMIP historical vs. true historical (monthly)
l1, = plt.semilogy(MonthlyQ_h)
l2, = plt.semilogy(CMIPscenarios[:,0])
plt.legend([l1,l2],['Observations','VIC'])
plt.savefig('VIC_v_Observations_Monthly.png')
plt.clf()

# plot CMIP historical vs. true historical (annual)
l1,= plt.semilogy(range(1950,2014),AnnualQ_h)
l2, = plt.semilogy(range(1950,2014),CMIP_annual[:,0])
plt.legend([l1,l2],['Observations','VIC'])
plt.savefig('VIC_v_Observations_Annual.png')
plt.clf()

# compute and plot log-space residuals (annual)
log_resid = np.log(AnnualQ_h) - np.log(CMIP_annual[:,0])

# log-space residuals divided by mean log-space prediction
log_resid_frac = log_resid/np.mean(np.log(CMIP_annual[:,0]))

# residuals vs. fitted
plt.scatter(np.log(CMIP_annual[:,0]),log_resid_frac)
plt.plot([np.min(np.log(CMIP_annual[:,0])),np.max(np.log(CMIP_annual[:,0]))],[0,0],c='r')
plt.xlabel('Log-space Prediction')
plt.ylabel('Log-space Residuals / Log-space Prediction')
plt.savefig('LogResidualFrac_v_LogFitted.png')
plt.clf()
# fairly homoscedastic but biased

# acf of residuals
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
sm.graphics.tsa.plot_acf(log_resid_frac,ax=ax1)
sm.graphics.tsa.plot_pacf(log_resid_frac,ax=ax2)
ax2.set_xlim([0,10])
ax2.set_ylim([-1,1])
plt.savefig('ACF_PACF_LogResidualFrac.png')
plt.clf()

# normal QQ of log-space residuals divded by mean log-space prediction
sm.qqplot(log_resid_frac,ss.norm,fit=True,line='45')
plt.savefig('LogResidualFrac_QQ.png')
plt.clf()

def norm_MC(Nyears,theoretical,dataCorr):
    rhoVector = np.zeros(10000)
    for i in range(10000):
        simulated = ss.norm.rvs(0,1, size=Nyears)
        rhoVector[i] = np.corrcoef(np.sort(simulated), theoretical)[0,1]
            
    count = 0
    for i in range(len(rhoVector)):
        if dataCorr < rhoVector[i]:
            count = count + 1
            
    p_value = 1 - count/10000
 
    return p_value

# test if log-space residuals divided by log-space predicted mean are normally distributed
Nyears = len(log_resid_frac)
m = np.arange(Nyears)+1
p = (m-0.5)/Nyears
norm = ss.norm.ppf(p,0,1)
normRho = np.corrcoef((np.sort(log_resid_frac)-np.mean(log_resid_frac))/np.std(log_resid_frac),norm)[0,1]
normSigLevel = norm_MC(Nyears, norm, normRho)
print(normSigLevel)

mu_resid = np.mean(log_resid_frac)
std_resid = np.std(log_resid_frac)
rho1_resid = np.corrcoef(log_resid_frac[1:],log_resid_frac[0:-1])[0,1]

# 1) take log-space CMIP simulations
# 2) take mean and variance of normal distribution fitted above
# 3) multiply by mean log-space CMIP prediction
# 4) generate normal AR1 noise from that distribution and add it to log-space CMIP simulations
# 5) fit HMM 
# 6) repeat 100 times and find mean parameter estimates
def fitHMM(TransformedQ):
    
    # fit HMM to all of data
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(TransformedQ,[len(TransformedQ),1]))
    hidden_states = model.predict(np.reshape(TransformedQ,[len(TransformedQ),1]))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
    
    logProb = model.score(np.reshape(TransformedQ,[len(TransformedQ),1]))
    
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
    
    return hidden_states, mus, sigmas, P, logProb

def fitParams(flows):
    # create matrices to store the parameters
    # each row is a different simulation
    # columns are mu0, sigma0, mu1, sigma1, p00, p11
    params = np.zeros([6])
    hidden_states, mus, sigmas, P, logProb = fitHMM(np.log(flows))
    
    params[0] = mus[0]
    params[1] = sigmas[0]
    params[2] = mus[1]
    params[3] = sigmas[1]
    params[4] = P[0,0]
    params[5] = P[1,1]
        
    return params

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

meanParams = np.array([np.mean(simParams,0)])
meanParams = pd.DataFrame({'mu0':meanParams[:,0],'sigma0':meanParams[:,1],\
                          'mu1':meanParams[:,2],'sigma1':meanParams[:,3],\
                          'p00':meanParams[:,4],'p11':meanParams[:,5]})
meanParams['Ensemble'] = 'MeanVIC'
    
simParams = pd.DataFrame({'mu0':simParams[:,0],'sigma0':simParams[:,1],\
                          'mu1':simParams[:,2],'sigma1':simParams[:,3],\
                          'p00':simParams[:,4],'p11':simParams[:,5]})
simParams['Ensemble'] = 'VICsims'

allSamples = pd.concat([simParams,meanParams,baseParams])

sns.set_style("dark")

colors=['#ff7f00','#377eb8','#000000']
sns.set_palette(sns.color_palette(colors))

sns.pairplot(allSamples,hue='Ensemble')
plt.savefig('CMIPmean_v_True.png')
plt.clf()


# repeat over all CMIP scenarios
simParams = np.zeros([nsims,97,6])

for i in range(97): # number of CMIP scenarios
    logFlows = np.log(CMIP_annual[:,i+1])
    noise = np.zeros(len(logFlows))
    mu_noise = mu_resid*np.mean(logFlows)
    std_noise = std_resid*np.mean(logFlows)
    for j in range(nsims): # add noise and fit HMM 100x
        noise[0] = ss.norm.rvs(mu_noise,std_noise)
        for k in range(len(logFlows)-1):
            noise[k+1] = mu_noise + rho1_resid*(noise[k] - mu_noise) + \
                            ss.norm.rvs(0,std_noise)*np.sqrt(1-rho1_resid**2)
        noisyLogFlows = np.exp((logFlows + noise))/1233.48 # convert to real-space and then acre-ft
        simParams[j,i,:] = fitParams(noisyLogFlows)
        
meanParams = np.mean(simParams,0)

meanParams = pd.DataFrame({'mu0':meanParams[:,0],'sigma0':meanParams[:,1],\
                          'mu1':meanParams[:,2],'sigma1':meanParams[:,3],\
                          'p00':meanParams[:,4],'p11':meanParams[:,5]})
meanParams.to_csv('MeanCMIPparams.txt')
