import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from fitHMM import fitHMM, makePPandQQplot

plt.switch_backend('agg')

# load paleo data at Cisco
Paleo = pd.read_csv('Cisco_Recon_v_Observed_v_Stateline.csv')

# re-scale Cisco data to estimate data at CO-UT state line
factor = np.nanmean(Paleo['ObservedNaturalStateline']/Paleo['ObservedNaturalCisco'])
Paleo['ScaledNaturalCisco'] = Paleo['ObservedNaturalCisco']*factor
Paleo['ScaledReconCisco'] = Paleo['ReconCisco']*factor

# compute residual between observed stateline flow and scaled reconstructed flow
Paleo['ScalingResid'] = Paleo['ObservedNaturalStateline'] - Paleo['ScaledReconCisco']
Paleo['FractionScalingResid'] = Paleo['ScalingResid']/Paleo['ScaledReconCisco']

# compare scaled reconstruction at Cisco vs. Observed at state line
l1, = plt.plot(Paleo['Year'][340:429],Paleo['ObservedNaturalStateline'][340:429])
l2, = plt.plot(Paleo['Year'][340:429],Paleo['ScaledReconCisco'][340:429])
plt.legend([l1,l2],['Observed State line','Scaled Reconstructed Cisco'],loc='upper left')
plt.savefig('Recon_v_Observed_Stateline.png')
plt.clf()

# perform diagnosis of residuals (from https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034)
# residuals vs. fitted values
def diagnostics(fittedvalues, residuals, data, predictand):
    sns.residplot(fittedvalues, predictand, data=data, lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    # qq plot of residuals vs. normal fit
    sm.qqplot(residuals,ss.norm,fit=True,line='45')
    
    # autocorrelation of residuals
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    sm.graphics.tsa.plot_acf(residuals,ax=ax)
    ax.set_xlim([0,10])
    
    ax = fig.add_subplot(2,1,2)
    sm.graphics.tsa.plot_pacf(residuals,ax=ax)
    ax.set_xlim([0,10])
    
    return None

# scaling of reconstructed Cisco flows to predict CO-UT stateline flows
diagnostics(Paleo['ScaledReconCisco'][340:429],Paleo['ScalingResid'][340:429],Paleo.iloc[340:429],'ObservedNaturalStateline')

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
Nyears = len(Paleo['ScalingResid'][340:429])
m = np.arange(Nyears)+1
p = (m-0.5)/Nyears
norm = ss.norm.ppf(p,0,1)
normRho = np.corrcoef((np.sort(Paleo['ScalingResid'][340:429])-np.mean(Paleo['ScalingResid'][340:429]))/np.std(Paleo['ScalingResid'][340:429]),norm)[0,1]
normSigLevel = norm_MC(Nyears, norm, normRho)
print(normSigLevel)

# fit HMM to true observations of flows at state line over historical record
def fitParams(flows):
    # create matrices to store the parameters
    # each row is a different simulation
    # columns are mu0, sigma0, mu1, sigma1, p00, p11
    params = np.zeros([6])
    hidden_states, mus, sigmas, P, logProb = fitHMM(np.log(flows))
    #makePPandQQplot(np.log(flows), mus, sigmas, P, 'No Noise Recon Fit ', 'ReconFits/NoNoiseFit')
    
    params[0] = mus[0]
    params[1] = sigmas[0]
    params[2] = mus[1]
    params[3] = sigmas[1]
    params[4] = P[0,0]
    params[5] = P[1,1]
        
    return params

trueParams = np.array([[15.25811235, 0.2590614, 15.66100725, 0.25217403, 0.67910715, 0.64916877]])
trueParams = pd.DataFrame({'mu0':trueParams[:,0],'sigma0':trueParams[:,1],\
                           'mu1':trueParams[:,2],'sigma1':trueParams[:,3],\
                           'p00':trueParams[:,4],'p11':trueParams[:,5]})
trueParams['Ensemble'] = 'Observations'

# fit HMM to scaled, reconstructed Cisco flows to represent estimate of flow at state line
# after first adding random errors from residuals of scaled, reconstructed Cisco flows vs. CO-UT state line flows
# repeat 100 times and see how estimates compare with those over the historical record itself
nsims = 100
simParams = np.zeros([nsims,6])
stdev = np.std(Paleo['FractionScalingResid'][340:429])
for i in range(nsims):
    flows = Paleo['ScaledReconCisco'][340:429] + Paleo['ScaledReconCisco'][340:429]*ss.norm.rvs(0,stdev,429-340)
    simParams[i,:] = fitParams(np.array(flows), i+1)

# make scatter plot of fitted params to observations (trueParams) compared to
# fitted params to scaled reconstructed flows + residuals (simParams) and their average
means = np.mean(simParams,0)
simParams = pd.DataFrame({'mu0':simParams[:,0],'sigma0':simParams[:,1],\
                          'mu1':simParams[:,2],'sigma1':simParams[:,3],\
                          'p00':simParams[:,4],'p11':simParams[:,5]})
simParams['Ensemble'] = 'Recon+Noise'

noNoiseParams = fitParams(np.array(Paleo['ScaledReconCisco'][340:429]))

simParams.loc[nsims] = [means[0], means[2], means[4], means[5], means[1], means[3], 'MeanRecon+Noise']
simParams.loc[nsims+1] = [noNoiseParams[0], noNoiseParams[2], noNoiseParams[4], \
              noNoiseParams[5], noNoiseParams[1], noNoiseParams[3], 'Recon']
simParams.loc[nsims] = list(means) + ['MeanRecon+Noise']
simParams.loc[nsims+1] = list(noNoiseParams) + ['MeanRecon']
simParams.to_csv('SimulatedPaleoHistParams.txt')

allParams = pd.concat([simParams,trueParams])

sns.set(font_scale=1.2)
sns.pairplot(allParams,hue='Ensemble',plot_kws={"s": 50})
plt.savefig('Paleo_scatter.png')
plt.clf()

# repeat over 366 64-yr moving windows of whole paleo-record and track mean parameter estimates over time
nsims = 100
stdev = np.std(Paleo['FractionScalingResid'][340:429])
simParams = np.zeros([nsims,366,6])
for i in range(nsims):
    flows = Paleo['ScaledReconCisco'][0:429] + Paleo['ScaledReconCisco'][0:429]*ss.norm.rvs(0,stdev,429)
    for j in range(366):
        simParams[i,j,:] = fitParams(np.array(flows[j:(j+64)]))

# find mean over nsims
meanParams = np.mean(simParams,0)

# make scatter plot of 366 mean parameter estimates, CMIP parameter estimates, and historical parameter estimates
meanParams = pd.DataFrame({'mu0':meanParams[:,0],'sigma0':meanParams[:,1],\
                          'mu1':meanParams[:,2],'sigma1':meanParams[:,3],\
                          'p00':meanParams[:,4],'p11':meanParams[:,5]})
meanParams.to_csv('MeanPaleoParams.txt')