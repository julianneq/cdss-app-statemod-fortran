import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt
from utils import fitParams

def makeFigureS8_ReconstructionParams():

    # load paleo data at Cisco
    Paleo = pd.read_csv('../Qgen/Reconstruction/Cisco_Recon_v_Observed_v_Stateline.csv')
    
    # re-scale Cisco data to estimate data at CO-UT state line
    factor = np.nanmean(Paleo['ObservedNaturalStateline']/Paleo['ObservedNaturalCisco'])
    Paleo['ScaledNaturalCisco'] = Paleo['ObservedNaturalCisco']*factor
    Paleo['ScaledReconCisco'] = Paleo['ReconCisco']*factor
    
    # compute residual between observed stateline flow and scaled reconstructed flow
    Paleo['ScalingResid'] = Paleo['ObservedNaturalStateline'] - Paleo['ScaledReconCisco']
    Paleo['FractionScalingResid'] = Paleo['ScalingResid']/Paleo['ScaledReconCisco']
    
    # HMM parameters over the historical record    
    trueParams = np.array([[15.258112, 0.259061, 15.661007, 0.252174, 0.679107, 0.649169]])
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
        simParams[i,:] = fitParams(np.array(flows))
    
    # make scatter plot of fitted params to observations (trueParams) compared to
    # fitted params to scaled reconstructed flows + residuals (simParams) and their average
    meanParams = np.array([np.mean(simParams,0)])
    meanParams = pd.DataFrame({'mu0':meanParams[:,0],'sigma0':meanParams[:,1],\
                              'mu1':meanParams[:,2],'sigma1':meanParams[:,3],\
                              'p00':meanParams[:,4],'p11':meanParams[:,5]})
    meanParams['Ensemble'] = 'Mean Reconstruction + Noise'

    simParams = pd.DataFrame({'mu0':simParams[:,0],'sigma0':simParams[:,1],\
                              'mu1':simParams[:,2],'sigma1':simParams[:,3],\
                              'p00':simParams[:,4],'p11':simParams[:,5]})
    simParams['Ensemble'] = 'Reconstruction + Noise'

    noNoiseParams = np.array([fitParams(np.array(Paleo['ScaledReconCisco'][340:429]))])
    noNoiseParams = pd.DataFrame({'mu0':noNoiseParams[:,0],'sigma0':noNoiseParams[:,1],\
                              'mu1':noNoiseParams[:,2],'sigma1':noNoiseParams[:,3],\
                              'p00':noNoiseParams[:,4],'p11':noNoiseParams[:,5]})
    noNoiseParams['Ensemble'] = 'Reconstruction'

    allParams = pd.concat([simParams,meanParams,noNoiseParams,trueParams])
    
    
    sns.set_style("dark")
    
    colors=['#ff7f00','#377eb8','#4daf4a','#000000']
    sns.set_palette(sns.color_palette(colors))
    col = allParams.columns.tolist()
    
    fig = sns.pairplot(allParams,hue='Ensemble',corner=True)            
    plt.savefig('FigureS8_ReconstructionParams.pdf')
    plt.clf()
    
    return None