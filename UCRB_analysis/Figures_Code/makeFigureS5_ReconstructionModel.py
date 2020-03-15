import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt

def makeFigureS5_ReconstructionModel():

    sns.set()
    
    # load paleo data at Cisco
    Paleo = pd.read_csv('../Qgen/Reconstruction/Cisco_Recon_v_Observed_v_Stateline.csv')
    
    # convert acre-ft to m^3
    Paleo['ReconCisco'] = Paleo['ReconCisco']*1233.48
    Paleo['ObservedNaturalCisco'] = Paleo['ObservedNaturalCisco']*1233.48
    Paleo['ObservedNaturalStateline'] = Paleo['ObservedNaturalStateline']*1233.48
    
    # re-scale Cisco data to estimate data at CO-UT state line
    factor = np.nanmean(Paleo['ObservedNaturalStateline']/Paleo['ObservedNaturalCisco'])
    Paleo['ScaledNaturalCisco'] = Paleo['ObservedNaturalCisco']*factor
    Paleo['ScaledReconCisco'] = Paleo['ReconCisco']*factor
    
    # compute residual between observed stateline flow and scaled reconstructed flow
    Paleo['ScalingResid'] = Paleo['ObservedNaturalStateline'] - Paleo['ScaledReconCisco']
    Paleo['FractionScalingResid'] = Paleo['ScalingResid']/Paleo['ScaledReconCisco']
    
    
    fig = plt.figure()
    
    # compare scaled reconstruction at Cisco vs. Observed at state line
    ax = fig.add_subplot(2,2,1)
    l1, = ax.plot(Paleo['Year'][340:429],Paleo['ObservedNaturalStateline'][340:429])
    l2, = ax.plot(Paleo['Year'][340:429],Paleo['ScaledReconCisco'][340:429])
    ax.legend([l1,l2],['Observed Flow at State Line','Scaled Reconstructed Flow at Cisco'],loc='lower left',fontsize=14)
    ax.set_xlabel('Year',fontsize=16)
    ax.set_ylabel('Annual Flow (m' + r'$^3$' + ')',fontsize=16)
    ax.set_ylim([-0.15E10,1.4E10])
    ax.tick_params(axis='both',labelsize=14)
    
    # plot residuals vs. fitted
    ax = fig.add_subplot(2,2,2)
    ax.scatter(Paleo['ScaledReconCisco'][340:429],Paleo['FractionScalingResid'][340:429])
    ax.plot([0.95*np.min(Paleo['ScaledReconCisco'][340:429]),1.02*np.max(Paleo['ScaledReconCisco'][340:429])],[0,0],c='k')
    ax.set_xlim([0.95*np.min(Paleo['ScaledReconCisco'][340:429]),1.02*np.max(Paleo['ScaledReconCisco'][340:429])])
    ax.set_xlabel('Fitted Flow at Stateline',fontsize=16)
    ax.set_ylabel('Residual / Fitted Value',fontsize=16)
    ax.set_ylim([-0.4,0.4])
    ax.tick_params(axis='both',labelsize=14)
    
    # plot acf and pacf of residuals
    ax = fig.add_subplot(2,2,3)
    sm.graphics.tsa.plot_acf(Paleo['FractionScalingResid'][340:429],ax=ax)
    ax.set_title('')
    ax.set_ylabel('Autocorrelation of\nResiduals / Fitted Values',fontsize=16)
    ax.set_xlabel('Lag',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlim([-2,21])
    
    # make normal QQ plot of residual fractions
    x_sorted = np.sort(Paleo['FractionScalingResid'][340:429])
    p_observed = np.arange(1,len(Paleo['FractionScalingResid'][340:429])+1,1)/(len(Paleo['FractionScalingResid'][340:429])+1)
    x_fitted = ss.norm.ppf(p_observed, np.mean(x_sorted), np.std(x_sorted,ddof=1))
    
    ax = fig.add_subplot(2,2,4)
    ax.scatter(x_sorted,x_fitted,color='b')
    ax.plot([1.05*np.min(x_sorted),1.05*np.max(x_sorted)],[1.05*np.min(x_sorted),1.05*np.max(x_sorted)],color='r')
    ax.set_xlim([1.05*np.min(x_sorted),1.05*np.max(x_sorted)])
    ax.set_xlabel('Residuals / Fitted Values',fontsize=16)
    ax.set_ylabel('Theoretical Normal Quantiles',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    
    fig.set_size_inches([15.6,9.6])
    fig.savefig('FigureS5_ReconstructionModel.pdf')
    fig.clf()
    
    return None

