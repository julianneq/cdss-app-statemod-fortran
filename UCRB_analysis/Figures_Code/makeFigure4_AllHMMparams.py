import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import setupProblem, getSamples

def makeFigure4_AllHMMparams():
    
    baseSOWparams = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    Historical = pd.DataFrame({'mu0':baseSOWparams[:,0],'sigma0':baseSOWparams[:,1],\
                           'mu1':baseSOWparams[:,2],'sigma1':baseSOWparams[:,3],\
                           'p00':baseSOWparams[:,4],'p11':baseSOWparams[:,5]})
    Historical['Ensemble'] = 'Historical'
    
    # load CMIP 3 and CMIP 5 flow data at last node
    CMIP = loadData('CMIPunscaled_SOWs')
    Paleo = loadData('Paleo_SOWs')
    Original_1000 = loadData('LHsamples_original_1000_AnnQonly')
    Wider_1000 = loadData('LHsamples_wider_1000_AnnQonly')
    
    # merge all samples into a dataframe
    CMIP['Ensemble'] = 'CMIP'
    Paleo['Ensemble'] = 'Paleo'
    Original_1000['Ensemble'] = 'Box Around Historical'
    Wider_1000['Ensemble'] = 'All Encompassing'
    
    allSamples = pd.concat([Wider_1000,Original_1000,Paleo,CMIP,Historical])
    col = allSamples.columns.tolist()
    
    sns.set_style("dark")
    
    colors=['#bebada','#fb8072','#b3de69','#ffffb3','#80b1d3']
    sns.set_palette(sns.color_palette(colors))
    
    fig = sns.pairplot(allSamples,hue='Ensemble',corner=True)
    for j in range(len(baseSOWparams[0])-1):
        for i in range(0,j+1):
            fig.axes[j+1,i].set_xlim((np.min(allSamples[col[i]]),np.max(allSamples[col[i]])))
            fig.axes[j+1,i].set_ylim((np.min(allSamples[col[j+1]]),np.max(allSamples[col[j+1]])))
    
    plt.savefig('Figure4_AllHMMparams.pdf')
    plt.clf()
    
    return None

def loadData(design):
    param_bounds, param_names, params_no, problem = setupProblem(design)
    samples, rows_to_keep = getSamples(design, params_no, param_bounds)
    df = pd.DataFrame({'mu0':samples[:,0],'sigma0':samples[:,1],\
                       'mu1':samples[:,2],'sigma1':samples[:,3],\
                       'p00':samples[:,4],'p11':samples[:,5]})

    return df