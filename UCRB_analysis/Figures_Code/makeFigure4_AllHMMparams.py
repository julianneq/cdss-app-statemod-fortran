import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

baseSOWparams = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
Historical = pd.DataFrame({'mu0':baseSOWparams[:,0],'sigma0':baseSOWparams[:,1],\
                           'mu1':baseSOWparams[:,2],'sigma1':baseSOWparams[:,3],\
                           'p00':baseSOWparams[:,4],'p11':baseSOWparams[:,5]})
Historical['Ensemble'] = 'Historical'

def loadData(design):
    df = np.loadtxt(design, usecols=[7,8,9,10,11,12])
    df = pd.DataFrame({'mu0':df[:,0],'sigma0':df[:,1],\
                       'mu1':df[:,2],'sigma1':df[:,3],\
                       'p00':df[:,4],'p11':df[:,5]})

    return df

# load CMIP 3 and CMIP 5 flow data at last node
CMIP = loadData('../Qgen/CMIPunscaled_SOWs.txt')
Paleo = loadData('../Qgen/Paleo_SOWs.txt')
Original_1000 = loadData('../Qgen/LHsamples_original_1000_AnnQonly.txt')
Wider_1000 = loadData('../Qgen/LHsamples_wider_1000_AnnQonly.txt')

# merge all samples into a dataframe
CMIP['Ensemble'] = 'CMIP'
Paleo['Ensemble'] = 'Paleo'
Original_1000['Ensemble'] = 'Box Around Historical'
Wider_1000['Ensemble'] = 'All Encompassing'
param_bounds=np.loadtxt('../Qgen/uncertain_params_wider.txt', usecols=(1,2))[7:13,:]

# convert streamflow multipliers/deltas in samples to HMM parameter values
def convertMultToParams(multipliers):
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers.iloc[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers.iloc[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers.iloc[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers.iloc[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers.iloc[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers.iloc[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

# convert HMM parameter values to streamflow multipliers/deltas in samples 
def convertParamsToMult(params):
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

def correctMultipliers(samples):
    # find SOWs where mu0 > mu1 and swap their wet and dry state parameters
    HMMparams = convertMultToParams(samples)
    for i in range(np.shape(HMMparams)[0]):
        if HMMparams[i,0] > HMMparams[i,2]: # dry state mean above wet state mean
            # swap dry and wet state parameters to correct labels
            mu0 = HMMparams[i,2]
            std0 = HMMparams[i,3]
            mu1 = HMMparams[i,0]
            std1 = HMMparams[i,1]
            p00 = HMMparams[i,5]
            p11 = HMMparams[i,4]
            newParams = np.array([[mu0, std0, mu1, std1, p00, p11]])
            samples.iloc[i,0:6] = convertParamsToMult(newParams)[0]
            
    return samples

allSamples = pd.concat([Wider_1000,Original_1000,Paleo,CMIP,Historical])
allSamples = correctMultipliers(allSamples)

col = allSamples.columns.tolist()

sns.set_style("dark")

colors=['#bebada','#fb8072','#b3de69','#ffffb3','#80b1d3']
sns.set_palette(sns.color_palette(colors))

fig = sns.pairplot(allSamples,hue='Ensemble',corner=True)
for j in range(len(baseSOWparams[0])-1):
    for i in range(0,j+1):
        #fig.axes[j+1,i].set_xlim((np.min(allSamples[col[i]]),np.max(allSamples[col[i]])))
        #fig.axes[j+1,i].set_ylim((np.min(allSamples[col[j+1]]),np.max(allSamples[col[j+1]])))
        fig.axes[j+1,i].set_xlim((param_bounds[i][0],param_bounds[i][1]))
        fig.axes[j+1,i].set_ylim((param_bounds[j+1][0],param_bounds[j+1][1]))

plt.savefig('Figure4_AllHMMparams.pdf')
plt.clf()