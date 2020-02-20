import numpy as np
import pandas as pd
from fitHMM import fitHMM
import scipy.stats as ss

# fit HMM parameters
def fitParams(flows):
    # create matrices to store the parameters
    # elements are mu0, sigma0, mu1, sigma1, p00, p11
    params = np.zeros([6])
    hidden_states, mus, sigmas, P, logProb = fitHMM(np.log(flows))
    
    params[0] = mus[0]
    params[1] = sigmas[0]
    params[2] = mus[1]
    params[3] = sigmas[1]
    params[4] = P[0,0]
    params[5] = P[1,1]
        
    return params

# fit HMM to historical record
AnnualQ = np.array(pd.DataFrame.from_csv('AnnualQ.csv'))
histParams = fitParams(AnnualQ[35::,-1]) # last 2/3 of record

# IWR, RES, TBD, M_I, Shoshone, ENVflows, EVAdelta, mu0, sigma0, mu1, sigma1, p00, p11, snowshift
baseSOWparams = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


# load CMIP 3 and CMIP 5 flow data at last node
def processCMIPdata(version, nsims):
    # load flows
    CMIPflows = np.loadtxt('CMIP/CMIP' + version + '_flows.csv',delimiter=',')
    
    # reshape into nsims x 64 years x 12 months
    CMIPflows = np.reshape(CMIPflows,[nsims,64,12]) # simulation x year x month
    
    # calculate time series of annual flows in each CMIP simulation
    CMIPflows = np.sum(CMIPflows,2)
    
    # fit HMM to CMIP scenarios
    CMIPparams = np.zeros([np.shape(CMIPflows)[0],6])
    for i in range(np.shape(CMIPflows)[0]):
        CMIPparams[i,:] = fitParams(CMIPflows[i,:])
        
    CMIP_SOWs = np.tile(baseSOWparams,(np.shape(CMIPflows)[0],1))
    CMIP_SOWs[:,7:11] = CMIPparams[:,0:4] / histParams[0:4]
    CMIP_SOWs[:,11:13] = CMIPparams[:,4::] - histParams[4::]
    
    np.savetxt('CMIP' + version + '_SOWs.txt',CMIP_SOWs)
    
    return CMIP_SOWs

CMIP3_SOWs = processCMIPdata('3',112)
CMIP5_SOWs =processCMIPdata('5',97)
CMIP_SOWs = np.concatenate((CMIP3_SOWs,CMIP5_SOWs),0)

np.savetxt('CMIP3_SOWs.txt',CMIP3_SOWs)
np.savetxt('CMIP5_SOWs.txt',CMIP5_SOWs)
np.savetxt('CMIP_SOWs.txt',CMIP_SOWs)

# repeat with unscaled CMIP data and mean parameters
CMIPunscaled_params = np.loadtxt('CMIP/MeanCMIPparams.txt',skiprows=1)
CMIPunscaled_SOWs = np.tile(baseSOWparams,(np.shape(CMIPunscaled_params)[0],1))
CMIPunscaled_SOWs[:,7:11] = CMIPunscaled_params[:,0:4] / histParams[0:4]
CMIPunscaled_SOWs[:,11:13] = CMIPunscaled_params[:,4::] - histParams[4::]

np.savetxt('CMIPunscaled_SOWs.txt',CMIPunscaled_SOWs)


# repeat with Paleo data
meanPaleoParams = np.loadtxt('CMIP/MeanCMIPparams.txt',skiprows=1)

Paleo_SOWs = np.tile(baseSOWparams,(429-64+1,1))
Paleo_SOWs[:,7:11] = meanPaleoParams[:,0:4] / histParams[0:4]
Paleo_SOWs[:,11:13] = meanPaleoParams[:,4::] - histParams[4::]

np.savetxt('Paleo_SOWs.txt',Paleo_SOWs)


# remove demand and snowmelt changes from LHsample designs
def changeOnlyAnnualQ(filename):
    design = np.loadxt(filename + '.txt')

    design[:,0:7] = np.tile(baseSOWparams[0:7],(np.shape(design)[0],1))
    design[:,-1] = np.tile(baseSOWparams[-1],(np.shape(design)[0]))

    np.savetxt(filename + '_AnnQonly.txt', design)

    return None

changeOnlyAnnualQ('LHsamples_narrowed_200')
changeOnlyAnnualQ('LHsamples_original_200')
changeOnlyAnnualQ('LHsamples_wider_200')
changeOnlyAnnualQ('LHsamples_narrowed_1000')
changeOnlyAnnualQ('LHsamples_original_1000')
changeOnlyAnnualQ('LHsamples_wider_1000')