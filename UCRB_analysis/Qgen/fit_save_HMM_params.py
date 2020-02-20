import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from fitHMM import *

# Fit HMM to last 70 years of historical data
AnnualQ = np.array(pd.DataFrame.from_csv('AnnualQ.csv'))[35::] # only fit to last 70 years (1944-2013)

logQ = np.log(AnnualQ[:,-1])
hidden_states, mus, sigmas, P, logProb = fitHMM(logQ)
print(mus)
print(sigmas)
print(P)

makePPandQQplot(logQ, mus, sigmas, P, 'Historical Fit ', '1944-2013')
plotDistribution(logQ, mus, sigmas, P, 'MixedGaussianFit.png')
plotLogDistribution(AnnualQ[:,-1], mus, sigmas, P, 'MixedGaussianFit_logscale.png')
plotLogTimeSeries(logQ, hidden_states, 'Flow at State Line (af)', 'StateTseries_logscale.png')
assessFit(logQ, hidden_states, mus, sigmas, P, '1944-2013', 'Historical')

# Fit HMM to period of overlap with paleo reconstruction
AnnualQ = np.array(pd.DataFrame.from_csv('AnnualQ.csv'))[0:89] # 1909-1997
print(mus)
print(sigmas)
print(P)

makePPandQQplot(logQ, mus, sigmas, P, 'Historical Fit 1909-1997', '1909-1997')
assessFit(logQ, hidden_states, mus, sigmas, P, '1909-1997', 'Historical')
