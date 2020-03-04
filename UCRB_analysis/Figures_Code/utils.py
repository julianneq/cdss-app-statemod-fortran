#from hmmlearn.hmm import GaussianHMM
from scipy import stats as ss
import numpy as np
import pandas as pd
from SALib.analyze import delta
import statsmodels.api as sm
'''
def fitHMM(TransformedQ):
    # fit HMM
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(TransformedQ,[len(TransformedQ),1]))
    hidden_states = model.predict(np.reshape(TransformedQ,[len(TransformedQ),1]))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
    
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
    
    return hidden_states, mus, sigmas, P
'''
def findQuantiles(mus, sigmas, piNew):
    x = np.empty([10000])
    qx = np.empty([10000])
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    qx = piNew[0]*ss.norm.cdf(x,mus[0],sigmas[0]) + \
        piNew[1]*ss.norm.cdf(x,mus[1],sigmas[1])
        
    return x, qx

def convertMultToParams(multipliers):
    '''convert streamflow multipliers/deltas in LHsamples to HMM parameter values'''
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

def convertParamsToMult(params):
    '''convert HMM parameter values to streamflow multipliers/deltas in LHsamples '''
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

def fitOLS(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0])
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors
    #fit OLS regression
    ols = sm.OLS(dta['Shortage'], dta[cols])
    result = ols.fit()
    return result

def Sobol_per_structure(design, ID):
    # set up problem for SALib
    if design == 'LHsamples_original_1000_AnnQonly':
        param_bounds=np.loadtxt('../Qgen/uncertain_params_original.txt', usecols=(1,2))[7:13,:]
    elif design == 'LHsamples_wider_1000_AnnQonly':
        param_bounds=np.loadtxt('../Qgen/uncertain_params_wider.txt', usecols=(1,2))[7:13,:]
    elif design == 'Paleo_SOWs':
    	param_bounds=np.loadtxt('../Qgen/uncertain_params_paleo.txt',usecols=(1,2))[7:13,:]
    elif design == 'CMIP_SOWs':
    	param_bounds=np.loadtxt('../Qgen/uncertain_params_CMIP.txt',usecols=(1,2))[7:13,:]
    elif design == 'CMIPunscaled_SOWs':
        param_bounds=np.loadtxt('../Qgen/uncertain_params_CMIPunscaled.txt',usecols=(1,2))[7:13,:]
    
    param_names=['mu0','sigma0','mu1','sigma1','p00','p11']
    params_no = len(param_names)
    problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
    }
    
    samples = np.loadtxt('../Qgen/' + design + '.txt')[:,7:13]
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
            samples[i,:] = convertParamsToMult(newParams)
            
    # remove samples no longer in param_bounds
    rows_to_keep = np.intersect1d(np.where(samples[:,0]>=0)[0],np.where(samples[:,0]<=0)[0])
    for i in range(params_no):
        within_rows = np.intersect1d(np.where(samples[:,i] > param_bounds[i][0])[0], np.where(samples[:,i] < param_bounds[i][1])[0])
        rows_to_keep = np.union1d(rows_to_keep,within_rows)
    
    samples = samples[rows_to_keep,:]
    
    # constants
    percentiles = np.arange(0,100)
    nsamples = len(rows_to_keep)
    nrealizations = 10
    idx = np.arange(2,22,2)
    nyears = 105
    nmonths = 12
    
    # initiate dataframes to store sensitivity results
    S1 = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    S1_conf = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    R2_scores = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    S1.index=S1_conf.index = R2_scores.index = param_names
    
    # load shortage data for this experimental design
    SYN_short = np.load('../../../Simulation_outputs/' + design + '/' + ID + '_info.npy')
    # remove columns for year (0) and demand (odd columns)
    SYN_short = SYN_short[:,idx,:]
    SYN_short = SYN_short[:,:,rows_to_keep]
    # replace failed runs with np.nan (currently -999.9)
    SYN_short[SYN_short < 0] = np.nan
    
    # Reshape into water years
    # Create matrix of [no. years x no. months x no. experiments]
    f_SYN_short = np.zeros([nyears, nmonths, nsamples*nrealizations])
    for i in range(nsamples):
        for j in range(nrealizations):
            f_SYN_short[:,:,i*nrealizations+j] = np.reshape(SYN_short[:,j,i], [nyears, nmonths])

    # Shortage per water year
    f_SYN_short_WY = np.sum(f_SYN_short,axis=1)

    # Identify droughts at percentiles
    syn_magnitude = np.zeros([len(percentiles),nsamples*nrealizations])
    for j in range(nsamples*nrealizations):
        syn_magnitude[:,j] = [np.percentile(f_SYN_short_WY[:,j], i) for i in percentiles]

    # Delta Method analysis
    for i in range(len(percentiles)):
        if syn_magnitude[i,:].any():
            try:
                result = delta.analyze(problem, np.repeat(samples, nrealizations, axis = 0), syn_magnitude[i,:], print_to_console=False, num_resamples=2)
                S1[percentiles[i]] = result['S1']
                S1_conf[percentiles[i]] = result['S1_conf']
            except:
                pass

    S1.to_csv('../Simulation_outputs/' + design + '/'+ ID + '_S1.csv')
    S1_conf.to_csv('../Simulation_outputs/' + design + '/'+ ID + '_S1_conf.csv')
    
    # OLS regression analysis
    dta = pd.DataFrame(data = np.repeat(samples, nrealizations, axis = 0), columns=param_names)
    for i in range(len(percentiles)):
        dta['Shortage'] = syn_magnitude[i,:]
        for m in range(params_no):
            predictors = dta.columns.tolist()[m:(m+1)]
            result = fitOLS(dta, predictors)
            R2_scores.at[param_names[m],percentiles[i]]=result.rsquared
            
    R2_scores.to_csv('../Simulation_outputs/' + design + '/' + ID + '_R2.csv')
    
    return None
