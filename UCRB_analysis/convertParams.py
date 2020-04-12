import numpy as np

def convertLogMultToRealMult(logMultipliers):
    # mu0, sigma0, mu1, sigma1, p00, p11
    # first four columns are multipliers in log-space; conver to multipliers in real-space
    realMultipliers = np.zeros(np.shape(logMultipliers))
    realMultipliers[:,4::] = logMultipliers[:,4::] # transition probability shifts unchanged
    
    logBase = np.array([15.258112,0.259061,15.661007,0.252174])
    realBase = np.zeros(np.shape(logBase))
    realBase[0] = np.exp(logBase[0]+0.5*logBase[1]**2) # real space mu0
    realBase[2] = np.exp(logBase[2]+0.5*logBase[3]**2) # real space mu1
    realBase[1] = np.sqrt(realBase[0]**2 * (np.exp(logBase[1])-1)) # real space sigma0
    realBase[3] = np.sqrt(realBase[2]**2 * (np.exp(logBase[3])-1)) # real space sigma1
    
    logParams = convertMultToParams(logMultipliers)
    realParams = np.zeros(np.shape(logParams))
    realParams[:,0] = np.exp(logParams[:,0]+0.5*logParams[:,1]**2)
    realParams[:,2] = np.exp(logParams[:,2]+0.5*logParams[:,3]**2)
    realParams[:,1] = np.sqrt(realParams[:,0]**2 * (np.exp(logParams[:,1])-1))
    realParams[:,3] = np.sqrt(realParams[:,2]**2 * (np.exp(logParams[:,3])-1))
    
    realMultipliers[:,0:4] = realParams[:,0:4]/realBase
    
    return realMultipliers

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

def getSamples(design, params_no, param_bounds):
    samples = np.loadtxt('Qgen/' + design + '.txt')[:,7:13]
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
    rows_to_keep = np.union1d(np.where(samples[:,0]>=0)[0],np.where(samples[:,0]<=0)[0])
    for i in range(params_no):
        within_rows = np.intersect1d(np.where(samples[:,i] >= param_bounds[i][0])[0], np.where(samples[:,i] <= param_bounds[i][1])[0])
        rows_to_keep = np.intersect1d(rows_to_keep,within_rows)
    
    samples = samples[rows_to_keep,:]
    
    return samples, rows_to_keep

def setupProblem(design):
    # set up problem for SALib
    if design == 'LHsamples_original_1000_AnnQonly' or design == 'LHsamples_original_100_AnnQonly':
        param_bounds = np.loadtxt('Qgen/uncertain_params_original.txt', usecols=(1,2))[7:13,:]
    elif design == 'LHsamples_wider_1000_AnnQonly' or design == 'LHsamples_wider_1000_AnnQonly':
        param_bounds = np.loadtxt('Qgen/uncertain_params_wider.txt', usecols=(1,2))[7:13,:]
    elif design == 'CMIPunscaled_SOWs':
        param_bounds = np.loadtxt('Qgen/uncertain_params_CMIPunscaled.txt', usecols=(1,2))[7:13,:]
    elif design == 'Paleo_SOWs':
        param_bounds = np.loadtxt('Qgen/uncertain_params_paleo.txt', usecols=(1,2))[7:13,:]
     
    param_names=['mu0','sigma0','mu1','sigma1','p00','p11']
    params_no = len(param_names)
    problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
    }
    
    return param_bounds, param_names, params_no, problem

designs = ['LHsamples_original_1000_AnnQonly','LHsamples_wider_1000_AnnQonly','CMIPunscaled_SOWs','Paleo_SOWs']
for design in designs:
    param_bounds, param_names, params_no, problem = setupProblem(design)
    samples, rows_to_keep = getSamples(design, params_no, param_bounds)
    RealSpaceSamples = convertLogMultToRealMult(samples)
    print(np.min(RealSpaceSamples,0))
    print(np.max(RealSpaceSamples,0))