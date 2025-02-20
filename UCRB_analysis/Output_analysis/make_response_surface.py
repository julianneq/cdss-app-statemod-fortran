import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
from mpi4py import MPI
import math
import sys
plt.ioff()

design = str(sys.argv[1])
idx = np.arange(2,22,2)

LHsamples = np.loadtxt('../Qgen/' + design + '.txt')[:,7:13]
# convert streamflow multipliers/deltas in LHsamples to HMM parameter values
def convertMultToParams(multipliers):
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

# convert HMM parameter values to streamflow multipliers/deltas in LHsamples 
def convertParamsToMult(params):
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

# find SOWs where mu0 > mu1 and swap their wet and dry state parameters
HMMparams = convertMultToParams(LHsamples)
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
        LHsamples[i,:] = convertParamsToMult(newParams)

CMIPsamples = np.loadtxt('../Qgen/CMIPunscaled_SOWs.txt')[:,7:13]
PaleoSamples = np.loadtxt('../Qgen/Paleo_SOWs.txt')[:,7:13]
if design == 'LHsamples_original_1000_AnnQonly' or design == 'LHsamples_original_200_AnnQonly':
    param_bounds=np.loadtxt('../Qgen/uncertain_params_original.txt', usecols=(1,2))[7:13,:]
elif design == 'LHsamples_narrowed_1000_AnnQonly' or design == 'LHsamples_narrowed_200_AnnQonly':
    param_bounds=np.loadtxt('../Qgen/uncertain_params_narrowed.txt', usecols=(1,2))[7:13,:]
elif design == 'LHsamples_wider_1000_AnnQonly' or design == 'LHsamples_wider_200_AnnQonly':
    param_bounds=np.loadtxt('../Qgen/uncertain_params_wider.txt', usecols=(1,2))[7:13,:]
elif design == 'Paleo_SOWs':
	param_bounds=np.loadtxt('../Qgen/uncertain_params_paleo.txt',usecols=(1,2))[7:13,:]
elif design == 'CMIP_SOWs':
	param_bounds=np.loadtxt('../Qgen/uncertain_params_CMIP.txt',usecols=(1,2))[7:13,:]
elif design == 'CMIPunscaled_SOWs':
    param_bounds=np.loadtxt('../Qgen/uncertain_params_CMIPunscaled.txt',usecols=(1,2))[7:13,:]
	
SOW_values = np.array([1,1,1,1,0,0]) #Default parameter values for base SOW
realizations = 10
param_names=['XBM_mu0','XBM_sigma0','XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11']
params_no = len(param_names)
problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
}

# remove samples no longer in param_bounds
rows_to_keep = np.union1d(np.where(LHsamples[:,0]>=0)[0],np.where(LHsamples[:,0]<=0)[0])
for i in range(params_no):
    within_rows = np.intersect1d(np.where(LHsamples[:,i] >= param_bounds[i][0])[0], np.where(LHsamples[:,i] <= param_bounds[i][1])[0])
    rows_to_keep = np.intersect1d(rows_to_keep,within_rows)

LHsamples = LHsamples[rows_to_keep,:]
samples = len(LHsamples[:,0])

percentiles = np.arange(10, 110, 10)
all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist() 
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

#==============================================================================
# Function for water years
#==============================================================================
empty=[]
n=12
HIS_short = np.loadtxt('../../../'+design+'/Infofiles/7202003/7202003_info_hist.txt')[:,2]
# replace failed runs with np.nan (currently -999.9)
HIS_short[HIS_short < 0] = np.nan

def shortage_duration(sequence):
    cnt_shrt = [sequence[i]>0 for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur

def fitOLS(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0])
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors + ['Interaction']
    #fit OLS regression
    ols = sm.OLS(np.log(dta['Shortage']+0.0001), dta[cols])
    result = ols.fit()
    return result

def plotContourMap(ax, result, dta, contour_cmap, dot_cmap, xgrid, ygrid, \
    xvar, yvar):
 
    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    grid = np.column_stack([np.ones(len(x)),x,y,x*y])
 
    z = np.exp(result.predict(grid))-0.0001
    Z = np.reshape(z, np.shape(X))
    norm = mpl.colors.Normalize(np.min(z), np.max(z))

    contourset = ax.contourf(X, Y, Z, cmap=contour_cmap, norm=norm)
    ax.scatter(dta[xvar].values, dta[yvar].values, c=dta['Shortage'].values, edgecolor='none', cmap=dot_cmap, norm=norm)
    ax.set_xlim(0.99*np.nanmin(X),1.01*np.nanmax(X))
    ax.set_ylim(0.99*np.nanmin(Y),1.01*np.nanmax(Y))
    ax.set_xlabel(xvar,fontsize=10)
    ax.set_ylabel(yvar,fontsize=10)
    ax.tick_params(axis='both',labelsize=6)
    return contourset

def plotContourSOWmap(ax, result, CMIP, Paleo, contour_cmap, xgrid, ygrid, \
    xvar, yvar):
 
    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    grid = np.column_stack([np.ones(len(x)),x,y,x*y])
 
    z = np.exp(result.predict(grid))-0.0001
    Z = np.reshape(z, np.shape(X))
    norm = mpl.colors.Normalize(np.min(z), np.max(z))

    contourset = ax.contourf(X, Y, Z, cmap=contour_cmap, norm=norm)
    ax.scatter(CMIP[xvar].values, CMIP[yvar].values, c='#ffffb3', edgecolor='none')
    ax.scatter(Paleo[xvar].values, Paleo[yvar].values, c='#b3de69', edgecolor='none')
    ax.set_xlim(0.99*np.nanmin(X),1.01*np.nanmax(X))
    ax.set_ylim(0.99*np.nanmin(Y),1.01*np.nanmax(Y))
    ax.set_xlabel(xvar,fontsize=10)
    ax.set_ylabel(yvar,fontsize=10)
    ax.tick_params(axis='both',labelsize=6)
    return contourset

def make_response_surface(ID):
    '''
    Perform analysis for shortage magnitude
    '''
    R2_scores = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    SYN_short = np.zeros([len(HIS_short), samples * realizations])
    for j in range(samples):
        data= np.loadtxt('../../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_' + str(rows_to_keep[j]+1) + '.txt')
        # replace failed runs with np.nan (currently -999.9)
        data[data < 0] = np.nan
        try:
            SYN_short[:,j*realizations:j*realizations+realizations]=data[:,idx]
        except IndexError:
            print(ID + '_info_' + str(j+1))
    # Reshape into water years
    # Create matrix of [no. years x no. months x no. experiments]
    f_SYN_short = np.zeros([int(np.size(HIS_short)/n),n, samples*realizations])
    for i in range(samples*realizations):
        f_SYN_short[:,:,i]= np.reshape(SYN_short[:,i], (int(np.size(SYN_short[:,i])/n), n))

    # Shortage per water year
    f_SYN_short_WY = np.sum(f_SYN_short,axis=1)

    # Identify droughts at percentiles
    syn_magnitude = np.zeros([len(percentiles),samples*realizations])
    for j in range(samples*realizations):
        syn_magnitude[:,j]=[np.percentile(f_SYN_short_WY[:,j], i) for i in percentiles]

    # define color map for shortage contours
    dot_cmap = mpl.cm.get_cmap('RdBu_r')
    contour_cmap = mpl.cm.get_cmap('RdBu_r')
    CMIP = pd.DataFrame(data = np.repeat(CMIPsamples, realizations, axis = 0), columns=param_names)
    Paleo = pd.DataFrame(data = np.repeat(PaleoSamples, realizations, axis = 0), columns=param_names)

    # OLS regression analysis
    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
    for i in range(len(percentiles)):
        dta['Shortage']=syn_magnitude[i,:]
        # find two most informative predictors
        R2_scores = pd.read_csv('../../../'+design+'/Magnitude_Sensitivity_analysis/'+ ID + '_R2.csv')
        percentile_scores = R2_scores[str(int(percentiles[i]-1))]
        if percentile_scores[0] > 0:
            top_two = list(np.argsort(percentile_scores)[4::]) # sorts from lowest to highest so take last two
            predictors = list([param_names[top_two[1]],param_names[top_two[0]]])
            dta['Interaction'] = dta[predictors[0]]*dta[predictors[1]]
            result = fitOLS(dta, predictors)
            fig, axes = plt.subplots(1,1)
            xgrid = np.arange(param_bounds[top_two[1]][0], param_bounds[top_two[1]][1], \
            	    np.around((param_bounds[top_two[1]][1]-param_bounds[top_two[1]][0])/100,decimals=4))
            ygrid = np.arange(param_bounds[top_two[0]][0], param_bounds[top_two[0]][1], \
            	    np.around((param_bounds[top_two[0]][1]-param_bounds[top_two[0]][0])/100,decimals=4))
            contourset = plotContourMap(axes, result, dta, contour_cmap, dot_cmap, \
            	xgrid, ygrid, predictors[0], predictors[1])
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(contourset, cax=cbar_ax)
            cbar_ax.set_ylabel('Shortage',fontsize=12)
            yticklabels = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(yticklabels,fontsize=10)
            fig.set_size_inches([14.5,8])
            fig.suptitle(str(int(percentiles[i])) + 'th percentile shortage for '+ ID)
            fig.savefig('../../../'+design+'/Factor_mapping/ResponseSurfaces/'+\
                        ID+'/'+str(int(percentiles[i]))+'th_percentile_shortage.png')
            plt.close()

            fig, axes = plt.subplots(1,1)
            contourset = plotContourSOWmap(axes, result, CMIP, Paleo, contour_cmap, \
                        xgrid, ygrid, predictors[0], predictors[1])
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(contourset, cax=cbar_ax)
            cbar_ax.set_ylabel('Shortage',fontsize=12)
            yticklabels = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(yticklabels,fontsize=10)
            fig.set_size_inches([14.5,8])
            fig.suptitle(str(int(percentiles[i])) + 'th percentile shortage for '+ ID)
            fig.savefig('../../../'+design+'/Factor_mapping/ResponseSurfaces/'+\
                        ID+'/'+str(int(percentiles[i]))+'th_percentile_shortage_SOWs.png')
            plt.close()

    return None

# =============================================================================
# Start parallelization (running each structure in parallel)
# =============================================================================
    
# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(nStructures/nprocs))
remainder = nStructures % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count

# Run simulation
for k in range(start, stop):
    make_response_surface(all_IDs[k])