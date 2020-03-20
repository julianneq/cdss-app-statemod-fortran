import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils import setupProblem, getSamples, fitOLS_interact, calc_syn_magnitude
from makeFigure6_ShortageDistns import plotSDC
from makeFigure7_VarianceDecomposition import plotSums
from makeFigure8_ResponseSurfaces import plotResponseSurface

def makeFigureS9_ResponseSurfaces2():

    sns.set_style("white")
    
    # constants, vectors
    design = 'LHsamples_wider_1000_AnnQonly'
    structure = '7200645'
    idx = np.arange(2,22,2)
    percentiles = [50, 90]
    nrealizations = 10
    nyears = 105
    nmonths = 12
    
    # plotting characteristics
    shortage_cmap = mpl.cm.get_cmap('RdBu_r')
    colors = ['#de2d26', '#fb6a4a', '#3182bd', '#6baed6', '#a50f15', '#08519c', '#9e9ac8']
              
    # find which samples are still in param_bounds after flipping misidentified wet and dry states
    param_bounds, param_names, params_no, problem = setupProblem(design)
    samples, rows_to_keep = getSamples(design, params_no, param_bounds)
    nsamples = len(rows_to_keep)
    
    # load historical shortage data and convert acre-ft to m^3
    hist_short = np.loadtxt('../Simulation_outputs/' + structure + '_info_hist.txt')[:,2]*1233.48
    # replace failed runs with np.nan (currently -999.9)
    hist_short[hist_short < 0] = np.nan
    
    # load shortage data for this experimental design
    SYN_short = np.load('../Simulation_outputs/' + design + '/' + structure + '_info.npy')
    # remove columns for year (0) and demand (odd columns) and convert acre-ft to m^3
    SYN_short = SYN_short[:,idx,:]*1233.48
    SYN_short = SYN_short[:,:,rows_to_keep]
    # replace failed runs with np.nan (currently -999.9)
    SYN_short[SYN_short < 0] = np.nan
    # Identify droughts at percentiles
    syn_magnitude = calc_syn_magnitude(nyears, nmonths, nrealizations, nsamples, percentiles, SYN_short)
    # reshape synthetic shortage data into 12*nyears x nsamples*nrealizations
    SYN_short = SYN_short.reshape([np.shape(SYN_short)[0],np.shape(SYN_short)[1]*np.shape(SYN_short)[2]])
    
    # create data frames of shortage and SOWs
    CMIPsamples = np.loadtxt('../Qgen/CMIPunscaled_SOWs.txt')[:,7:13]
    PaleoSamples = np.loadtxt('../Qgen/Paleo_SOWs.txt')[:,7:13]
    CMIP = pd.DataFrame(data = np.repeat(CMIPsamples, nrealizations, axis = 0), columns=param_names)
    Paleo = pd.DataFrame(data = np.repeat(PaleoSamples, nrealizations, axis = 0), columns=param_names)
    dta = pd.DataFrame(data = np.repeat(samples, nrealizations, axis = 0), columns=param_names)
    R2_scores = pd.read_csv('../Simulation_outputs/' + design + '/' + structure + '_R2.csv')
    
    
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.5))
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    # plot shortage distribution for this structure under all-encompassing experiment
    ax1 = axes[0,0]
    handles, labels = plotSDC(ax1, SYN_short, hist_short, nsamples, nrealizations)
    ax1.set_ylim(0,370000000)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
    ax1.tick_params(axis='both',labelsize=14)
    ax1.set_ylabel('Shortage (m' + r'$^3$' + ')',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax1.plot([percentile, percentile],[0,6200000],c='k')
    
    # plot variance decomposition for this structure under all-encompassing experiment
    ax2 = axes[1,0]
    S1_values = pd.read_csv('../Simulation_outputs/' + design + '/'+ structure + '_S1.csv')
    plotSums(S1_values, ax2, colors)
    ax2.set_ylim([0,1])
    ax2.tick_params(axis='both',labelsize=14)
    ax2.set_ylabel('Portion of Variance',fontsize=14)
    ax2.set_xlabel('Shortage Percentile',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax2.plot([percentile, percentile],[0,1],c='k')
    
    for i in range(len(percentiles)):
        dta['Shortage'] = syn_magnitude[i,:]
        # find average shortage across realizations in each SOW
        avg_dta = dta.groupby(['mu0','mu1','sigma0','sigma1','p00','p11'],as_index=False)[['Shortage']].mean()
        percentile_scores = R2_scores[str(int(percentiles[i]-1))]
        if percentile_scores[0] > 0:
            top_two = list(np.argsort(percentile_scores)[::-1][:2]) # sorts from lowest to highest so take last two
            predictors = list([param_names[top_two[1]],param_names[top_two[0]]])
            avg_dta['Interaction'] = avg_dta[predictors[0]]*avg_dta[predictors[1]]
            result = fitOLS_interact(avg_dta, predictors)
            xgrid = np.arange(param_bounds[top_two[1]][0], param_bounds[top_two[1]][1], \
            	    np.around((param_bounds[top_two[1]][1]-param_bounds[top_two[1]][0])/100,decimals=4))
            ygrid = np.arange(param_bounds[top_two[0]][0], param_bounds[top_two[0]][1], \
            	    np.around((param_bounds[top_two[0]][1]-param_bounds[top_two[0]][0])/100,decimals=4))
            
            # plot average shortage in each SOW and prediction from regression
            plotResponseSurface(axes[0,i+1], result, avg_dta, CMIP, Paleo, shortage_cmap, shortage_cmap, \
            	xgrid, ygrid, predictors[0], predictors[1], otherSOWs = False)
            axes[0,i+1].set_title(str(percentiles[i]) + 'th Percentile',fontsize=16)
            fig.savefig('FigureS9_ResponseSurfaces2.pdf')
            
            # plot prediction from regression with CMIP and Paleo samples on top
            plotResponseSurface(axes[1,i+1], result, avg_dta, CMIP, Paleo, shortage_cmap, shortage_cmap, \
            	xgrid, ygrid, predictors[0], predictors[1], otherSOWs = True)
            fig.savefig('FigureS9_ResponseSurfaces2.pdf')

    fig.savefig('FigureS9_ResponseSurfaces2.pdf')
    fig.clf()
    
    return None