import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils import calcFailureHeatmap, setupProblem, getSamples, fitLogit, fitLogit_interact, roundup, calcPseudoR2
from makeFigure6_ShortageDistns import plotSDC
from makeFigure9_FactorMaps import plotFailureHeatmap, highlight_cell, plotFactorMap

def makeFigureS10_FactorMaps2():

    sns.set_style("white")
    
    # constants, vectors
    design = 'LHsamples_wider_1000_AnnQonly'
    structure = '7200645'
    idx = np.arange(2,22,2)
    percentiles = [50, 70, 90, 90]
    short_magnitudes = [10, 10, 10, 30]
    nrealizations = 10
    axes_indices = [(0,1),(0,2),(1,1),(1,2)]
    
    # plotting characteristics
    probability_cmap = mpl.cm.get_cmap('RdBu')
    success_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
    contour_levels = np.arange(0.0, 1.05,0.1)
              
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
    # reshape synthetic shortage data into 12*nyears x nsamples*nrealizations
    SYN_short = SYN_short.reshape([np.shape(SYN_short)[0],np.shape(SYN_short)[1]*np.shape(SYN_short)[2]])
    
    # create data frames of shortage and SOWs
    dta = pd.DataFrame(data = np.repeat(samples, nrealizations, axis = 0), columns=param_names)
    
    
    fig, axes = plt.subplots(2,3,figsize=(18.2,9.1))
    fig.subplots_adjust(hspace=0.5,right=0.8,wspace=0.5)  
    # plot shortage distribution for this structure under all-encompassing experiment
    ax1 = axes[0,0]
    handles, labels = plotSDC(ax1, SYN_short, hist_short, nsamples, nrealizations)
    ax1.set_ylim([0,370000000])
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
    ax1.tick_params(axis='both',labelsize=14)
    ax1.set_ylabel('Shortage (m' + r'$^3$' + ')',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax1.plot([percentile, percentile],[0,370000000],c='k')
    
    # plotfailure heatmap for this structure under all-encompassing experiment
    ax2 = axes[1,0]
    allSOWs, historic_percents, frequencies, magnitudes, gridcells, im = plotFailureHeatmap(ax2, design, structure)
    addPercentileBlocks(historic_percents, gridcells, percentiles, short_magnitudes, ax2)
    allSOWsperformance = allSOWs/100
    historic_percents = [roundup(x) for x in historic_percents]
    all_pseudo_r_scores = calcPseudoR2(frequencies, magnitudes, params_no, allSOWsperformance, dta, structure, design)
    
    for i in range(len(percentiles)):
        dta['Success'] = allSOWsperformance[list(frequencies).index(100-percentiles[i]),int(short_magnitudes[i]/10)-1,:]
        # consider each SOW a success if 50% or more realizations were a success
        avg_dta = dta.groupby(['mu0','mu1','sigma0','sigma1','p00','p11'],as_index=False)[['Success']].mean()
        avg_dta.loc[np.where(avg_dta['Success']>=0.5)[0],'Success'] = 1
        avg_dta.loc[np.where(avg_dta['Success']<0.5)[0],'Success'] = 0
        # load pseudo R2 of predictors for this magnitude/frequency combination
        pseudo_r_scores = all_pseudo_r_scores[str((100-percentiles[i]))+'yrs_'+str(short_magnitudes[i])+'prc'].values
        if pseudo_r_scores.any():
            top_predictors = np.argsort(pseudo_r_scores)[::-1][:2]
            ranges = param_bounds[top_predictors]
            # define grid of x (1st predictor), and y (2nd predictor) dimensions
            # to plot contour map over
            xgrid = np.arange(param_bounds[top_predictors[0]][0], 
                              param_bounds[top_predictors[0]][1], np.around((ranges[0][1]-ranges[0][0])/100,decimals=4))
            ygrid = np.arange(param_bounds[top_predictors[1]][0], 
                              param_bounds[top_predictors[1]][1], np.around((ranges[1][1]-ranges[1][0])/100,decimals=4))
            all_predictors = [ avg_dta.columns.tolist()[k] for k in top_predictors]
            avg_dta['Interaction'] = avg_dta[all_predictors[0]]*avg_dta[all_predictors[1]]
            # fit logistic regression model with top two predictors of success and their interaction
            # unless the matrix is non-singular; then drop the interaction
            try:
                result = fitLogit_interact(avg_dta, [all_predictors[k] for k in [0,1]])
            except:
                result = fitLogit(avg_dta, [all_predictors[k] for k in [0,1]])
            
            # plot success/failure for each SOW on top of logistic regression estimate of probability of success
            contourset = plotFactorMap(axes[axes_indices[i]], result, avg_dta, probability_cmap, success_cmap, contour_levels, xgrid, ygrid, \
                          all_predictors[0], all_predictors[1])
            axes[axes_indices[i]].set_title("Success if " + str(short_magnitudes[i]) + "% shortage\n<" + str((100-percentiles[i])) + "% of the time", fontsize=16)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contourset, cax=cbar_ax)
    cbar.ax.set_ylabel("Predicted Probability of Success", rotation=-90, va="bottom",fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig("FigureS10_FactorMaps2.pdf")
    fig.clf()
    
    return None

def addPercentileBlocks(historic_percents, gridcells, percentiles, short_magnitudes, ax):   
    for i in range(len(percentiles)):
        highlight_cell(int(short_magnitudes[i]/10)-1, int((100-percentiles[i])/10)-1, ax, color="black", linewidth=2)
            
    for i in range(len(historic_percents)):
        if historic_percents[i] != 0:
            highlight_cell(i, gridcells[i], ax, color="orange", linewidth=2)
                
    return None