import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils import calcFailureHeatmap, setupProblem, getSamples, fitLogit_interact, roundup, calcPseudoR2
from makeFigure6_ShortageDistns import plotSDC
from makeFigure8_ResponseSurfaces import getLabels

def makeFigure9_FactorMaps():

    sns.set_style("white")
    
    # constants, vectors
    design = 'LHsamples_wider_1000_AnnQonly'
    structure = '53_ADC022'
    short_idx = np.arange(2,22,2)
    demand_idx = np.arange(1,21,2)
    percentiles = [50, 90]
    nrealizations = 10
    
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
    hist_demand = np.loadtxt('../Simulation_outputs/' + structure + '_info_hist.txt')[:,1]*1233.48
    # replace failed runs with np.nan (currently -999.9)
    hist_short[hist_short < 0] = np.nan
    
    # load shortage data for this experimental design
    SYN = np.load('../Simulation_outputs/' + design + '/' + structure + '_info.npy')
    # extract columns for year shortage and demand and convert acre-ft to ^3
    SYN_short = SYN[:,short_idx,:]*1233.48
    SYN_demand = SYN[:,demand_idx,:]*1233.48
    # use just the samples within the experimental design
    SYN_short = SYN_short[:,:,rows_to_keep]
    SYN_demand = SYN_demand[:,:,rows_to_keep]
    # replace failed runs with np.nan (currently -999.9)
    SYN_short[SYN_short < 0] = np.nan
    # reshape synthetic shortage data into 12*nyears x nsamples*nrealizations
    SYN_short = SYN_short.reshape([np.shape(SYN_short)[0],np.shape(SYN_short)[1]*np.shape(SYN_short)[2]])
    SYN_demand = SYN_demand.reshape([np.shape(SYN_demand)[0],np.shape(SYN_demand)[1]*np.shape(SYN_demand)[2]])
    
    # create data frames of shortage and SOWs
    dta = pd.DataFrame(data = np.repeat(samples, nrealizations, axis = 0), columns=param_names)
    
    
    fig, axes = plt.subplots(2,4,figsize=(24.3,9.1))
    fig.subplots_adjust(hspace=0.5,right=0.8,wspace=0.5)        
    # plot shortage distribution for this structure under all-encompassing experiment
    ax1 = axes[0,0]
    handles, labels = plotSDC(ax1, SYN_short, SYN_demand, hist_short, hist_demand, nsamples, nrealizations, True)
    ax1.set_ylim([0,1])
    ax1.tick_params(axis='both',labelsize=14)
    ax1.set_ylabel('Shortage/Demand',fontsize=14)
    ax1.set_xlabel('Shortage Percentile',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax1.plot([percentile, percentile],[0,1],c='k')
    
    # plotfailure heatmap for this structure under all-encompassing experiment
    ax2 = axes[1,0]
    allSOWs, historic_percents, frequencies, magnitudes, gridcells, im = plotFailureHeatmap(ax2, design, structure)
    addPercentileBlocks(historic_percents, gridcells, percentiles, ax2)
    allSOWsperformance = allSOWs/100
    historic_percents = [roundup(x) for x in historic_percents]
    #all_pseudo_r_scores = calcPseudoR2(frequencies, magnitudes, params_no, allSOWsperformance, dta, structure, design)
    all_pseudo_r_scores = pd.read_csv("../Simulation_outputs/" + design + "/" + structure + "_pseudo_r_scores.csv")
    
    for i in range(len(percentiles)):
        for j in range(3):
            # magnitude of shortage at this percentile to plot
            h = np.where(np.array(historic_percents) == 100 - percentiles[i])[0][0]
            if j == 0:
                h -= 2
            elif j == 2:
                h += 2
            # find out if each realization was a success or failure at this magnitude/frequency combination
            dta['Success'] = allSOWsperformance[list(frequencies).index(100-percentiles[i]),h,:]
            # consider each SOW a success if 50% or more realizations were a success
            avg_dta = dta.groupby(['mu0','mu1','sigma0','sigma1','p00','p11'],as_index=False)[['Success']].mean()
            avg_dta.loc[np.where(avg_dta['Success']>=0.5)[0],'Success'] = 1
            avg_dta.loc[np.where(avg_dta['Success']<0.5)[0],'Success'] = 0
            # load pseudo R2 of predictors for this magnitude/frequency combination
            pseudo_r_scores = all_pseudo_r_scores[str((100-percentiles[i]))+'yrs_'+str(magnitudes[h])+'prc'].values
            if pseudo_r_scores.any():
                top_predictors = np.argsort(pseudo_r_scores)[::-1][:2]
                ranges = param_bounds[top_predictors]
                # define grid of x (1st predictor), and y (2nd predictor) dimensions
                # to plot contour map over
                xgrid = np.arange(param_bounds[top_predictors[0]][0], 
                                  param_bounds[top_predictors[0]][1], np.around((ranges[0][1]-ranges[0][0])/100,decimals=4))
                ygrid = np.arange(param_bounds[top_predictors[1]][0], 
                                  param_bounds[top_predictors[1]][1], np.around((ranges[1][1]-ranges[1][0])/100,decimals=4))
                all_predictors = [ dta.columns.tolist()[k] for k in top_predictors]
                # fit logistic regression model with top two predictors of success and their interaction
                avg_dta['Interaction'] = avg_dta[all_predictors[0]]*dta[all_predictors[1]]
                result = fitLogit_interact(avg_dta, [all_predictors[k] for k in [0,1]])
                
                # plot success/failure for each SOW on top of logistic regression estimate of probability of success
                contourset = plotFactorMap(axes[i,j+1], result, avg_dta, probability_cmap, success_cmap, contour_levels, xgrid, ygrid, \
                              all_predictors[0], all_predictors[1])
                axes[i,j+1].set_title("Success if " + str(magnitudes[h]) + "% shortage\n<" + str((100-percentiles[i])) + "% of the time", fontsize=16)
                fig.savefig('Figure9_FactorMaps.pdf')
                
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contourset, cax=cbar_ax)
    cbar.ax.set_ylabel("Predicted Probability of Success", rotation=-90, va="bottom",fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig("Figure9_FactorMaps.pdf")
    fig.clf()
    
    return None

def highlight_cell(x,y, ax, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def addPercentileBlocks(historic_percents, gridcells, percentiles, ax):    
    for i in range(len(historic_percents)):
        if historic_percents[i] != 0:
            highlight_cell(i, gridcells[i], ax, color="orange", linewidth=2)
            if (9-gridcells[i])*10 in percentiles:
                highlight_cell(i-2, gridcells[i], ax, color="black", linewidth=2)
                highlight_cell(i+2, gridcells[i], ax, color="black", linewidth=2)
                
    return None

def plotFailureHeatmap(ax, design, ID, cbar=True):
    allSOWs, percentSOWs, historic_percents, magnitudes, frequencies, gridcells = calcFailureHeatmap(design, ID)
    
    im = ax.imshow(percentSOWs, norm = mpl.colors.Normalize(vmin=0.0,vmax=100.0), cmap='RdBu', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(magnitudes)))
    ax.set_xticklabels([str(x) for x in list(magnitudes)])
    ax.set_xlabel("Percent of demand that is short",fontsize=14)
    ax.set_ylabel("Percent of time\nshortage is experienced",fontsize=14)
    ax.set_yticks(np.arange(len(frequencies)))
    ax.set_yticklabels([str(x) for x in list(frequencies)])
        
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(percentSOWs.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(percentSOWs.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis='both',labelsize=14)
    
    if cbar == True:
        cbar = ax.figure.colorbar(im, ax=ax, cmap='RdBu')
        cbar.ax.set_ylabel("Percent of realizations\nin which criterion is met", rotation=-90, va="bottom",fontsize=14)
        cbar.ax.tick_params(labelsize=14)
    
    return allSOWs, historic_percents, frequencies, magnitudes, gridcells, im

def plotFactorMap(ax, result, dta, contour_cmap, dot_cmap, levels, xgrid, ygrid, xvar, yvar):
    
    xlabel = getLabels(xvar)
    ylabel = getLabels(yvar)
    
    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    if len(result.params) == 4:
        grid = np.column_stack([np.ones(len(x)),x,y,x*y])
    else:
        grid = np.column_stack([np.ones(len(x)),x,y])
 
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))

    contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap)
    ax.scatter(dta[xvar].values, dta[yvar].values, c=dta['Success'].values, edgecolor='none', cmap=dot_cmap, s=10)
    ax.set_xlim(np.nanmin(X),np.nanmax(X))
    ax.set_ylim(np.nanmin(Y),np.nanmax(Y))
    ax.set_xlabel(xlabel,fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)
    ax.tick_params(axis='both',labelsize=14)
    
    return contourset