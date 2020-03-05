import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils import calcFailureHeatmap
from makeFigure6_ShortageDistns import plotSDC
from makeFigure7_VarianceDecomposition import plotSums

def makeFigure8_FactorMaps():

    sns.set_style("white")
    
    design = 'LHsamples_wider_1000_AnnQonly'
    nsamples = 1000
    structure = '53_ADC022'
    idx = np.arange(2,22,2)
    percentiles = [30, 50, 90]
    nrealizations = 10
    colors = ["#de2d26", "#fb6a4a", "#3182bd", "#6baed6", "#a50f15", "#08519c", "#9e9ac8"]
    
    # load historical shortage data and convert acre-ft to m^3
    histData = np.loadtxt('../Simulation_outputs/' + structure + '_info_hist.txt')[:,2]*1233.48
    # replace failed runs with np.nan (currently -999.9)
    histData[histData < 0] = np.nan
    
    # load shortage data for this experimental design
    synthetic = np.load('../../../Simulation_outputs/' + design + '/' + structure + '_info.npy')
    # remove columns for year (0) and demand (odd columns) and convert acre-ft to m^3
    synthetic = synthetic[:,idx,:]*1233.48
    # reshape into 12*nyears x nsamples*nrealizations
    synthetic = synthetic.reshape([np.shape(synthetic)[0],np.shape(synthetic)[1]*np.shape(synthetic)[2]])
    # replace failed runs with np.nan (currently -999.9)
    synthetic[synthetic < 0] = np.nan
    
    fig = plt.figure()
    fig.set_size_inches([19.2,9.5])
    fig.subplots_adjust(hspace=0.3)
    # plot shortage distribution for this structure under all-encompassing experiment
    ax = fig.add_subplot(3,4,1)
    handles, labels = plotSDC(ax, synthetic, histData, nsamples, nrealizations)
    ax.set_ylim([0,6200000])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
    ax.tick_params(axis='y',labelsize=14)
    ax.tick_params(axis='x',labelbottom='off')
    ax.set_ylabel('Shortage (m' + r'$^3$' + ')',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax.plot([percentile, percentile],[0,6200000],c='k')
    
    # plot variance decomposition for this structure under all-encompassing experiment
    ax = fig.add_subplot(3,4,5)
    S1_values = pd.read_csv('../Simulation_outputs/' + design + '/'+ structure + '_S1.csv')
    plotSums(S1_values, ax, colors)
    ax.set_ylim([0,1])
    ax.tick_params(axis='both',labelsize=14)
    ax.set_ylabel('Portion of Variance',fontsize=14)
    ax.set_xlabel('Shortage Percentile',fontsize=14)
    # add lines at percentiles
    for percentile in percentiles:
        ax.plot([percentile, percentile],[0,1],c='k')
    
    # plotfailure heatmap for this structure under all-encompassing experiment
    ax = fig.add_subplot(3,4,9)
    plotFailureHeatmap(ax, design, structure, percentiles)
    
    for i in range(len(percentiles)):
        ax = fig.add_subplot(3,4,i+2)
        plotResponseSurface(otherSOWs=False)
        
        ax = fig.add_subplot(3,4,i+6)
        plotResponseSurface(otherSOWs=True)
        
        ax = fig.add_subplot(3,4,i+6)
        plotFactorMap()
    
    return None

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def plotFailureHeatmap(ax, design, ID, percentiles):
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
    
    cbar = ax.figure.colorbar(im, ax=ax, cmap='RdBu')
    cbar.ax.set_ylabel("Percent of realizations\nin which criterion is met", rotation=-90, va="bottom",fontsize=14)
    
    for i in range(len(historic_percents)):
        if historic_percents[i] != 0:
            if (9-gridcells[i])*10 in percentiles:
                highlight_cell(i,gridcells[i], color="black", linewidth=2)
            else:
                highlight_cell(i,gridcells[i], color="orange", linewidth=2)
    
    return allSOWs, historic_percents

def plotResponseSurface(otherSOWs):
    
    return None

def plotFactorMap():
    
    return None
