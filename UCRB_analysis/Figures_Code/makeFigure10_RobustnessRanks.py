import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats as ss
from makeFigure9_FactorMaps import plotFailureHeatmap, highlight_cell

def makeFigure10_RobustnessRanks():

    allRobustness = np.load('../Output_analysis/RobustnessRanks.npy')
    nstructures = np.shape(allRobustness)[2]
    colors=['#fb8072','#ffffb3','#b3de69','#bebada'] # original, CMIP, paleo, wider
    colOrder = [0,2,3,1] # order of columns to plot
    labels = ['Box Around Historical','CMIP','Paleo','All Encompassing']
    design = 'LHsamples_wider_1000_AnnQonly'
    structure = '53_ADC022'
            
    #sns.set_style("dark")
    mags = np.array([10,50,100])
    freqs = np.array([50,100])
    
    sns.set_style("dark")
    
    fig = plt.figure()
    
    for i in range(len(mags)):
        ax = fig.add_subplot(2,3,i+1)
        # sort structures in all-encompassing design
        sortedIndices = np.argsort(-allRobustness[int(mags[i]/10)-1,0,:,1]) # freq=10% of the time
        for k in range(4): # loop through experimental designs
            ax.scatter(range(1,nstructures+1),\
                       ss.rankdata(-allRobustness[int(mags[i]/10)-1,0,:,colOrder[k]])[sortedIndices],\
                           color=colors[k],label=labels[k])
    
        if i == 2:
            ax.set_xlabel('Robustness Rank in\n"All Encompassing" Experiment',fontsize=16)
        elif i == 0:
            ax.set_ylabel('Robustness Rank in\neach Experiment',fontsize=16)
            
        ax.tick_params(axis='both',labelsize=14)
        ax.set_title('% of SOWs with ' + str(mags[i]) + '% shortage\n<10% of the time',fontsize=18)
     
    for j in range(len(freqs)):
        ax = fig.add_subplot(2,3,j+4)
        # sort structures in all-encompassing design
        sortedIndices = np.argsort(-allRobustness[0,int(freqs[j]/10)-1,:,1]) # mag=10% of demand
        for k in range(4): # loop through experimental designs
            ax.scatter(range(1,nstructures+1),\
                       ss.rankdata(-allRobustness[0,int(freqs[j]/10)-1,:,colOrder[k]])[sortedIndices],\
                           color=colors[k],label=labels[k])
    
        if j == 0:
            ax.set_ylabel('Robustness Rank in\neach Experiment',fontsize=16)
            
        ax.tick_params(axis='both',labelsize=14)
        ax.set_title('% of SOWs with 10% shortage\n<' + str(freqs[j]) + '% of the time',fontsize=18)
        ax.set_xlabel('Robustness Rank in\n"All Encompassing" Experiment',fontsize=16)
    
    handles, labels = ax.get_legend_handles_labels()
        
    ax = fig.add_subplot(2,3,6)
    allSOWs, historic_percents, frequencies, magnitudes, gridcells, im = plotFailureHeatmap(ax, design, structure)
    for i in range(len(historic_percents)):
        if historic_percents[i] != 0: # highlight historical frequencies at each magnitude in orange
            highlight_cell(i ,gridcells[i], color="orange", linewidth=2)
    # highlight criteria in ranks in black
    for i in range(len(mags)):
        highlight_cell(0, int(mags[i]/10)-1, color='black', linewidth=2)
    for i in range(len(freqs)):
        highlight_cell(int(freqs[i]/10)-1, 0, color='black', linewidth=2)
    
    fig.set_size_inches([19.2,9.6])
    fig.subplots_adjust(bottom=0.2,wspace=0.25,hspace=0.35)
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=16)
    fig.savefig('Figure10_RobustnessRanks.pdf')
    fig.clf()
    
    return None