from matplotlib import pyplot as plt
import seaborn as sns
from makeFigure9_FactorMaps import plotFailureHeatmap, highlight_cell

def makeFigureS12_SatisficingSurfaces():
    
    designs = ['LHsamples_original_1000_AnnQonly','CMIPunscaled_SOWs','Paleo_SOWs','LHsamples_wider_1000_AnnQonly']
    titles = ['Box Around Historical','CMIP','Paleo','All-Encompassing']
    structures = ['53_ADC022','7200645']
    
    sns.set_style("dark")
    
    fig = plt.figure()
    
    for i,structure in enumerate(structures):
        for j,design in enumerate(designs):
            ax = fig.add_subplot(2,4,i*4+j+1)
            allSOWs, historic_percents, frequencies, magnitudes, gridcells, im = plotFailureHeatmap(ax, design, structure, False)
            for k in range(len(historic_percents)):
                if historic_percents[k] != 0: # highlight historical frequencies at each magnitude in orange
                    highlight_cell(k, gridcells[k], ax, color="orange", linewidth=2)
            
            if i == 0:
                ax.set_title(titles[j])

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Predicted Probability of Success", rotation=-90, va="bottom",fontsize=16)
    fig.subplots_adjust(right=0.8,wspace=0.5)
    fig.set_size_inches([18.4,8.5])
    fig.savefig("FigureS12_SatisficingSurface.pdf")
    fig.clf()
    
    return None
