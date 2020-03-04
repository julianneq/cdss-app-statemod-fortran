import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils import Sobol_per_structure

sns.set_style("white")

designs = ['LHsamples_original_1000_AnnQonly','CMIPunscaled_SOWs','Paleo_SOWs','LHsamples_wider_1000_AnnQonly']
titles = ['Box Around Historical','CMIP','Paleo','All-Encompassing']
structures = ['53_ADC022','7200645']

colors = ["#de2d26", "#fb6a4a", "#3182bd", "#6baed6", "#a50f15", "#08519c", "#9e9ac8"]
mu0 = plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='none')
sigma0 = plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none')
mu1 = plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none')
sigma1 = plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none')
p00 = plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none')
p11 = plt.Rectangle((0,0), 1, 1, fc=colors[5], edgecolor='none')
Interact = plt.Rectangle((0,0), 1, 1, fc=colors[6], edgecolor='none')

def plotSums(df, ax):
    
    y1 = np.zeros([100])
    ymax = 1.0
    ymin = 0.0
    for k in range(len(colors)-1): # six 1st order SIs
        y2 = np.array(np.sum(df.iloc[0:(k+1),:])[1::])
        y2 = y2.astype(float)
        ax.plot(range(0,100),y2,c='None')
        ax.fill_between(range(0,100), y1, y2, color=colors[k])
        ymax = np.max([ymax,np.nanmax(y2)])
        y1 = y2
        
    y2 = np.ones([100])
    ZeroIndices = np.where(y1==0)
    y2[ZeroIndices] = 0
    negIndices = np.where(y1>1)
    y2[negIndices] = 1-y1[negIndices]
    ax.fill_between(range(0,100), y1, y2, where=y1<y2, color=colors[-1])
    ax.fill_between(range(0,100), y2, 0, where=y1>y2, color=colors[-1])
    ymax = max(ymax, np.nanmax(y2))
    ymin = min(ymin, np.nanmin(y2))
    ax.set_xlim([0,100])
    ax.set_ylim([ymin,ymax])
    ax.tick_params(axis='y',labelsize=14)

    return None

fig = plt.figure()
count = 1 # subplot counter
#for structure in structures:
#    for i, design in enumerate(designs):
        # perform variance decomposition
#        Sobol_per_structure(design, structure)
        
for structure in structures:
    for i, design in enumerate(designs):
        # load sensitivity indices
        S1_values = pd.read_csv('../Simulation_outputs/' + design + '/'+ structure + '_S1.csv')
        
        # plot shortage distribution
        ax = fig.add_subplot(2,4,count)
        plotSums(S1_values, ax)
        
        # only put labels on bottom row, title experiment
        if count <= 4:
            ax.tick_params(axis='x',labelbottom='off')
            ax.set_title(titles[count-1],fontsize=16)
        else:
            ax.tick_params(axis='x',labelsize=14)
            
        # iterature subplot counter
        count += 1

fig.set_size_inches([16,8])
fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.15, 'Percentile of Shortage', ha='center', fontsize=16)
fig.text(0.05, 0.5, 'Portion of Variance Explained', va='center', rotation=90, fontsize=16)
fig.legend([mu0,sigma0,mu1,sigma1,p00,p11,Interact],\
                  [r'$\mu_0$',r'$\sigma_0$',r'$\mu_1$',r'$\sigma_1$',r'$p_{00}$',r'$p_{11}$','Interactions'],\
                  loc='lower center', ncol=4, fontsize=16, frameon=True)
fig.savefig('Figure7_VarianceDecomposition.pdf')
fig.clf()
