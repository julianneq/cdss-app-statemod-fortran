import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os
from mpi4py import MPI
import math
import sys
plt.ioff()

design = str(sys.argv[1])


all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
#select_IDs = ['53_ADC022','7000550','7200645','7200799','7202003','3600687','3704614',]
nStructures = len(all_IDs)
#nStructures = len(select_IDs)
# Longform parameter names to use in figure legend
parameter_names_long = ['Dry state mu', 'Dry state sigma', 'Wet state mu', 
                        'Wet state sigma', 'Dry-to-dry probability', 
                        'Wet-to-wet probability', 'Interactions']
param_names=['XBM_mu0','XBM_sigma0', 'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11']

def plotSums(df, variable, colors, filename):
    
    mu0 = plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='none')
    sigma0 = plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none')
    mu1 = plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none')
    sigma1 = plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none')
    p00 = plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none')
    p11 = plt.Rectangle((0,0), 1, 1, fc=colors[5], edgecolor='none')
    Interact = plt.Rectangle((0,0), 1, 1, fc=colors[6], edgecolor='none')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    y1 = np.zeros([100])
    ymax = 1.0
    ymin = 0.0
    for k in range(len(param_names)): # six 1st order SIs
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
    ax.set_xlabel('Percentile of Shortage',fontsize=16)
    ax.set_ylabel('Portion of ' + variable,fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    fig.subplots_adjust(bottom=0.25)
    fig.legend([mu0,sigma0,mu1,sigma1,p00,p11,Interact],\
                  [r'$\mu_0$',r'$\sigma_0$',r'$\mu_1$',r'$\sigma_1$',r'$p_00$',r'$p_11$','Interactions'],\
                  loc='lower center', ncol=4, fontsize=16, frameon=True)
    fig.set_size_inches([7.75,6.5])
    fig.savefig(filename)
    fig.clf()

    return None
  
def plotVarianceDecomposition(structure_name):   
    '''
    Sensitivity analysis plots
    '''
    colors = ["#de2d26", "#fb6a4a", "#3182bd", "#6baed6", "#a50f15", "#08519c", "#9e9ac8"]
    
    Delta_values = pd.read_csv('../../../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
    S1_values = pd.read_csv('../../../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
    R2_values = pd.read_csv('../../../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
    
    titles = ["Delta","S1","R2"]
    values_to_plot = [Delta_values, S1_values, R2_values]
    variables = ['Density','Variance','Variance']
    for k in range(len(titles)):
        filename = '../../../'+design+'/VarianceDecomposition/' + structure_name + '_'+titles[k]+'.png'
        plotSums(values_to_plot[k], variables[k], colors, filename)

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
    
for i in range(start, stop):
   plotVarianceDecomposition(all_IDs[i])

#for i in range(len(all_IDs)):
#    #plotVarianceDecomposition(select_IDs[i])
#    plotVarianceDecomposition(all_IDs[i])
