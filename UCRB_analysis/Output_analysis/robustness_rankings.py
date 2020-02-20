import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
plt.ioff()

# =============================================================================
# Experiment set up
# =============================================================================
designs = ['LHsamples_original_1000_AnnQonly','LHsamples_wider_1000_AnnQonly',\
           'CMIPunscaled_SOWs','Paleo_SOWs']

all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)

allRobustness = np.zeros([10,10,nStructures,len(designs)])

for i, design in enumerate(designs):
	for j, ID in enumerate(all_IDs):
		allSOWs = np.load('../../../'+design+'/Factor_mapping/'+ ID + '_heatmap.npy')
		robustness = np.nanmean(allSOWs,2)
		allRobustness[:,:,j,i] = robustness

np.save('RobustnessRanks.npy',allRobustness)

