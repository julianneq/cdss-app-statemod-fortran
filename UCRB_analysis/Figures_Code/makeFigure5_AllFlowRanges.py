import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# load data and convert acre-ft to m^3
historic_data = np.load('../Qgen/historic_flows.npy')*1233.48
original_1000 = np.load('../Qgen/LHsamples_original_1000_AnnQonly_flows.npy')*1233.48
wider_1000 = np.load('../Qgen/LHsamples_wider_1000_AnnQonly_flows.npy')*1233.48
CMIP = np.load('../Qgen/CMIPunscaled_SOWs_flows.npy')*1233.48
Paleo = np.load('../Qgen/Paleo_SOWs_flows.npy')*1233.48

# compute annual sums
data = [np.sum(historic_data, axis=1),
np.sum(CMIP, axis=2), 
np.sum(original_1000, axis=2), 
np.sum(Paleo, axis=2), 
np.sum(wider_1000, axis=2)]

# plotting characteristics
labels=['Historical','CMIP','Box\naround\nHistorical','Paleo','All\nEncompassing']
colors = ['#80b1d3','#ffffb3','#fb8072','#b3de69','#bebada']

sns.set_style("darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111)
violinplots=ax.violinplot(data, vert=True)
violinplots['cbars'].set_edgecolor('black')
violinplots['cmins'].set_edgecolor('black')
violinplots['cmaxes'].set_edgecolor('black')
for i in range(len(violinplots['bodies'])):
    vp = violinplots['bodies'][i]
    vp.set_facecolor(colors[i])
    vp.set_edgecolor('black')
    vp.set_alpha(1)
ax.set_yscale( "log" )
ax.tick_params(axis='both',labelsize=14)
ax.set_xticks(np.arange(1,6))
ax.set_xticklabels(labels,fontsize=16)

fig.tight_layout()
fig.savefig('Figure5_AllFlowRanges.pdf')
fig.clf()