import numpy as np
from matplotlib import pyplot as plt

# load data
historic_data = np.load('historic_flows.npy')
original_1000 = np.load('LHsamples_original_1000_AnnQonly_flows.npy')
#original_200 = np.load('LHsamples_original_200_AnnQonly_flows.npy')
#narrowed_1000 = np.load('LHsamples_narrowed_1000_AnnQonly_flows.npy')
#narrowed_200 = np.load('LHsamples_narrowed_200_AnnQonly_flows.npy')
wider_1000 = np.load('LHsamples_wider_1000_AnnQonly_flows.npy')
#wider_200 = np.load('LHsamples_wider_200_AnnQonly_flows.npy')
CMIP = np.load('CMIPunscaled_SOWs_flows.npy')
Paleo = np.load('Paleo_SOWs_flows.npy')

colors = ['#bebada','#b3de69','#fb8072','#ffffb3','#80b1d3']
labels = ['All-encompassing','Paleo','Box around Historical','CMIP','Historical']
data = [wider_1000, Paleo, original_1000, CMIP, historic_data]
        
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
for i in range(len(data)-1):
    ax.fill_between(range(12), np.min(np.min(data[i], axis=0),axis=0),
    	np.max(np.max(data[i], axis=0),axis=0),
    	color=colors[i], label=labels[i])
ax.fill_between(range(12), np.min(data[-1], axis=0),
                    np.max(data[-1], axis=0), color=colors[-1],
                    label=labels[-1])
ax.set_yscale("log")               
ax.set_xlabel('Month',fontsize=16)
ax.set_ylabel('Flow at Last Node (af)',fontsize=16)
ax.set_xlim([0,11])
ax.tick_params(axis='both',labelsize=14)
ax.set_xticks(range(12))
ax.set_xticklabels(['O','N','D','J','F','M','A','M','J','J','A','S'])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
fig.subplots_adjust(bottom=0.2)
fig.legend(handles, labels, fontsize=16,loc='lower center',ncol=3)
ax.set_title('Streamflow across experiments',fontsize=18)
fig.savefig('hydrographs_log.png')
fig.clf()

data = [np.sum(historic_data, axis=1),
np.sum(CMIP, axis=2), 
np.sum(original_1000, axis=2), 
np.sum(Paleo, axis=2), 
np.sum(wider_1000, axis=2)]
labels=['Historical','CMIP','Box around Historical','Paleo','All-encompassing']
colors = ['#80b1d3','#ffffb3','#fb8072','#b3de69','#bebada']

fig = plt.figure(figsize=(12,9))
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
ax.set_ylabel('Flow at Last Node (af)',fontsize=20)
ax.set_xticks(np.arange(1,6))
ax.set_xticklabels(labels,fontsize=16)
plt.savefig('streamflow_violinplot_log.png')