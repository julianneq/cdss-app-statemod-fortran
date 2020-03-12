import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches
from scipy import stats
import seaborn as sns
from utils import setupProblem, getSamples

def makeFigure6_ShortageDistns():

    sns.set_style("white")
    
    designs = ['LHsamples_original_1000_AnnQonly','CMIPunscaled_SOWs','Paleo_SOWs','LHsamples_wider_1000_AnnQonly']
    nsamples = [1000,97,366,1000] # before removing those out of bounds
    titles = ['Box Around Historical','CMIP','Paleo','All-Encompassing']
    structures = ['53_ADC022','7200645']
    nrealizations = 10
    idx = np.arange(2,22,2)
    
    fig = plt.figure()
    count = 1 # subplot counter
    for structure in structures:
        # load historical shortage data and convert acre-ft to m^3
        hist_short = np.loadtxt('../Simulation_outputs/' + structure + '_info_hist.txt')[:,2]*1233.48
        # replace failed runs with np.nan (currently -999.9)
        hist_short[hist_short < 0] = np.nan
        for i, design in enumerate(designs):
            # find which samples are still in param_bounds after flipping misidentified wet and dry states
            param_bounds, param_names, params_no, problem = setupProblem(design)
            _, rows_to_keep = getSamples(design, params_no, param_bounds)
            nsamples[i] = len(rows_to_keep) # after removing those out of bounds after reclassification
            
            # load shortage data for this experimental design
            SYN_short = np.load('../../../Simulation_outputs/' + design + '/' + structure + '_info.npy')
            # remove columns for year (0) and demand (odd columns) and convert acre-ft to ^3
            SYN_short = SYN_short[:,idx,:]*1233.48
            SYN_short = SYN_short[:,:,rows_to_keep]
            # reshape into 12*nyears x nsamples*nrealizations
            SYN_short = SYN_short.reshape([np.shape(SYN_short)[0],np.shape(SYN_short)[1]*np.shape(SYN_short)[2]])
            # replace failed runs with np.nan (currently -999.9)
            SYN_short[SYN_short < 0] = np.nan
            
            # plot shortage distribution
            ax = fig.add_subplot(2,4,count)
            handles, labels = plotSDC(ax, SYN_short, hist_short, nsamples[i], nrealizations)
            
            # only put labels on bottom row/left column, make y ranges consistent, title experiment
            if count == 1 or count == 5:
                ax.tick_params(axis='y', labelsize=14)
            else:
                ax.tick_params(axis='y',labelleft='off')
                
            if count <= 4:
                ax.tick_params(axis='x',labelbottom='off')
                ax.set_title(titles[count-1],fontsize=16)
                ax.set_ylim(0,6200000)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
            else:
                ax.tick_params(axis='x',labelsize=14)
                ax.set_ylim(0,370000000)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(8,8))
                
            # iterature subplot counter
            count += 1
            
    fig.set_size_inches([16,8])
    fig.text(0.5, 0.15, 'Percentile', ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'Annual Shortage (m' + r'$^3$' + ')', va='center', rotation=90, fontsize=16)
    fig.subplots_adjust(bottom=0.22)
    labels_transposed = [labels[0],labels[6],labels[1],labels[7],labels[2],labels[8],labels[3],labels[9],labels[4],labels[10],labels[5]]
    handles_transposed = [handles[0],handles[6],handles[1],handles[7],handles[2],handles[8],handles[3],handles[9],handles[4],handles[10],handles[5]]
    legend = fig.legend(handles=handles_transposed, labels=labels_transposed, fontsize=16, loc='lower center', title='Frequency in experiment', ncol=6)
    plt.setp(legend.get_title(),fontsize=16)
    fig.savefig('Figure6_ShortageDistns.pdf')
    fig.clf()
    
    return None

def alpha(i, base=0.2):
    l = lambda x: x+base-x*base
    ar = [l(0)]
    for j in range(i):
        ar.append(l(ar[-1]))
    return ar[-1]
  
def plotSDC(ax, SYN_short, hist_short, nsamples, nrealizations):
    n = 12 # number of months
    #Reshape historic data to a [no. years x no. months] matrix
    f_hist = np.reshape(hist_short, (int(np.size(hist_short)/n), n))
    #Reshape to annual totals
    f_hist_totals = np.sum(f_hist,1)  
    #Calculate historical shortage duration curves
    F_hist = np.sort(f_hist_totals) # for inverse sorting add this at the end [::-1]
    
    #Reshape synthetic data
    #Create matrix of [no. years x no. months x no. samples]
    synthetic_global = np.zeros([int(np.size(hist_short)/n),n,nsamples*nrealizations])
    # Loop through every SOW and reshape to [no. years x no. months]
    for j in range(nsamples*nrealizations):
        synthetic_global[:,:,j]= np.reshape(SYN_short[:,j], (int(np.size(SYN_short[:,j])/n), n))
    #Reshape to annual totals
    synthetic_global_totals = np.sum(synthetic_global,1)
    
    p=np.arange(100,-10,-10)
    
    #Calculate synthetic shortage duration curves
    F_syn = np.empty([int(np.size(hist_short)/n),nsamples*nrealizations])
    F_syn[:] = np.NaN
    for j in range(nsamples*nrealizations):
        F_syn[:,j] = np.sort(synthetic_global_totals[:,j])
    
    # For each percentile of magnitude, calculate the percentile among the experiments run
    perc_scores = np.zeros_like(F_syn) 
    for m in range(int(np.size(hist_short)/n)):
        perc_scores[m,:] = [stats.percentileofscore(F_syn[m,:], j, 'rank') for j in F_syn[m,:]]
                
    P = np.arange(1.,len(F_hist)+1)*100 / len(F_hist)
    
    handles = []
    labels=[]
    color = '#000292'
    for i in range(len(p)):
        ax.fill_between(P, np.min(F_syn[:,:],1), np.percentile(F_syn[:,:], p[i], axis=1), color=color, alpha = 0.1)
        ax.plot(P, np.percentile(F_syn[:,:], p[i], axis=1), linewidth=0.5, color=color, alpha = 0.3)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color, alpha=alpha(i, base=0.1))
        handles.append(handle)
        label = "{:.0f} %".format(100-p[i])
        labels.append(label)
    ax.plot(P,F_hist, c='black', linewidth=2, label='Historical record')
    ax.set_xlim(1,100)
    
    return handles, labels