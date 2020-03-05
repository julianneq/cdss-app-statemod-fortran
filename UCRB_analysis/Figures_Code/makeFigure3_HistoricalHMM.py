from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as ss
import utils

def makeFigure3_HistoricalHMM():
    AnnualQ = np.array(pd.read_csv('../Qgen/AnnualQ.csv'))*1233.48 # convert acre-ft to m^3
    logQ = np.log(AnnualQ[35::,-1]) # last 70 years of log-space flows at last node
    hidden_states, mus, sigmas, P = utils.fitHMM(logQ) # fit HMM
    
    # plot Gaussian fits
    combinedHistogram(logQ, mus, sigmas, P)
    plotTimeSeries(logQ, hidden_states)

    return None

def combinedHistogram(TransformedQ, mus, sigmas, P):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])
        
    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = pi[0]*ss.norm.pdf(x_1,mus[1],sigmas[1])
            
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(TransformedQ, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State')
    l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State')
    l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined')
    ax.set_xlabel('Log of annual flow in m' + r'$^3$')
    ax.set_ylabel('Probability Density')
    
    fig.subplots_adjust(bottom=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, fontsize=14)
    fig.savefig('Figure3a.pdf')
    fig.clf()
            
    return None

def plotTimeSeries(TransformedQ, hidden_states):
    
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xs = np.arange(len(TransformedQ))+1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], TransformedQ[masks], c='r', label='Dry State')
    masks = hidden_states == 1
    ax.scatter(xs[masks], TransformedQ[masks], c='b', label='Wet State')
    ax.plot(xs, TransformedQ, c='k')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Log of annual flow in m' + r'$^3$')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig('Figure3b.pdf')
    fig.clf()
    
    return None