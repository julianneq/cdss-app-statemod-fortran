from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as ss
import utils

def fitHMM(TransformedQ):
    
    # fit HMM to all of data
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(TransformedQ,[len(TransformedQ),1]))
    hidden_states = model.predict(np.reshape(TransformedQ,[len(TransformedQ),1]))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
    
    logProb = model.score(np.reshape(TransformedQ,[len(TransformedQ),1]))
    #samples = model.sample(105)
    
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
    
    return hidden_states, mus, sigmas, P, logProb

def plotTimeSeries(TransformedQ, hidden_states, ylabel, filename):
    
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
    ax.set_ylabel(ylabel)
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()
    
    return None

def plotDistribution(TransformedQ, mus, sigmas, P, filename):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = pi[1]*ss.norm.pdf(x_1,mus[1],sigmas[1])
    
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])
            
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(TransformedQ, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State Distn')
    l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State Distn')
    l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined State Distn')
    ax.set_xlabel('log(Flow at state line (af))')
    ax.set_ylabel('Probability Density')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.savefig(filename)
    fig.clf()
            
    return None

def plotLogTimeSeries(TransformedQ, hidden_states, ylabel, filename):
    
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xs = np.arange(len(TransformedQ))+1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], np.exp(TransformedQ[masks]), c='r', label='Dry State')
    masks = hidden_states == 1
    ax.scatter(xs[masks], np.exp(TransformedQ[masks]), c='b', label='Wet State')
    ax.plot(xs, np.exp(TransformedQ), c='k')
    
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()
    
    return None

def plotLogDistribution(TransformedQ, mus, sigmas, P, filename):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])

    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = pi[1]*ss.norm.pdf(x_1,mus[1],sigmas[1])
    
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])

    hist, bins, _ = plt.hist(TransformedQ, density=True)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]),len(bins))
    weights = np.ones_like(TransformedQ)/float(len(TransformedQ))
            
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1, = ax.plot(np.exp(x_0), fx_0, c='r', linewidth=2, label='Dry State Distn')
    l2, = ax.plot(np.exp(x_1), fx_1, c='b', linewidth=2, label='Wet State Distn')
    l3, = ax.plot(np.exp(x), fx, c='k', linewidth=2, label='Combined State Distn')
    ax.hist(TransformedQ, bins=logbins, color='k', alpha=0.5, density=True, weights=weights)
    ax.set_ylabel('Probability',fontsize=16)
    ax.set_xlabel('Flow at state line (af)',fontsize=16)
    ax.set_xscale('log')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, fontsize=14)
    fig.savefig(filename)
    fig.clf()
            
    return None

def makePPandQQplot(TransformedQ, mus, sigmas, P, title, figname):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x_sorted = np.sort(TransformedQ)
    Fx_empirical = np.arange(1,len(TransformedQ)+1,1)/(len(TransformedQ)+1)
    Fx_fitted = pi[0]*ss.norm.cdf(x_sorted,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.cdf(x_sorted,mus[1],sigmas[1])
        
    sns.set()
    plt.scatter(Fx_empirical,Fx_fitted,color='b')
    plt.plot([0,1],[0,1],color='r')
    plt.xlabel('Empirical CDF')
    plt.ylabel('Fitted CDF')
    plt.title(title)
    plt.savefig(figname + '_PP.png')
    plt.clf()
    
    x_dist, qx_dist = utils.findQuantiles(mus, sigmas, pi)
    x_fitted = np.zeros(len(x_sorted))
    for i in range(len(x_fitted)):
        x_fitted[i] = x_dist[(np.abs(qx_dist-Fx_empirical[i])).argmin()]
        
    sns.set()
    plt.scatter(x_sorted,x_fitted,color='b')
    plt.plot([np.min(x_sorted),np.max(x_sorted)],[np.min(x_sorted),np.max(x_sorted)],color='r')
    plt.xlabel('Observations')
    plt.ylabel('Fitted Quantiles')
    plt.title(title)
    plt.savefig(figname + '_QQ.png')
    plt.clf()
    
    return None

def findQuantiles(mus, sigmas, piNew):
    x = np.empty([10000])
    qx = np.empty([10000])
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    qx = piNew[0]*ss.norm.cdf(x,mus[0],sigmas[0]) + \
        piNew[1]*ss.norm.cdf(x,mus[1],sigmas[1])
        
    return x, qx

def assessFit(logQ, hidden_states, mus, sigmas, P, year, dataset):
    
    sns.set()
    
    masks0 = hidden_states == 0
    masks1 = hidden_states == 1
    
    plt.hist(logQ[masks0],color='r',alpha=0.5,density=True)
    plt.hist(logQ[masks1],color='b',alpha=0.5,density=True)

    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = ss.norm.pdf(x_1,mus[1],sigmas[1])
    
    l1, = plt.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State Distn')
    l2, = plt.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State Distn')
    
    plt.title(year)
    plt.savefig(year + '_class' + dataset + '.png')
    plt.clf()
    
    x0_sorted = np.sort(logQ[masks0])
    p0_observed = np.arange(1,len(x0_sorted)+1,1)/(len(x0_sorted)+1)
    x0_fitted = ss.norm.ppf(p0_observed,mus[0],sigmas[0])
    
    x1_sorted = np.sort(logQ[masks1])
    p1_observed = np.arange(1,len(x1_sorted)+1,1)/(len(x1_sorted)+1)
    x1_fitted = ss.norm.ppf(p1_observed,mus[1],sigmas[1])
    
    minimum= np.min([np.min(logQ),np.min(x0_fitted),np.min(x1_fitted)])
    maximum = np.max([np.max(logQ),np.max(x0_fitted),np.max(x1_fitted)])
    
    plt.scatter(x0_sorted,x0_fitted,c='r')
    plt.scatter(x1_sorted,x1_fitted,c='b')
    plt.plot([minimum,maximum],[minimum,maximum],c='k')
    plt.savefig(year + '_class' + dataset + 'QQ.png')
    plt.clf()

    Pxt = np.zeros([len(logQ)+1,2])
    Pxt[0,:] = np.array([0.0,1.0])
    for t in range(len(logQ)):
        Pxt[t+1,:] = np.dot(Pxt[t,:],P)

    Eyt = np.zeros(len(logQ)+1)
    for t in range(len(Eyt)):
        Eyt[t] = Pxt[t,0]*mus[0] + Pxt[t,1]*mus[1]
        
    yt_bar = np.zeros(len(logQ))
    yt_bar[0] = logQ[0]
    for t in range(len(yt_bar)-1):
        yt_bar[t+1] = (yt_bar[t]*(t+1) + logQ[t+1]) / (t+2)
        
    plt.plot(yt_bar,c='r')
    plt.plot(Eyt,c='k')
    plt.title('Starting in wet state')
    plt.xlabel('Time (yrs)')
    plt.ylabel('E[y_t]')
    plt.savefig('StartWet_Ey_v_ybar.png')
    plt.clf()

    Pxt = np.zeros([len(logQ)+1,2])
    Pxt[0,:] = np.array([1.0,0.0])
    for t in range(len(logQ)):
        Pxt[t+1,:] = np.dot(Pxt[t,:],P)

    Eyt = np.zeros(len(logQ)+1)
    for t in range(len(Eyt)):
        Eyt[t] = Pxt[t,0]*mus[0] + Pxt[t,1]*mus[1]
        
    yt_bar = np.zeros(len(logQ))
    yt_bar[0] = logQ[0]
    for t in range(len(yt_bar)-1):
        yt_bar[t+1] = (yt_bar[t]*(t+1) + logQ[t+1]) / (t+2)
        
    plt.plot(yt_bar,c='r')
    plt.plot(Eyt,c='k')
    plt.title('Starting in dry state')
    plt.xlabel('Time (yrs)')
    plt.ylabel('E[y_t]')
    plt.savefig('StartDry_Ey_v_ybar.png')
    plt.clf()
    
    return None