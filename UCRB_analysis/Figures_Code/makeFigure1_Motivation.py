import numpy as np
import matplotlib.pyplot as plt
from SALib.analyze import delta
import matplotlib as mpl
import seaborn as sns
from scipy import stats as ss

def makeFigure1_Motivation():
    # generate fake response data vs. two uncertainties for two policies
    param_bounds = np.array([[-1,1],[0,2]])
    # policy 1 vs. policy 2
    result1, xa, ya, z1 = calcSindices(param_bounds, 1)
    result2, xa, ya, z2 = calcSindices(param_bounds, 2)
    
    # repeat with different ranges of uncertainties
    param_bounds = np.array([[-1,2],[0,3]])
    # policy 1 vs. policy 2
    result1b, xb, yb, z1b = calcSindices(param_bounds, 1)
    result2b, xb, yb, z2b = calcSindices(param_bounds, 2)
    
    # generate 2 sets of MVN points in each region
    np.random.seed(7)
    mu1 = [0,1]
    cov1 = [[0.2,-0.7*0.2],[-0.7*0.2,0.2]]
    rvs1 = ss.multivariate_normal.rvs(mu1,cov1,20)
    
    np.random.seed(7)
    mu2 = [0.5,1.5]
    cov2 = [[0.3,0.7*0.3],[0.7*0.3,0.3]]
    rvs2 = ss.multivariate_normal.rvs(mu2,cov2,20)
    
    
    # plotting parameters
    contour_cmap = mpl.colors.ListedColormap(np.array([[228,26,28],[55,126,184]])/255.0)
    vmin = np.min([np.min(z1),np.min(z1b),np.min(z2),np.min(z2b)])
    vmax = np.max([np.max(z1),np.max(z1b),np.max(z2),np.max(z2b)])
    contour_levels = [vmin, 0.0, vmax]
    
    # make plot
    sns.set_style("dark")
    fig, axes = plt.subplots(3,4,figsize=(13.7,9.5))
    
    # plot failure response surfaces
    pBase, p1 = plotResponseSurface(axes[0,0], contour_cmap, xa, ya, z1, rvs1, contour_levels, 'Policy 1', '#006d2c', ylabel=True)
    pBase, p1 = plotResponseSurface(axes[0,1], contour_cmap, xa, ya, z2, rvs1, contour_levels, 'Policy 2', '#984ea3')
    pBase, p1 = plotResponseSurface(axes[0,2], contour_cmap, xb, yb, z1b, rvs1, contour_levels, 'Policy 1', '#006d2c')
    plotResponseSurface(axes[0,3], contour_cmap, xb, yb, z2b, rvs1, contour_levels, 'Policy 2', '#984ea3')
    # make box for smaller region within larger region
    axes[0,2].plot([1,1],[0,2],c='k')
    axes[0,2].plot([-1,1],[2,2],c='k')
    axes[0,3].plot([1,1],[0,2],c='k')
    axes[0,3].plot([-1,1],[2,2],c='k')
    
    # plot fake CMIP samples in each region
    p2 = axes[0,2].scatter(rvs2[:,0],rvs2[:,1],c='#ffffb3')
    axes[0,3].scatter(rvs2[:,0],rvs2[:,1],c='#ffffb3')
    
    # plot probability distributions for each policy in region 1
    prob1 = computePDF(mu1, cov1, 1)
    prob2 = computePDF(mu1, cov1, 2)
    KDEplot(axes[1,0], prob1, prob2, 1)
    
    # probability of success bar chart for region 1
    probabilityChart(axes[1,1], prob1, prob2)
    
    # plot probability distributions for each policy in region 2
    prob1 = computeMixedPDF(mu1, cov1, mu2, cov2, 1)
    prob2 = computeMixedPDF(mu1, cov1, mu2, cov2, 2)
    KDEplot(axes[1,2], prob1, prob2, 2)
    
    # probability of success bar chart for region 2
    probabilityChart(axes[1,3], prob1, prob2)
    
    # make bar chart of sensitivities
    variables = ['Precipitation','Temperature','Interactions']
    sensitivityChart(axes[2,0], variables, result1, ylabel=True)
    sensitivityChart(axes[2,1], variables, result2)
    sensitivityChart(axes[2,2], variables, result1b)
    sensitivityChart(axes[2,3], variables, result2b)
    
    # add legend
    fig.subplots_adjust(wspace=0.4,hspace=0.3,bottom=0.13)
    B1 = plt.Rectangle((0, 0), 1, 1, fc='#377eb8', edgecolor='none') # success color
    B2 = plt.Rectangle((0, 0), 1, 1, fc='#e41a1c', edgecolor='none') # failure color
    fig.legend([pBase, p1,p2,B1,B2], ['Historical','Paleo','CMIP','Success','Failure'], ncol=5, loc='lower center',fontsize=16)
    fig.savefig('Figure1_Motivation.pdf')
    fig.clf()
    
    return None

def calcSindices(param_bounds, policy):
    x1 = np.linspace(param_bounds[0,0],param_bounds[0,1],100)
    y1 = np.linspace(param_bounds[1,0],param_bounds[1,1],100)
    xv, yv = np.meshgrid(x1,y1)
    x = xv.flatten()
    y = yv.flatten()
    if policy == 1:
        z = (x+1) + -4*y + (x+1)*y + 1
    else:
        z = (x+1) + -3*y + (x+1)*y
    
    param_names=['x','y']
    params_no = len(param_names)
    problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
    }
    samples = np.transpose(np.array([x,y]))
    
    result = delta.analyze(problem, samples, z, print_to_console=False, num_resamples=2)
    
    return result, x1, y1, z

def plotResponseSurface(ax, contour_cmap, xgrid, ygrid, z, rvs, contour_levels, title, titlecolor, ylabel=False):
    
    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.reshape(z, np.shape(X))
    
    ax.contourf(X, Y, Z, contour_levels, cmap=contour_cmap)
        
    ax.set_xlim(np.nanmin(X),np.nanmax(X))
    ax.set_ylim(np.nanmin(Y),np.nanmax(Y))
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xlabel(r'$\Delta$' + ' Precipitation', fontsize=16)
    pBase = ax.scatter(0,0,c='k',s=100)
    ax.set_title(title,color=titlecolor, fontsize=16)
    if ylabel == True:
        ax.set_ylabel(r'$\Delta$' + ' Temperature', fontsize=16)
        
    # plot fake Paleo samples
    p1 = ax.scatter(rvs[:,0],rvs[:,1],c='#b3de69')
    
    return pBase, p1

def sensitivityChart(ax, variables, result, ylabel=False):
    S_indices = np.array([result['S1'][0],result['S1'][1],1-np.sum(result['S1'])])
    ax.barh(np.arange(len(variables)),S_indices,color='k')
    ax.set_xlim([0,1])
    if ylabel == True:
        ax.set_yticks(np.arange(len(variables)))
        ax.set_yticklabels(variables,fontsize=16)
        ax.invert_yaxis()
    else:
        ax.set_yticks(np.arange(len(variables)))
        ax.set_yticklabels('')
        ax.invert_yaxis()
        
    ax.tick_params(axis='x',labelsize=14)
    ax.set_xlabel('Variance Explained',fontsize=16)
    
    return None

def computePDF(mu, cov, policy):
    rvs = ss.multivariate_normal.rvs(mu,cov,1000)
    x = rvs[:,0]
    y = rvs[:,1]
    if policy == 1:
        z = (x+1) + -4*y + (x+1)*y + 1
    else:
        z = (x+1) + -3*y + (x+1)*y
    
    return z

def computeMixedPDF(mu1, cov1, mu2, cov2, policy):
    np.random.seed(7)
    rvs1 = ss.multivariate_normal.rvs(mu1,cov1,1000)
    np.random.seed(7)
    rvs2 = ss.multivariate_normal.rvs(mu2,cov2,1000)
    x = np.concatenate((rvs1[:,0],rvs2[:,0]),0)
    y = np.concatenate((rvs1[:,1],rvs2[:,1]),0)
    if policy == 1:
        z = (x+1) + -4*y + (x+1)*y + 1
    else:
        z = (x+1) + -3*y + (x+1)*y
    
    return z

def KDEplot(ax, prob1, prob2, region):
    sns.kdeplot(prob1,color='#006d2c',ax=ax)
    sns.kdeplot(prob2,color='#984ea3',ax=ax)
    xmin = np.min([np.min(prob1),np.min(prob2)])
    xmax = np.max([np.max(prob1),np.max(prob2)])
    if region == 1:
        ymax = 0.25
    else:
        ymax = 0.35
    
    #ax.plot([0,0],[0,ymax],c='k')
    ax.set_ylim([0,ymax])
    ax.set_xlim([xmin,xmax])
    ax.set_xticklabels('')
    ax.set_xlabel('Reliability',fontsize=16)
    ax.tick_params(axis='y',labelsize=16)
    ax.fill_between([xmin,0],[0,0],[ymax,ymax],color='#fb9a99')
    ax.fill_between([0,xmax],[0,0],[ymax,ymax],color='#a6cee3')
    ax.set_ylabel('Probability Density',fontsize=16)
    
    return None

def probabilityChart(ax, prob1, prob2):
    ax.bar([0,1], [len(prob1[prob1>0])/len(prob1),len(prob2[prob2>0])/len(prob2)], color=['#006d2c','#984ea3'])
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Policy 1','Policy 2'],fontsize=16)
    ax.tick_params(axis='y',labelsize=16)
    ax.set_ylabel('P(Success)',fontsize=16)

    return None