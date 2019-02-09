import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def CPlot(corr_mat,
          axis_labs,
          cmap = "PuBu",
          pad = 0.05,
          rad = 250,
          xlab = 'x',
          ylab = 'y',
          fs = 14):
    """
        This function generates correlation plot given correlation matrix (n, n).
        ------------------------
        
        Input:  corr_mat    - correlation matrix, np.ndarray of dim (n,n)
                axis_labels - custom labels for axis, list of length n
                cmap        - color map
                pad         - for separation of cbar from plot
                rad         - determines sircle radii
                
        Output: figure        
         
    """
    
    n = len(axis_labs)
    init_plotting() #invokes global plot parameters 
    xtick_pos = []
    ytick_pos = []
    c = []
    
    #separate matrix into 3 sep. arrays
    for i in range(n):
        x = i + 0.5   
        for j in range(n):
            y = j + 0.5  
            xtick_pos.append(x)
            ytick_pos.append(y)    
            c.append(corr_mat[i,j])
    
    #set tick labels
    plt.xticks(ytick_pos[0:n], axis_labs, rotation= 45)
    plt.yticks(ytick_pos[0:n], axis_labs)       
    
    #set axis labels
    plt.xlabel(xlab, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs)
    
    #set axis range
    plt.xlim((0, n))
    plt.ylim((0, n))
    
    norm = plt.Normalize(-1,1)
    plt.scatter(xtick_pos,ytick_pos, c = c, s = rad, norm=norm, cmap = cmap)
    cbar = plt.colorbar(pad=0.06)
    #cbar.outline.set_visible(False)
    plt.text(n + 0.75 , n + 0.75, r"$\overline{C}_{ij}$", fontdict = None, fontsize=fs)


def init_plotting():
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['axes.linewidth'] = 1.1
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.direction'] ='out'
    plt.rcParams['ytick.direction'] ='out'
    plt.rcParams['font.weight'] ='bold'
