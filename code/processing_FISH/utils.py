# Import standard libraries
import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
import networkx as nx
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import time
from tqdm import tqdm


def unnesting(df, explode):
    """ Helper function to explode a dataframe based on a given column
    Args:
        df: (pandas DataFrame) the datframe to explode
        explode: (list of String) list of columns to explode on
    
    Returns:
        Exploded dataframe
    """
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    return df1.join(df.drop(labels=explode, axis=1), how='left')


def plot_heatmap(df, xticklabels, yticklabels, xlabel, ylabel, size, vmax, vmin=0, fontsize=3, save_to='', add_patches=[], cmap='rocket'):
    '''
    Plots the heatmap of the input dataframe.
    Args:
        df: (pandas DataFrame) dataframe to plot
        xticklabels: (Numpy array) labels of x-ticks
        ticklabels: (Numpy array) labels of y-ticks
        xlabel: (String) label of x-axis
        ylabel: (String) label of y-axis
        size: (int) size of figure
        vmax: (float) color bar upper limit
        save_to: (String) name of figure for saving
        add_patches: (list) list of patches to add on the image
    Returns:
        void
    '''
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111)
    im = ax.imshow(df, origin='upper', interpolation='none', cmap=cmap)
    # Place ticks at the middle of every pixel
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_ylim(len(yticklabels)-0.5, -0.5)
    # Use input dataframe row and column names as ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    # Put x axis labels on top
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    # Define axis name
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add patches to the axis
    if len(add_patches)>0:
        for patch in add_patches:
            rect = patches.Rectangle((patch[0], patch[1]), patch[2], patch[3], 
                                     linewidth=2, edgecolor=patch[4], facecolor='none')
            ax.add_patch(rect)
    
    # Add colorbar
    fig.colorbar(im, fraction=0.046, pad=0.04)
    im.set_clim(vmin, vmax)
    if save_to != '':
        # Save plot
        plt.savefig(save_to+'.pdf', format='pdf')
    plt.show()




