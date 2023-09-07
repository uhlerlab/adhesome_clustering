# Import standard libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
import csv
import pandas as pd
import networkx as nx
import community
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import time
from tqdm import tqdm
import random
import subprocess
import utils as lu


def format_node_label(raw_label):
    '''
    Function to put node label in a square format
    Args:
        raw_label: (Numpy array) array of gene names
    Returns:
        The formatted label (a string)
    '''
    label = raw_label.copy()   
    # Line breaks
    for i in range(1, len(label)//2+1):
        if 2*i<len(label):
            label[2*i-1] = label[2*i-1]+' \n '
    label = ' '.join(label)
    return label


def from_nx(graph, pyvisnet, default_node_size=1, 
            default_edge_weight=1, edge_weight_scale=1, edge_color_scale=1, edge_threshold=0, hidden_edges=[], 
            shape='circle'):
    """
    This method takes an exisitng Networkx graph and translates
    it to a PyVis graph format that can be accepted by the VisJs
    API in the Jinja2 template. This operation is done in place.
    """
    nx_graph = graph.copy()
    assert(isinstance(nx_graph, nx.Graph))
    edges = nx_graph.edges(data=True)
    nodes = nx_graph.nodes(data=True)

    if len(edges) > 0:
        for e in edges:
            # Specify node size
            if 'size' not in nodes[e[0]].keys():
                nodes[e[0]]['size'] = default_node_size
            nodes[e[0]]['size'] = int(nodes[e[0]]['size'])
            if 'size' not in nodes[e[1]].keys():
                nodes[e[1]]['size'] = default_node_size
            nodes[e[1]]['size'] = int(nodes[e[1]]['size'])
            # Specify node color
            if nodes[e[0]]['n_TFs']>0:
                nodes[e[0]]['color'] = 'lightcoral'
            else:
                nodes[e[0]]['color'] = 'lightcoral'
            if nodes[e[1]]['n_TFs']>0:
                nodes[e[1]]['color'] = 'lightcoral'
            else:
                nodes[e[1]]['color'] = 'lightcoral'
            # Specify node title
            if len(nodes[e[0]]['all_TFs'])>0:
                nodes[e[0]]['title'] = 'adhesome: '+','.join(nodes[e[0]]['all_adhesomes'])+' / '+'TFs: '+','.join(nodes[e[0]]['all_TFs'])
            else:
                nodes[e[0]]['title'] = 'adhesome: '+','.join(nodes[e[0]]['all_adhesomes'])+' / '+'no TF'
            if nodes[e[1]]['n_TFs']>0:
                nodes[e[1]]['title'] = 'adhesome: '+','.join(nodes[e[1]]['all_adhesomes'])+' / '+'TFs: '+','.join(nodes[e[1]]['all_TFs'])
            else:
                nodes[e[1]]['title'] = 'adhesome: '+','.join(nodes[e[1]]['all_adhesomes'])+' / '+'no TF'
            pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
            pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)

            if 'weight' not in e[2].keys():
                e[2]['weight'] = default_edge_weight
            edge_dict = e[2].copy()
            edge_dict["value"] = e[2]['weight']*edge_weight_scale
            edge_dict["title"] = 'TF: '+e[2]['TFs']+'\n'+'val: '+str(e[2]['weight'])
            edge_dict["label"] = e[2]['TFs']
            edge_dict["color"] = colfunc(e[2]['weight']*edge_color_scale)
            edge_dict["arrowStrikethrough"] = False
            if (e[2]['weight']>edge_threshold) and (e[2]['TFs'] not in hidden_edges):
                pyvisnet.add_edge(e[0], e[1], **edge_dict)

    for node in nx.isolates(nx_graph):
        if 'size' not in nodes[node].keys():
            nodes[node]['size']=default_node_size
            nodes[node]['color'] = 'lightcoral'
        pyvisnet.add_node(node, **nodes[node], shape=shape)
    
    
def colfunc(val, minval=0, maxval=1):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    RED, YELLOW, GREEN  = (1, 0, 0), (1, 1, 0), (0, 1, 0)
    CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
    WHITE = (1, 1, 1)
    f = float(val-minval) / (maxval-minval)
    return mpl.colors.rgb2hex(tuple(f*(b-a)+a for (a, b) in zip(WHITE, RED)))
    
    
def tfnx_from_nx(graph, pyvisnet, 
                 default_node_size=1, default_edge_weight=1, 
                 shape='circle'):
    """
    This method takes an exisitng Networkx graph and translates
    it to a PyVis graph format that can be accepted by the VisJs
    API in the Jinja2 template. This operation is done in place.
    """
    nx_graph = graph.copy()
    assert(isinstance(nx_graph, nx.Graph))
    edges = nx_graph.edges(data=True)
    nodes = nx_graph.nodes(data=True)

    if len(edges) > 0:
        for e in edges:
            # Specify node size
            if 'size' not in nodes[e[0]].keys():
                nodes[e[0]]['size'] = default_node_size
            nodes[e[0]]['size'] = int(nodes[e[0]]['size'])
            if nodes[e[0]]['significant_tf']==1:
                nodes[e[0]]['color'] = 'lightcoral'
            else:
                nodes[e[0]]['color'] = 'dodgerblue'
            if 'size' not in nodes[e[1]].keys():
                nodes[e[1]]['size'] = default_node_size
            nodes[e[1]]['size'] = int(nodes[e[1]]['size'])
            if nodes[e[1]]['significant_tf']==1:
                nodes[e[1]]['color'] = 'lightcoral'
            else:
                nodes[e[1]]['color'] = 'dodgerblue'
            pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
            pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)
            # Specify edge characteristics
            e[2]['weight'] = default_edge_weight
            e[2]['color'] = '#cbcff5'
            if e[0]==e[1]:
                e[2]['weight'] = default_edge_weight
                e[2]['color'] = '#fa0202'
            edge_dict = e[2].copy()
            edge_dict["width"] = e[2]['weight']
            edge_dict["color"] = e[2]['color']
            edge_dict["arrowStrikethrough"] = False
            pyvisnet.add_edge(e[0], e[1], **edge_dict)

    for node in nx.isolates(nx_graph):
        if 'size' not in nodes[node].keys():
            nodes[node]['size']=default_node_size
        if nodes[node]['significant_tf']==1:
            nodes[node]['color'] = 'lightcoral'
        else:
            nodes[node]['color'] = 'dodgerblue'
        pyvisnet.add_node(node, **nodes[node], shape=shape)
    
    
    
    
    
    