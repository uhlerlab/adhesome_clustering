# Import standard libraries
import sys, getopt
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import sys
import math
import pybedtools
import hicstraw
from tqdm.notebook import tqdm

# Import custom libraries
sys.path.append("..")
import process_hic_contacts_inter as phc

def parse_config(config_filename):
    '''
    Reads config file
    Args:
        config_filename: (string) configuration file name
    Returns:
        A dictionary specifying the main directories, cell type, resolution, quality, chromosomes
    '''
    config = json.load(open(config_filename))
    return(config)

def main():
    
    # Parse command line arguments
    argv = sys.argv[1:]
    try:
        options, args = getopt.getopt(argv, "c:", ["config="])
    except:
        print('Incorrect arguments!')
        
    for name, value in options:
        if name in ('-c', '--config'):
            config_filename = value
    
    # Parse config file
    config = parse_config(config_filename)
    genome_dir = config['GENOME_DIR']
    arrowhead_dir = config['ARROWHEAD_DIR']
    saving_dir = config['SAVING_DIR']
    cell_type = config['HIC_CELLTYPE']
    quality = config['HIC_QUALITY']
    resol = config['HIC_RESOLUTION']
    normalization_list = config['DOMAINS_NORMALIZATION_LIST']
    resolution_list = config['DOMAINS_RESOLUTION_LIST']
    chrom_list = config['chrs']
    
    for normalization in normalization_list:
        
        for resolution in resolution_list:
            
            print(f'Process domains from {normalization} normalization at resolution {resolution}')
    
            ############################################################
            # Load raw domains for all chromosomes
            ############################################################
            print('Load raw domains for all chromosomes')

            tads_dict = {}
            for chrom in chrom_list:
                fname = arrowhead_dir+f'{normalization}/{resolution}/{chrom}/{resolution}_blocks.bedpe'
                if os.path.isfile(fname):
                    tads_df = pd.read_csv(fname, sep='\t', header=0)
                    tads_df = tads_df.drop(labels=0, axis=0)
                    tads_df[['x1', 'y1', 'x2', 'y2']] = tads_df[['x1', 'y1', 'x2', 'y2']].astype(int)
                    tads_df.loc[:, 'length'] = tads_df['x2']-tads_df['x1']
                    tads_df = tads_df.sort_values(by=['x1', 'x2'])
                    tads_df = tads_df.reset_index(drop=True)
                    tads_dict[chrom] = tads_df

            ############################################################
            # Drop nested domains and merge overlapping domains
            ############################################################
            print('Drop nested domains and merge overlapping domains')

            processed_tads_list = []
            unprocessed_tads_list = []
            for chrom in tads_dict.keys():
                tads_df = tads_dict[chrom]

                # Select TAD location columns
                domainsDT_tmp = tads_df[['#chr1', 'x1', 'x2']].copy()
                domainsDT_tmp.columns = ['chromo', 'start', 'end']
                domainsDT_tmp.loc[:, 'start'] = domainsDT_tmp['start']+1

                # Order the domains
                domainsDT_tmp = domainsDT_tmp.sort_values(by=['start', 'end'], ascending=[True, True])
                domainsDT_tmp = domainsDT_tmp.drop_duplicates()

                # Discard nested and overlapping domains
                domainsDT = pd.DataFrame(columns=["chromo", "start", "end"])
                n_all = len(domainsDT_tmp)
                n_nested = 0
                n_overlapping = 0

                for i in domainsDT_tmp.index:
                    curr_chromo = domainsDT_tmp.loc[i]["chromo"]
                    curr_start = domainsDT_tmp.loc[i]["start"]
                    curr_end = domainsDT_tmp.loc[i]["end"]
                    # if there is a nested domain, discard it
                    if any(
                        (domainsDT_tmp["start"].drop(labels=[i]) <= curr_start) &
                        (domainsDT_tmp["end"].drop(labels=[i]) >= curr_end)
                    ):
                        n_nested = n_nested+1
                        continue
                    # if the current domain overlaps previous domain(s), merge it
                    if any(curr_start < domainsDT["end"]):
                        n_overlapping = n_overlapping+1
                        domainsDT.iloc[-1, 2] = curr_end
                        continue
                    curr_line = pd.DataFrame(
                        {"chromo": curr_chromo, "start": curr_start, "end": curr_end}, index=[0]
                    )
                    domainsDT = pd.concat([domainsDT, curr_line], ignore_index=True)
                    # ensure ordering of the domains
                    domainsDT = domainsDT.sort_values(["start", "end"])

                domainsDT.loc[:, 'length'] = domainsDT['end']-domainsDT['start']+1
                processed_tads_list.append(domainsDT)
                
                domainsDT_tmp.loc[:, 'length'] = domainsDT_tmp['end']-domainsDT_tmp['start']+1
                unprocessed_tads_list.append(domainsDT_tmp)

            ############################################################
            # Concatenate all TADs for all chromosomes
            ############################################################
            print('Concatenate all TADs for all chromosomes')

            processed_tads_df = pd.concat(processed_tads_list)
            fname = arrowhead_dir+f'{normalization}/{resolution}/processed_tads_list.csv'
            processed_tads_df.to_csv(fname)
            
            unprocessed_tads_df = pd.concat(unprocessed_tads_list)
            fname = arrowhead_dir+f'{normalization}/{resolution}/unprocessed_tads_list.csv'
            unprocessed_tads_df.to_csv(fname)
            print('---------------------------------------------------------------')


if __name__ == "__main__":
    main()