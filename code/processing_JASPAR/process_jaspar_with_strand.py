# Import standard libraries
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import pickle
import itertools
from itertools import groupby
import os.path
import math
import pybedtools
import time
from tqdm import tqdm
import random
import MOODS.parsers
import MOODS.tools
import MOODS.scan
import subprocess
# Custom libraries
import utils as lu


def get_homosapiens_tfbs(tf_dir):
    '''
    Get list of all Homo Sapiens TFBS
    Args:
        tf_dir: (str) directory fo all TF information
    Returns:
        The list of all Homo Sapiens TFBS
    '''
    # Read metadata
    jaspar_df = pd.read_csv(tf_dir+'JASPAR-HomoSapiens.csv', 
                            header=0, usecols=['ID','Name','Species','Class','Family'])
    homosapiens_ids = np.unique(jaspar_df['ID'])
    return(homosapiens_ids)


def convert_jaspar_to_pfm(jaspar_dir, pfm_dir, selected_tfbs):
    '''
    Converts .jaspar position weight matrix files into .pfm format
    Args:
        jaspar_dir: (str) directory of .jaspar files
        pfm_dir: (str) directory where to create the .pfm files
        selected_tfbs: (Numpy array) the tfbs's for which the conversion should be done
    Returns:
        Void
    '''
    # Loop over all files in jaspar_dir and convert the Homo Sapiens ones to .pfm
    for filename in os.listdir(jaspar_dir):
        if filename.endswith(".jaspar") and (filename.split('.jaspar')[0] in selected_tfbs): 
            # Open .jaspar file
            with open(jaspar_dir+filename) as f:
                lines = f.readlines()
            # Change format
            lines[1] = lines[1].replace('A  [', '').replace(']', '').lstrip()
            lines[2] = lines[2].replace('C  [', '').replace(']', '').lstrip()
            lines[3] = lines[3].replace('G  [', '').replace(']', '').lstrip()
            lines[4] = lines[4].replace('T  [', '').replace(']', '').lstrip()
            # Save to .pfm file
            pfm_file = open(pfm_dir+filename.split('.jaspar')[0]+'.pfm', "w")
            for line in lines:
                pfm_file.write(line)
            pfm_file.close()


def create_ucsc_hgnc_dict(gene_id_filename):
    '''
    Create a dictionary matching gene UCSC names with HGNC names
    Args:
        gene_id_filename: (str) path and name of file for matching UCSC gene names and HGNC gene names
    Returns:
        A dictionary matching gene UCSC names with HGNC names
    '''
    df_name0 = pd.read_csv(gene_id_filename, sep = '\t', header = 0,dtype={"rfamAcc": str, "tRnaName": str})
    df_name0 = df_name0[['#kgID','geneSymbol']]
    df_name0.columns = ['transcript','geneSymbol']
    df_name0['geneSymbol'] = df_name0['geneSymbol'].str.upper()
    ucsc_to_hgnc = {df_name0.iloc[i,0]: df_name0.iloc[i,1] for i in range(df_name0.shape[0])}
    return(ucsc_to_hgnc)


def build_fasta_files(fasta_name, ucsc_to_hgnc, selected_genes, output_dir):
    '''
    Builds multiple fasta files from a large combined fasta file
    Args:
        fasta_name: (str) path and name of the large fasta file
        ucsc_to_hgnc: (dict) mapping between UCSC gene names and HGNC gene names
        selected_genes: (Numpy array) genes of interest
        output_dir: (str) directory where to create the fasta files
    Returns:
        Void
    '''
    fh = open(fasta_name)
    # ditch the boolean (x[0]) and just keep the header or sequence since we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        # read the gene
        headerStr = header.__next__().strip()
        ucsc_gene = headerStr.split(' ')[0].split('_')[2]
        hgnc_gene = ucsc_to_hgnc[ucsc_gene]
        loc = headerStr.split(' ')[1]
        # join all sequence lines to one.
        seq = "\n".join(s.strip() for s in faiter.__next__())
        if (hgnc_gene in selected_genes): 
            with open(output_dir+ucsc_gene+'.fa', 'w') as f:
                f.write("%s\n" % headerStr)
                f.write("%s" % seq)


def main():
    print('Specify directories')
    # Directory of adhesome data
    dir_adhesome = '/home/louiscam/projects/gpcr/data/adhesome_data/'
    # Directory of genome data
    dir_genome = '/home/louiscam/projects/gpcr/data/genome_data/'
    prom_hg19_seq_dir = dir_genome+'prom_hg19_seq_dir/'
    # Directory of processed HiC
    dir_processed_hic = '/home/louiscam/projects/gpcr/save/processed_hic_data_dir/'
    # Directory for storing preliminary results
    prelim_results_dir = '/home/louiscam/projects/gpcr/save/prelim_results_dir/'
    # Directory of epigenomic data
    epigenome_dir = '/home/louiscam/projects/gpcr/data/epigenome_data/'
    processed_epigenome_data_dir = '/home/louiscam/projects/gpcr/save/processed_epigenome_data_dir/'
    # Saving directory
    saving_dir = '/home/louiscam/projects/gpcr/save/figures/'
    # Directory of JASPAR data
    tf_dir = '/home/louiscam/projects/gpcr/data/tf_data/'
    jaspar_dir = tf_dir+'jaspar_data/'
    pfm_dir = tf_dir+'pfm_data/'
    moods_out_dir = tf_dir+'moods_cage_outdir_with_strand/'
    cage_fasta_dir = dir_genome+'cage_hg19_seq_dir_with_strand/'
    
    # Convert PFMs into .pfm format
    print('Create PFMs for all TF binding motifs')
    homosapiens_ids = get_homosapiens_tfbs(tf_dir)
    convert_jaspar_to_pfm(jaspar_dir, pfm_dir, homosapiens_ids)
    
    # Run MOODS for each CAGE peak
    print('Run MOODS on the window around each CAGE peak')
    MOODS_exec = "~/MOODS/python/scripts/moods-dna.py"
    all_cage_seq_files = os.listdir(cage_fasta_dir)
    n_all_cage_seq_files = len(all_cage_seq_files)
    for i in range(n_all_cage_seq_files):
        cage_file = all_cage_seq_files[i]
        if cage_file.endswith(".fa"):
            cage_name = cage_file.strip('.fa')
            result = subprocess.run(MOODS_exec+
                                    " -m "+pfm_dir+'*.pfm'+
                                    " -s "+cage_fasta_dir+cage_file+
                                    " -p "+"0.00001"+
                                    " --bg "+"0.25 0.25 0.25 0.25"+
                                    " --ps "+"1.0"+
                                    " -o "+moods_out_dir+cage_name+'.txt', 
                                    shell=True)
            print('Peak '+cage_name+' completed ('+str(i+1)+'/'+str(n_all_cage_seq_files)+')')

if __name__ == "__main__":
    main()
    