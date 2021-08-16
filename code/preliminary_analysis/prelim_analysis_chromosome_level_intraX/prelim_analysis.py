# Import standard libraries
import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math

# Custom libraries
import prototype_utils as pu


def load_adhesome_data(adhesome_components_filename):
    '''
    Loads dataframe of intrinsic and associated adhesome genes
    Args:
        adhesome_components_filename: (String) adhesome file name
    Returns:
        A pandas DataFrame including adhesome genes with metadata
    '''
    df_components = pd.read_csv(adhesome_components_filename, sep = ',', header = 0)
    df_components['Official Symbol'] = df_components['Official Symbol'].str.upper()
    df_components.columns = ['geneSymbol','geneID','proteinName','swisprotID','synonyms','functionalCategory','FA']
    return(df_components)
    

def get_gene_locations(gene_locations_filename, gene_id_filename):
    '''
    Loads dataframe of gene locations on the hg19 reference genome using the UCSC
    gene nomenclature, and matches the name of every UCSC gene to the corresponding 
    HGNC name
    Args:
        gene_locations_filename: (String) file name for the UCSC gene locations
        gene_id_filename: (String) file name for the UCSC-HGNC gene names map
    Returns:
        A pandas DataFrame of HGNC gene locations on the hg19 reference genome
    '''
    # Load gene locations
    df_loc0 = pd.read_csv(gene_locations_filename, sep = '\t', header = 0)
    
    # Load correspondance between UCSC gene name and HGNC gene name
    df_name0 = pd.read_csv(gene_id_filename, sep = '\t', header = 0,dtype={"rfamAcc": str, "tRnaName": str})
    # Only keep UCSC name and geneSymbol
    df_name0 = df_name0[['#kgID','geneSymbol']]
    df_name0.columns = ['transcript','geneSymbol']
    df_name0['geneSymbol'] = df_name0['geneSymbol'].str.upper()
    
    # Merge df_loc0 and df_name0
    df_loc1 = pd.merge(df_name0, df_loc0, on=['transcript'])
    
    # Filter out irregular chromosomes
    keep_chrom = ['chr'+str(i) for i in np.arange(1,23)]
    df_loc2 = df_loc1[df_loc1['#chrom'].isin(keep_chrom)]
    df_loc2 = df_loc2.sort_values(by=['geneSymbol'])
    
    # Remove ribosomal RNA (may be able to remove more irrelevant things, how?)
    df_loc3 = df_loc2[~df_loc2['geneSymbol'].str.contains('RRNA')]
    
    # Drop duplicates
    df_loc4 = df_loc3.drop_duplicates(subset=['geneSymbol'], keep='first')
    
    return(df_loc4)
    
    
def get_adhesome_genes_location(df_components, df_loc):
    '''
    Retrieves the location of each adhesome gene on the hg19 reference genome
    Args:
        df_components: (pandas DataFrame) dataframe of adhesome genes with metadata
        df_loc: (pandas DataFrame) dataframe of gene locations on the hg19 reference genome
    Returns:
        A pandas DataFrame indicating the position of every adhesome gene on the hg19 reference genome
    '''
    # Find location of adhesome genes by merging df_components and df_loc
    adhesome_loc_df0 = pd.merge(df_components, df_loc, on=['geneSymbol'], how='inner')
    missing_adhesome_genes = list(set(df_components['geneSymbol']).difference(set(df_loc['geneSymbol'])))
    print('Adhesome genes absent from UCSC genes: '+str(missing_adhesome_genes))
    
    # Only keep relevant columns for subsequent analysis
    adhesome_loc_df = adhesome_loc_df0[['geneSymbol','#chrom','chromStart','chromEnd']]
    adhesome_loc_df.columns = ['gene','chrom','genoStart','genoEnd']
    adhesome_loc_df.loc[:,'geneLength'] = list(adhesome_loc_df['genoEnd']-adhesome_loc_df['genoStart'])
    
    return(adhesome_loc_df)
 

def get_adhesome_genes_loci(adhesome_loc_df, resol):
    '''
    Finds all loci on the hg19 reference genome corresponding to adhesome genes
    Args:
        adhesome_loc_df: (pandas DataFrame) dataframe containing the location of adhesome genes on the hg19 reference genome
        resol: (int) the resolution of HiC data
    Returns:
        A pandas DataFrame where each row corresponds to one adhesom gene locus, including gene coverage information (i.e. the 
        proportion of the locus occupied by the corresponding gene)
    '''
    # Specify start locus and end locus
    adhesome_loc_df.loc[:,'startLocus_id'] = list((adhesome_loc_df['genoStart']//resol+1))
    adhesome_loc_df.loc[:,'endLocus_id'] = list((adhesome_loc_df['genoEnd']//resol+1))
    adhesome_loc_df.loc[:,'startLocus'] = list((adhesome_loc_df['genoStart']//resol+1)*resol)
    adhesome_loc_df.loc[:,'endLocus'] = list((adhesome_loc_df['genoEnd']//resol+1)*resol)
    
    # Compute coverage of the gene on its start locus and end locus
    mask = adhesome_loc_df['startLocus']==adhesome_loc_df['endLocus']
    adhesome_loc_df.loc[mask,'startLocus_coverage'] = adhesome_loc_df.loc[mask,'geneLength']/resol
    adhesome_loc_df.loc[mask,'endLocus_coverage'] = adhesome_loc_df.loc[mask,'startLocus_coverage']
    mask = adhesome_loc_df['startLocus']<adhesome_loc_df['endLocus']
    adhesome_loc_df.loc[mask,'startLocus_coverage'] = (adhesome_loc_df.loc[mask,'startLocus']-adhesome_loc_df.loc[mask,'genoStart'])/resol
    adhesome_loc_df.loc[mask,'endLocus_coverage'] = (adhesome_loc_df.loc[mask,'genoEnd']-adhesome_loc_df.loc[mask,'endLocus']+resol)/resol
    
    # Add column with list of loci
    ngenes = adhesome_loc_df.shape[0]
    adhesome_loc_df.loc[:,'loci'] = [np.arange(adhesome_loc_df.loc[i,'startLocus_id'], adhesome_loc_df.loc[i,'endLocus_id']+1) for i in range(ngenes)]
    
    # Explode dataframe based on loci
    adhesome_loc_df = pu.unnesting(adhesome_loc_df,['loci'])
    
    # Reorganize columns of dataframe
    adhesome_loc_df = adhesome_loc_df[['gene','chrom','loci',
                                       'genoStart','genoEnd','geneLength',
                                       'startLocus_id','startLocus','startLocus_coverage',
                                       'endLocus_id','endLocus','endLocus_coverage']]
    return(adhesome_loc_df)   


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
    print('Options successfully parsed, read arguments...')
    config = parse_config(config_filename)
    raw_hic_dir = config['HIC_DIR']
    genome_dir = config['GENOME_DIR']
    adhesome_dir = config['ADHESOME_DIR']
    prelim_results_dir = config['PRELIM_RESULTS_DIR']
    processed_hic_data_dir = config['PROCESSED_HIC_DATA_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    quality = config['HIC_QUALITY']
    chr_list = config['chrs']
    filter_size = config['CENTROMERE_FILTER_SIZE']
    
    # Load adhesome data
    print('Load adhesome data...')
    adhesome_components_filename = adhesome_dir+'components.csv'
    df_components = load_adhesome_data(adhesome_components_filename)
    
    # Load position of genes on the genome
    print('Load gene locations on hg19 reference genome...')
    gene_locations_filename = genome_dir+'chrom_hg19.loc_canonical'
    gene_id_filename = genome_dir+'chrom_hg19.name'
    df_loc = get_gene_locations(gene_locations_filename, gene_id_filename)
    
    # Find location of adhesome genes by merging df_components and df_loc
    print('Find location of adhesome genes on hg19 reference genome...')
    adhesome_loc_df = get_adhesome_genes_location(df_components, df_loc)

    # Specify start locus and end locus
    print('Find HiC loci occupied by adhesome genes...')
    adhesome_loc_df = get_adhesome_genes_loci(adhesome_loc_df, resol)

    # List all chromosome pairs and compare HiC contact values between adhesome loci versus HiC contact values between all loci
    chr_pairs = list(itertools.combinations(chr_list, 2))
    names = ['chr'+str(i) for i in chr_list]
    adhesome_mean_hic_df = pd.DataFrame(np.zeros((len(chr_list),len(chr_list))), index=names, columns=names)
    random_mean_hic_df = pd.DataFrame(np.zeros((len(chr_list),len(chr_list))), index=names, columns=names)
    effect_size_df = pd.DataFrame(np.zeros((len(chr_list),len(chr_list))), index=names, columns=names)
    
    # Initialize empty figure
    grid_size = [np.floor(np.sqrt(len(chr_pairs))),np.ceil(np.sqrt(len(chr_pairs)))]
    fig = plt.figure(figsize=(50,50))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    pair_count = 0
    
    print("Parse adhesome HiC data for all "+str(len(chr_pairs))+" chromosome pairs")
    for pair in chr_pairs:
        print('Parse adhesome HiC data for chromosome pair '+str(pair))
        print('---------------------------------------------------------------')
        chr1, chr2 = pair
        pair_count += 1
        
        # Restrict adhesome dataframe to chr1 and chr2
        print('Select adhesome genes')
        adhesome_chr1_df = adhesome_loc_df.loc[adhesome_loc_df['chrom']=='chr'+str(chr1)]
        adhesome_chr2_df = adhesome_loc_df.loc[adhesome_loc_df['chrom']=='chr'+str(chr2)]
        
        # Load HiC data for this chromosome pair
        print('Load processed HiC data')
        processed_hic_filename = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.pkl'
        hic_chpair_df = pickle.load(open(processed_hic_data_dir+processed_hic_filename, 'rb'))
        
        # Restrict HiC data to loci corresponding to adhesome genes
        print('Restrict HiC data to adhesome genes')
        adhesome_chr1_loci = adhesome_chr1_df['loci']*resol
        adhesome_chr2_loci = adhesome_chr2_df['loci']*resol
        hic_chpair_df_restrict = hic_chpair_df.loc[adhesome_chr1_loci,adhesome_chr2_loci]
        
        # Restrict HiC data to random set of loci
        np.random.seed(13)
        nloci_chr1 = len(adhesome_chr1_loci)
        nloci_chr2 = len(adhesome_chr2_loci)
        random_chr1_loci = np.random.choice(hic_chpair_df.index.values, nloci_chr1)
        random_chr2_loci = np.random.choice(hic_chpair_df.columns.values, nloci_chr2)
        hic_chpair_df_random = hic_chpair_df.loc[random_chr1_loci,random_chr2_loci]
        
        # Compute average HiC contacts between adhesome loci and between random loci
        random_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)] = np.mean(hic_chpair_df_random.values.flatten())
        adhesome_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)] = np.mean(hic_chpair_df_restrict.values.flatten())
        effect_size_df.loc['chr'+str(chr1), 'chr'+str(chr2)] = (adhesome_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)]-random_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)])/np.std(hic_chpair_df_random.values.flatten())
        
        # Fill in grid
        ax = fig.add_subplot(grid_size[0], grid_size[1], pair_count)
        sns.distplot(hic_chpair_df_random.values.flatten(), 
                     kde=True, hist=False,
                     ax=ax,
                     color='green',
                     axlabel='HiC contact values', label='HiC contacts across random loci')
        ax.axvline(random_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)], color='green', linestyle='dashed')
        sns.distplot(hic_chpair_df_restrict.values.flatten(), 
                     kde=True, hist=False,
                     ax=ax,
                     color='orange',
                     label='HiC contacts across adhesome loci')
        ax.axvline(adhesome_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)], color='orange', linestyle='dashed')
        ax.get_legend().remove()
        ax.set_title('chr'+str(chr1)+'/'+'chr'+str(chr2))
        
        '''
        # Plot density plots of HiC contacts between adhesome loci and between random loci
        print('Save HiC contacts density plots')
        plt.figure()
        sns.distplot(hic_chpair_df_random.values.flatten(), 
                     kde=True, hist=False, 
                     color='green',
                     axlabel='HiC contact values', label='HiC contacts across random loci')
        plt.axvline(random_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)], color='green', linestyle='dashed')
        sns.distplot(hic_chpair_df_restrict.values.flatten(), 
                     kde=True, hist=False, 
                     color='orange',
                    label='HiC contacts across adhesome loci')
        plt.axvline(adhesome_mean_hic_df.loc['chr'+str(chr1), 'chr'+str(chr2)], color='orange', linestyle='dashed')
        plotname = 'adhesome_random_hic_contacts_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'.png'
        plt.savefig(prelim_results_dir+plotname, format='png')
        plt.close()
        '''
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor = (0,-0.05,1,1), bbox_transform = plt.gcf().transFigure, fontsize='xx-large')
    plotname = 'adhesome_vs_random_hic_contacts_grid'+'.pdf'
    plt.savefig(prelim_results_dir+plotname, format='pdf')
    plt.close()
    
    # Save average HiC contact matrices for all chromosome pairs as pickle
    picklename_all = prelim_results_dir+'average_hic_randomloci'+'.pkl'
    pickle.dump(random_mean_hic_df, open(picklename_all, 'wb'))
    picklename_adhesome = prelim_results_dir+'average_hic_adhesomeloci'+'.pkl'
    pickle.dump(adhesome_mean_hic_df, open(picklename_adhesome, 'wb'))
    picklename_effectsize = prelim_results_dir+'adhesome_hic_contacts_effectsize'+'.pkl'
    pickle.dump(effect_size_df, open(picklename_effectsize, 'wb'))


if __name__ == "__main__":
    main()
    