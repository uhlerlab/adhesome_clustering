import sys, getopt
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import random
from scipy import sparse
from scipy.special import comb
from scipy.special import gammaln
from scipy.special import erfcx
from scipy.stats import norm
import scipy.stats
import seaborn
import csv
import pandas as pd
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import os
from joblib import Parallel, delayed

""" This script runs Large Average Submatrix algorithm on Hi-C contact matrices. 
Output is contiguous submatrices with a large average for each chromosome pair.
Adjust the threshold for this algorithm in the configuration file run_params.json if necessary.
Script borrowed from https://github.com/anastasiyabel/functional_chromosome_interactions/"""


def get_chrom_sizes(genome_dir,resol):
    '''
    Constructs dataframe of chromosome sizes (in bp and in loci counts at the chosen resolution)
    Args:
        genome_dir: (string) directory of genome data
        resol: (int) hic resolution
    Returns:
        A pandas datafrawith columns chr (chromosome), size, size_loci, size_roundup
    '''
    sizes_filename = genome_dir+'chrom_hg19.sizes'
    df_sizes = pd.read_csv(sizes_filename, sep = '\t', header = None, names=['chr','size'])
    df_sizes['chr'] = df_sizes['chr'].str.split('chr',expand=True)[1]
    df_sizes['size_loci'] = np.ceil(df_sizes['size']/resol).astype(int)
    df_sizes['size_roundup'] = np.ceil(df_sizes['size']/resol).astype(int)*resol
    df_sizes = df_sizes[~df_sizes['chr'].str.contains('hap|alt|Un|rand')]
    return(df_sizes)


def simulate_data():
    '''
    Function used to simlate a 20x40 random array from a standard Gaussian distribution
    Args:
        void
    Returns:
        Numpy array with independent standard Gaussian data
    '''
    data = np.random.randn(20, 40)
    x = 2.5*np.random.randn(5,3) + 10
    data[2:7, 3:6] = x
    return(data)


def transform(data):
    '''
    Function to log-transform the input data matrix
    Args:
        data: (Numpy array) data matrix to transform
    Returns:
        The log-transformed matrix
    '''
    data_trans = np.log(1 + data)
    return(data_trans)


def rescale(data, mean, std):
    '''
    Converts a data matrix to z-score
    Args:
        data: (Numpy array) data matrix to rescale
        mean: (float) centering factor
        std: (float) scaling factor
    Returns:
        The centered and rescaled data matrix
    '''
    # convert to z scores
    z = (data-mean)/std
    return(z)


def check_submatrix_below_threshold(sub_matrix, threshold):
    '''
    Function to check whether the score of the input submatrix is below the input threshold
    Args:
        sub_matrix: (Numpy array) submatrix to check
        threshold: (float) score threshold
    Returns:
        A Boolean specifying whether the submatrix scoe is below threshold
    '''
    # Check whether score is below threshold based on the the size of submatrix
    num_rows, num_cols = sub_matrix.shape
    avg_sub_matrix = np.sqrt(num_rows*num_cols)*np.average(sub_matrix)
    if (avg_sub_matrix > threshold):
        check = True
    else:
        check = False
    return check


def residual(U, data, rows, cols):
    '''
    Subtract average of submatrix U from the full data matrix
    Args:
        U: (Numpy array) submatrix
        data: (Numpy array) full data matrix
        rows: (Numpy array) row indices corresponding to U in data
        cols: (Numpy array) column indices corresponding to U in data
    Returns:
        The full data matrix from which the average of U has been subtracted
    '''
    avg = np.mean(U)
    # only subtract avg for submatrix
    data[np.ix_(rows,cols)] = data[np.ix_(rows,cols)] - avg
    return(data)


def large_average_submatrix_adj(data, chr1, chr2, threshold_new):
    '''
    
    '''
    # store some data on iterations
    dir = config["HIC_NEW_DIR"] + str(chr1) + '_' + str(chr2) + '/'
    if (os.path.isdir(dir) == False):
        os.makedirs(dir)

    # algorithm until score falls below threshold
    continue_search = True
    iter = 0
    # store matrices
    start_rows, stop_rows, start_cols, stop_cols, best_score_list, avg_list = [], [], [], [], [], []

    while (continue_search):
        rows, cols, sub_matrix, best_score = search_main(data, dir, iter)
        data = residual(sub_matrix, data, rows, cols)
        # check whether score is below threshold based on the the size of submatrix
        continue_search = check_submatrix_below_threshold(sub_matrix, threshold_new)
        if (continue_search == True):
            start_rows.append(rows[0])
            stop_rows.append(rows[-1])
            start_cols.append(cols[0])
            stop_cols.append(cols[-1])
            best_score_list.append(best_score)
            avg_list.append(np.average(sub_matrix))
            iter = iter + 1
            print 'Best score = ', best_score
            print 'Average = ', np.average(sub_matrix)
            print rows[0], rows[-1], cols[0], cols[-1]

    return start_rows, stop_rows, start_cols, stop_cols, best_score_list, avg_list


def search_main(data, dir, iter):
    '''
    
    Args:
        data: (Numpy array) data matrix on which the LAS algorithm is applied
        dir: (string) directory to save iteration statistics
        iter: (int) current iteration
    Returns:
        Information on the best submatrix
    '''
    num_iter = 100
    # keep track of submatrix params that you get out
    search_attributes = np.empty((num_iter, 5))
    for iteration in range(num_iter):
        start_row, k, start_col, l, curr_score = search(data)
        search_attributes[iteration] = start_row, k, start_col,l ,curr_score

    # save the iterations
    np.savetxt(dir + 'sub_matrix' + str(iter) + '.txt', search_attributes)


    best_start_row, best_k, best_start_col, best_l, best_score =  search_attributes[np.argmax(search_attributes[:,4])]
    rows = np.arange(best_start_row, best_start_row + best_k, dtype = 'int')
    cols = np.arange(best_start_col, best_start_col + best_l, dtype = 'int')
    sub_matrix = data[np.ix_(rows,cols)]
    return rows, cols, sub_matrix, best_score

def score(U, data):
    '''
    Computes the LAS submatrix score of submatrix U in data
    Args:
        U: (Numpy array) submatrix to score
        data: (Numpy array) HiC contact matrix
    Returns:
        The LAS score of U
    '''
    m,n = data.shape
    k,l = U.shape
    tau = np.mean(U)
    sc = comb(m,k)*comb(n,l)*norm.cdf(-tau*np.sqrt(k*l))
    sc = -np.log(sc)
    return sc


def score_sum(sum_u, k, l, data):
    '''
    Not sure what this is
    Args:
        sum_u:
        k:
        l:
        data:
    Returns:
        sc
    '''
    m,n = data.shape
    cnr = gammaln(m + 1) - gammaln(k+1) - gammaln(m-k+1)
    cnc = gammaln(n + 1) - gammaln(l+1) - gammaln(n-l+1)
    ar = sum_u/np.sqrt(k*l)
    rest2 = -(ar*ar)/2.0 + np.log(erfcx(ar/np.sqrt(2))*0.5)
    sc = -rest2 -cnr - cnc
    return sc

def grouped_sum(array, N):
    '''
    Not sure what this does
    Args:
        array:
        N:
    Returns:
        adj_sum
    '''
    length = len(array) - N + 1
    # initialize the array
    adj_sum = np.zeros((length))
    for i in range(0,N):
        adj_sum = adj_sum + array[i:length+i]
    return adj_sum


def search(data, resol):
    # The search space of the LAS algorithm is limited to contiguous submatrices of at most 10Mbx10Mb in size
    max_num_rows = int(10000000.0/resol)
    max_num_cols = int(10000000.0/resol)
    
    # Pick initial k, l randomly between 0 and the upper bound on the number of rows/columns
    k = random.randint(1, max_num_rows)
    l = random.randint(1, max_num_cols)
    
    # Run search procedure with fixed k, l first
    row_set, col_set = search_fixed_k_l(data, k,l)

    # Allow k and l to vary
    # Initialize the running average
    pre_score = -1000000
    curr_score = 0
    # Iterate until convergence
    while(pre_score != curr_score):
        # Sum across columns
        row_summed = np.sum(col_set, axis =1)
        start_row, k, score_rows = enumerate_adj_submatrix_scores(data, row_summed, max_num_rows, k, l, 'row')
        # Make a row set
        row_set = data[start_row:start_row+k, :]

        # columns
        col_summed = np.sum(row_set, axis =0)
        start_col, l, score_cols = enumerate_adj_submatrix_scores(data, col_summed, max_num_cols, k, l, 'col')
        # Make a col set
        col_set = data[:,start_col:start_col+l]

        # Update scores
        pre_score = curr_score
        curr_score = score_cols
        # print 'Score = ', pre_score, curr_score

    return start_row, k, start_col, l, curr_score


def enumerate_adj_submatrix_scores(data, row_summed, max_num_rows, k, l, row_or_col):
    if (row_or_col == 'row'):
            start_row_best_list = []
            start_row_best_ind_list = []
            # Let the number of rows to include vary (+1 to make the range inclusive)
            possible_num_rows = range(1, max_num_rows + 1)
            for i in possible_num_rows:
                # make all possible submatrices by summing adjacent rows
                adj_row_sum = grouped_sum(row_summed, i)
                score_list = [score_sum(sum_u, i, l, data) for sum_u in adj_row_sum]
                # find best starting row
                start_row_best_ind, start_row_best = max(enumerate(score_list), key=operator.itemgetter(1))
                start_row_best_ind_list.append(start_row_best_ind)
                start_row_best_list.append(start_row_best)
    if (row_or_col == 'col'):
            start_row_best_list = []
            start_row_best_ind_list = []
            possible_num_rows = range(1, max_num_rows + 1)
            for i in possible_num_rows:
                # make all possible submatrices by summing adjacent rows
                adj_row_sum = grouped_sum(row_summed, i)
                # LINE BELOW is THE ONLY DIFFENECE BETWEEN ROW AND COL CODE
                score_list = [score_sum(sum_u, k, i, data) for sum_u in adj_row_sum]
                # find best starting row
                start_row_best_ind, start_row_best = max(enumerate(score_list), key=operator.itemgetter(1))
                start_row_best_ind_list.append(start_row_best_ind)
                start_row_best_list.append(start_row_best)
    # choose the best scoring 
    ind, score_rows = max(enumerate(start_row_best_list), key=operator.itemgetter(1))
    start_row = start_row_best_ind_list[ind]
    k = possible_num_rows[ind]
    return start_row, k, score_rows


def search_fixed_k_l(data, k, l):
    '''
    Runs the basic search procedure of the LAS algorithm for fixed k and l
    Args:
        data: (Numpy array) matrix on which LAS is run
        k: (int) number of rows of the submatrix
        l: (int) number of columns of the submatrix
    Returns:
        Row set and column set of the submatrix obtained at convergence
    '''
    num_rows = data.shape[0]
    num_cols = data.shape[1]
    # Initialize (select l adjacent columns at random)
    start_col = random.randint(0, num_cols-l)
    col_set = data[:,start_col:start_col+l]
    n = col_set.shape[0]

    # Initialize the running average
    pre_avg = -1000000
    curr_avg = 0
    # Iterate until convergence
    while(pre_avg != curr_avg):
        # First get the k adjacent rows with the largest sum over the chosen l columns
        # Make another matrix that is the sum of k adjacent columnns
        row_summed_data = np.asarray([np.sum(col_set[i:i+k,:]) for i in range(0, n-k+1)])
        # Choose starting row that gave the largest sum
        start_row = np.argmax(row_summed_data)
        row_set = data[start_row:start_row+k, :]
        m = row_set.shape[1]

        # Then get l adjacent columns with the largest sum over the chosen k rows
        # Make another matrix that is the sum of l adjacent rows
        col_summed_data = np.asarray([np.sum(row_set[:,j:j+l]) for j in range(0, m-l+1)])
        # Choose starting row that gave the largest sum
        start_col = np.argmax(col_summed_data)
        col_set = data[:,start_col:start_col+l]
        n = col_set.shape[0]

        # Compute the new average of the submatrix
        sub_matrix = data[np.ix_(range(start_row, start_row+k), range(start_col, start_col+l))]
        # update averages
        pre_avg = curr_avg
        curr_avg = np.mean(sub_matrix)
        #print curr_avg, pre_avg
    return row_set, col_set


def df_remove_zeros_rows_cols(df):
    '''
    Remove indices of dataframe that are all zero col- or row-wise
    Args:
        df: (pandas DataFrame) dataframe of interest
    Returns:
        The pruned dataframe
    '''
    # Drop rows that are all 0
    df = df[(df.T != 0).any()]
    # Drop columns that are all 0
    df = df[df.columns[(df != 0).any()]]
    return df


def get_hic_matrix(hic_filename, chr1, chr2, resol, df_sizes):
    '''
    Constructs dense HiC contact matrix for the pair chr1/chr2
    Args:
        hic_filename: (string) name of HiC file
        chr1: (int) chromosome 1
        chr2: (int) chromosome 2
        resol: (int) HiC resolution
        df_sizes: (pandas DataFrame) dataframe with size of all chromosomes
    Returns:
        Dense HiC dataframe
    '''
    # Load HiC data in sparse format
    data = np.loadtxt(hic_filename, delimiter = '\t')
    row_ind = data[:,0]/resol
    col_ind = data[:,1]/resol
    contact_values = data[:,2]
    
    # Obtain chromosome sizes
    chr1_size = int(df_sizes[df_sizes['chr']==str(chr1)]['size_loci'])
    chr2_size = int(df_sizes[df_sizes['chr']==str(chr2)]['size_loci'])
    
    # Construct sparse HiC matrix
    hic_matrix = csr_matrix((contact_values, (row_ind, col_ind)), shape = (chr1_size, chr2_size))
    hic_dense = np.asarray(hic_matrix.todense())
    row_labels = np.arange(chr1_size)*resol
    col_labels = np.arange(chr2_size)*resol
    df = pd.DataFrame(hic_dense, index = row_labels, columns = col_labels)
    df = df.fillna(0)
    return(df)


def map_pos2rownum(df, row_pos):
    '''
    Retrieves the row number from the index value (position in the genome) in the input dataframe
    Args:
        df: (pandas DataFrame) dataframe
        row_pos: (int) index value
    Returns:
        The row number corresponding to the index value row_pos
    '''
    return(np.where(df.index.values == row_pos)[0][0])


def map_pos2colnum(df, col_pos):
    '''
    Retrieves the column number from the column name value (position in the genome) in the input dataframe
    Args:
        df: (pandas DataFrame) dataframe
        col_pos: (int) column name value
    Returns:
        The column number corresponding to the column name value col_pos
    '''
    return np.where(df.columns.values == str(int(col_pos)))[0][0]


def map_rownum2pos(df, row_num):
    '''
    Retrieves the index value (position in the genome) from a row number in the input dataframe
    Args:
        df: (pandas DataFrame) dataframe
        row_num: (int) row number
    Returns:
        The index value (position of the genome) corresponding to row_num in df
    '''
    positions = df.index.values
    return positions[row_num]


def map_colnum2pos(df, col_num):
    '''
    Retrieves the column name value (position in the genome) from a column number in the input dataframe
    Args:
        df: (pandas DataFrame) dataframe
        col_num: (int) column number
    Returns:
        The column name value (position of the genome, cast as a float) corresponding to col_num in df
    '''
    positions = df.columns.values
    return float(positions[col_num])


def map_num2pos(df, start_rows, stop_rows, start_cols, stop_cols):
    '''
    For each row and column number in a list, figure out what position in the genome it corresponds to
    Args:
        df: (pandas DataFrame) HiC dataframe with genome position annotations (index and column names)
        start_rows: (list of ints) list of row numbers
        stop_rows: (list of ints) list of row numbers
        start_cols: (list of ints) list of col numbers
        stop_cols: (list of ints) list of col numbers
    Returns:
        The corresponding lists of positions in the genome
    '''
    # for each row and column number figure out what position on the genome does it correspond to
    start_row_pos = [map_rownum2pos(df, row_num) for row_num in start_rows]
    stop_row_pos = [map_rownum2pos(df, row_num) for row_num in stop_rows]
    start_col_pos = [map_colnum2pos(df, col_num) for col_num in start_cols]
    stop_col_pos = [map_colnum2pos(df, col_num) for col_num in stop_cols]
    return start_row_pos, stop_row_pos, start_col_pos, stop_col_pos


def numclust_avg(pair):
    '''
    Not sure what this does
    Args:
        pair: (tuple of ints) pair of chromosomes
    Returns:
        void
    '''
    chr1, chr2 = pair
    fname = config["HIC_NEW_DIR"]  + 'intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.csv'
    # Check if the file exists
    if (os.path.isfile(fname) == True):

        df_intermingling = pd.read_csv(fname, index_col = 0)
        plt.figure()
        plt.plot(xrange(df_intermingling.shape[0]), df_intermingling['score'], 'o-')
        plt.xlabel('Cluster #')
        plt.ylabel('Score')
        plt.savefig(config["HIC_NEW_DIR"] + 'cluster_score.chr' + str(chr1) + '_chr' + str(chr2) + '.png')
        plt.close()

        plt.figure()
        plt.plot(xrange(df_intermingling.shape[0]), df_intermingling['avg'], 'o-')
        plt.xlabel('Cluster #')
        plt.ylabel('Average')
        plt.savefig(config["HIC_NEW_DIR"] + 'cluster_average.chr' + str(chr1) + '_chr' + str(chr2) + '.png')
        plt.close()

        
def determine_min_max_hic():
    '''
    Gets the minimum and maximum of log(1+x)-transformed rescaled HiC observed contacts
    across all chromosome pairs (for plotting)
    Args:
        void
    Returns:
        The min and max z-scores across transformed HiC data for all chromosome pairs
    '''
    # List all chromosome pairs
    chr_pairs = list(itertools.combinations(config["chrs"], 2))
    min_list = []
    max_list = []
    # For each pair determine max and min z-score, store in a list
    for pair in chr_pairs:
        chr1, chr2 = pair
        fname = config["HIC_NEW_DIR"]  + 'intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.csv'
        if (os.path.isfile(fname) == True):
                # read in hic matrix
                hic_filename =  config["HIC_FILT_DIR"] +'chr' + str(chr1) + '_chr' + str(chr2) + '.zscore.txt'
                df = pd.read_csv(hic_filename, index_col = 0)
                data = df.as_matrix()
        min_chr_pair = np.min(data)
        max_chr_pair = np.max(data)
        min_list.append(min_chr_pair)
        max_list.append(max_chr_pair)
    minl = min(min_list)
    maxl = max(max_list)
    return minl, maxl


def draw_identified_LASregions(pair, minl, maxl):
    '''
    Plot the log(1+x)-transformed and rescaled HiC contact map of the input pair of chromosomes and overlay 
    the identified LAS regions on top
    Args:
        pair: (tuple of ints) the two chromosomes of interest
        minl: (float) the minimum HiC contact z-score across all chromosome pairs
        maxl: (float) the maximum HiC contact z-score across all chromosome pairs
    Returms:
        void
    '''
    chr1, chr2 = pair
    plt.rc('font', family='serif')
    # No gridlines
    seaborn.set_style("dark", {'axes.grid':False})

    numclust = 50
    fname = config["HIC_NEW_DIR"]  + 'intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.csv'
    # Check if the file exists
    if (os.path.isfile(fname) == True):
        
        # Load transformed HiC data
        hic_filename =  config["HIC_FILT_DIR"] +'chr' + str(chr1) + '_chr' + str(chr2) + '.zscore.txt'
        df = pd.read_csv(hic_filename, index_col = 0)
        data = df.as_matrix()

        # We first plot transformed HiC data with a common scale for all chromosome pairs
        # plt.figure(figsize = (100, 100))
        plt.figure()
        plt.imshow(data, cmap = 'Reds', vmin = minl, vmax = maxl)
        cbar = plt.colorbar()
        cbar.set_label('log(1+x) transformed rescaled HiC observed contacts', fontsize = 12)
        cbar.solids.set_rasterized(True) 

        # Label ticks with genomic position (Mb)
        xaxis = range(0, df.shape[1], 100)
        xlabels = [str(map_colnum2pos(df, x)/1000000.0) for x in xaxis]
        plt.xticks(xaxis, xlabels)
        yaxis = range(0, df.shape[0], 100)
        ylabels = [str(map_rownum2pos(df, y)/1000000.0) for y in yaxis]
        plt.yticks(yaxis, ylabels)

        plt.xlabel('chr' + str(chr2) + ' (Mb)', fontsize = 14)
        plt.ylabel('chr' + str(chr1) + ' (Mb)', fontsize = 14)

        plt.savefig(config["HIC_NEW_DIR"] + 'hic_transformed_rescaled.chr' + str(chr1) + '_chr' + str(chr2) + 'commonscale.eps')

        # We then overlay the intermingling regions on top of the previously plotted HiC contact maps
        df_intermingling = pd.read_csv(fname, index_col = 0)
        # Iterate over all LAS regions found
        for num in range(0, len(df_intermingling)):
            region = df_intermingling.iloc[num]
            start_row = map_pos2rownum(df, region['start row'])
            stop_row = map_pos2rownum(df, region['stop row'])
            start_col = map_pos2colnum(df, region['start col'])
            stop_col = map_pos2colnum(df, region['stop col'])

            # draw vertical lines - columns are same
            plt.plot([start_col, start_col], [start_row, stop_row], 'k-', lw = 0.8)
            plt.plot([stop_col, stop_col], [start_row, stop_row], 'k-', lw = 0.8)
            # draw horizontal lines - rows are same
            plt.plot([start_col, stop_col], [start_row, start_row], 'k-', lw = 0.8)
            plt.plot([start_col, stop_col], [stop_row, stop_row], 'k-', lw = 0.8)

        plt.savefig(fname.split('.csv')[0] + 'commonscale.eps', format = 'eps', dpi = 1000)
        plt.close()


def run_LAS(pair, threshold_new):
    '''
    Runs the LAS algorithm for identifying highly interacting regions in the HiC contact map of the input chromosome
    pair and saves the result to a .csv file
    Args:
        pair: (tuple of ints) the two chromosomes of interest
        threshold_new: (float) the significance threshold for the LAS algorithm
    Returns:
        void
    '''
    chr1, chr2 = pair
    fname = config["HIC_NEW_DIR"] + 'intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) +'.avg_filt.csv'
    if (os.path.isfile(fname) == False):
        print('Running LAS algorithm for chromosome pair '+'chr'+str(ch1)+'chr'+str(ch2)
        
        # Read in HiC matrix
        hic_filename = config["HIC_FILT_DIR"] +'chr' + str(chr1) + '_chr' + str(chr2) + '.zscore.txt'
        df = pd.read_csv(hic_filename, index_col = 0)
        data = df.as_matrix()

        # Run LAS algorithm
        start_rows, stop_rows, start_cols, stop_cols, best_score_list, avg_list = large_average_submatrix_adj(data, chr1, chr2, threshold_new)

        # Convert indices to positions
        start_row_pos, stop_row_pos, start_col_pos, stop_col_pos = map_num2pos(df, start_rows, stop_rows, start_cols, stop_cols)

        # Store results in dataframe
        dic = {'start row' : pd.Series(start_row_pos), 'stop row' : pd.Series(stop_row_pos), 'start col' : pd.Series(start_col_pos), 'stop col': pd.Series(stop_col_pos), 'score' : pd.Series(best_score_list), 'avg' : pd.Series(avg_list)}
        df_intermingling = pd.DataFrame(dic, columns=dic.keys())
        
        # Save dataframe to csv      
        df_intermingling.to_csv(fname)

        
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
    genome_dir = config['GENOME_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    pval_threshold = config['pvalue_threshold']
    num_proc = config['NUM_PROC']
    chroms = config['chrs']
    las_regions_dir = config['LAS_REGIONS_DIR']

    # Define number of iterations of LAS and LAS z-score threshold
    print('Set LAS hyperparameters...')
    iters = 100
    threshold_new = scipy.stats.norm.ppf(pval_threshold)
    print("LAS z-score threshold = "+str(threshold_new))
    
    # Run LAS on all chromosome pairs
    chr_pairs = list(itertools.combinations(chroms, 2))
    print('Run LAS on all '+str(len(chr_pairs))+' chromosome pairs')
    Parallel(n_jobs = num_proc)(delayed(run_LAS)(pair, threshold_new) for pair in chr_pairs)  


if __name__ == "__main__":
    main()