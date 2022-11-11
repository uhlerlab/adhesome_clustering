import json
import os
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools

def parse_config(config_fn):
    config = json.load(open(config_fn))
    if 'LAS_OUT_DIR' in config:
        create_dir(config['LAS_OUT_DIR'])
    return config

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_chrom_size(chrom, directory):
        # get chrom size from known file
        lengths_filename = directory + 'chrom_hg19.sizes'
        with open(lengths_filename) as f:
                for line in f.readlines():
                        d = line.rstrip().split('\t')
                        if (d[0] == ('chr' + str(chrom))):
                                chrom_size = int(d[1])
        return chrom_size
