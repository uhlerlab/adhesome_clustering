#!/bin/bash

# IF THE CODE IS NOT RUNNING, IT MIGHT BE AN ENVIRONMENT PROBLEM; RUN THE FOLLOWING LINES OUTSIDE OF THIS FILE!
# conda create --name python2_env python=2.7.15
# conda install numpy pandas
# conda activate python2_env

#### HistMod_script.sh
# Script to assess the share of TADs showing a significant H3K27me3/H3K36me3 log10-ratio for a given TAD partition.
# Input: see following arguments.
# Output:
# - Results_df_*.txt: the number of TADs, the binsize used for permutation and the share of TADs showing a significant H3K27me3/H3K36me3 log10-ratio
# - Density_lr_*.pdf: The density plot of H3K27me3/H3K36me3 log10-ratio in real and permuted scenarios 
#
# Refer to Zufferey & Tavernari et al. for details.
# 
# Contact daniele.tavernari@unil.ch for enquiries

###### INPUT ######
chromosome_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
resolution_list=(5000 10000 25000)
normalization_list=("NONE" "KR")
share=0.1 # Share of average TAD size to use as bin size for permutation test
nshuf=10 # Number of shufflings for permutation test
fdr_thresh=0.1 # FDR threshold
code_folder="/home/louiscam/projects/gpcr/code/processing3_hic/IMR90_hic_preprocessing/TAD-benchmarking-scripts"
arrowhead_folder="/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead"
histone_peaks_folder="${arrowhead_folder}/HistonePeaks_Folder"

for norm in "${normalization_list[@]}"
do
    
    for resol in "${resolution_list[@]}"
    do
    
        echo "Histone marks enrichment for all chromosomes for TADs at resolution ${resol} with ${norm} normalization"
          
        for chrom in "${chromosome_list[@]}"
        do
        
            TadsFile="${arrowhead_folder}/${norm}/${resol}/${chrom}/processed_for_enrichment_tads_list.csv" # Tab-separated file containing the list of TADs. Each line (no header) should represent a TAD, with genomic coordinates (chr, start, end)
            h27_fc="${histone_peaks_folder}/H3K27me3_peaks_ENCFF062LIE_chr${chrom}.bedGraph" # BedGraph file containing the fold change vs control of ChIP-seq tracks for H3K27me3 mark
            h36_fc="${histone_peaks_folder}/H3K36me3_peaks_ENCFF039DZS_chr${chrom}.bedGraph" # BedGraph file containing the fold change vs control of ChIP-seq tracks for H3K36me3 mark
            OutFolder="${arrowhead_folder}/${norm}/${resol}/${chrom}" # Folder where results should be saved
            
            chr=$(awk '{print $1}' ${TadsFile} | uniq | cut -c 4-)
            binsizes_file="${OutFolder}/binsizedf_chr${chr}_share${share}.txt"
            ./HistMod_computeBinsizes_script.R ${TadsFile} ${binsizes_file} ${share}

            while read binsize; do
                binnedFile="${OutFolder}/chr${chr}_binsize${binsize}.bed"
                ./HistMod_binChr.py -o ${binnedFile} -c ${chr} -b ${binsize}

                # H3K27me3
                filein_fc=${h27_fc}
                filein_fc_binned_intersections="${OutFolder}/h27_fc_intersections.bedGraph"
                h27_binned="${OutFolder}/h27_fc_binsize${binsize}.bedGraph"
                bedtools intersect -wao -a ${binnedFile} -b ${filein_fc} > ${filein_fc_binned_intersections}
                ./HistMod_aggregateBdg.R ${filein_fc_binned_intersections} ${h27_binned} ${binsize}

                # H3K36me3
                filein_fc=${h36_fc}
                filein_fc_binned_intersections="${OutFolder}/h36_fc_intersections.bedGraph"
                h36_binned="${OutFolder}/h36_fc_binsize${binsize}.bedGraph"
                bedtools intersect -wao -a ${binnedFile} -b ${filein_fc} > ${filein_fc_binned_intersections}
                ./HistMod_aggregateBdg.R ${filein_fc_binned_intersections} ${h36_binned} ${binsize}
            done <${binsizes_file}

            ./HistMod_FDRclassic_script.R ${TadsFile} ${share} ${nshuf} ${fdr_thresh} ${chr} ${OutFolder} ${h27_binned} ${h36_binned}

            cd ${OutFolder}
            rm binsize* chr* h27* h36*
            cd ${code_folder}

        done
        
        echo "Complete"
    
    done
    
done