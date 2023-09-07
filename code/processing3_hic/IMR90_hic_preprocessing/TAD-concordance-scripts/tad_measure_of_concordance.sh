#!/bin/bash

chromosome_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
resolution_list=(5000 10000 25000)
normalization_list=("NONE" "KR")
arrowhead_folder="/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead"
OutFolder="${arrowhead_folder}/TAD_concordance_results"
chrSize_file="/home/louiscam/projects/gpcr/data/genome_data/chrom_hg19.sizes"

for chrom in "${chromosome_list[@]}"; do
    
    echo "Compute MoC between all TAD lists for chromosome ${chrom}"

    for norm1 in "${normalization_list[@]}"; do
        for resol1 in "${resolution_list[@]}"; do
            for norm2 in "${normalization_list[@]}"; do
                for resol2 in "${resolution_list[@]}"; do
                    
                    file1="${arrowhead_folder}/${norm1}/${resol1}/${chrom}/processed_for_enrichment_tads_list.bed"
                    file2="${arrowhead_folder}/${norm2}/${resol2}/${chrom}/processed_for_enrichment_tads_list.bed"
                    Rscript ./tad_measure_of_concordance.R ${file1} ${file2} ${chrSize_file} ${chrom} ${OutFolder}
                    
                done
            done
        done
    done

    echo "Complete"

done
