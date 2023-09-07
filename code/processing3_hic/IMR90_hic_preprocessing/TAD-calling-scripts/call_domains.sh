#!/bin/bash

chromosome_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
n_chromosomes=22
matrix_size=2000
resolution_list=(5000 10000 25000)
normalization_list=("NONE" "KR")
hic_file="/home/louiscam/projects/gpcr/data/hic_data/IMR90/GSE63525_IMR90_combined_30.hic"

for norm in "${normalization_list[@]}"
do
    
    mkdir "/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead/${norm}"
    
    for resol in "${resolution_list[@]}"
    do
    
        mkdir "/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead/${norm}/${resol}"
        
        for chrom in "${chromosome_list[@]}"
        do
    
            echo "TAD calling for chromosome ${chrom} with ${norm} normalization at ${resol} resolution"
            
            mkdir "/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead/${norm}/${resol}/${chrom}"
            output_folder="/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead/${norm}/${resol}/${chrom}"

            java -Xms1g -Xmx10g -jar ../juicer_tools.jar arrowhead -c ${chrom} -m ${matrix_size} -r ${resol} -k ${norm} --threads ${n_chromosomes} ${hic_file} ${output_folder}

            echo "Completed"
            
        done
    done
done
