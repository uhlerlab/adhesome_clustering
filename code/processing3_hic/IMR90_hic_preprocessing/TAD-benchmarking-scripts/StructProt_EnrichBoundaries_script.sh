#!/bin/bash

chromosome_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
resolution_list=(5000 10000 25000)
normalization_list=("NONE" "KR")
arrowhead_folder="/home/louiscam/projects/gpcr/save/processed_hic_domains/arrowhead"
ChrSizes_file="${arrowhead_folder}/chrom_hg19.sizes_arrowhead.csv"
ProteinPeaks_Folder="${arrowhead_folder}/ProteinPeaks_Folder/"

for norm in "${normalization_list[@]}"
do
        
    for resol in "${resolution_list[@]}"
    do
        
        resolution_kb=$((${resol} / 1000))
        echo "Enrichment of structural proteins at TAD boundaries at ${resolution_kb}kb resolution with ${norm} normalization"
        
        for chrom in "${chromosome_list[@]}"
        do
    
            TadsFile="${arrowhead_folder}/${norm}/${resol}/${chrom}/processed_for_enrichment_tads_list.csv"
            OutFolder="${arrowhead_folder}/${norm}/${resol}/${chrom}/"
            
            Rscript ./StructProt_EnrichBoundaries_script.R ${TadsFile} ${resolution_kb} ${OutFolder} ${ChrSizes_file} ${ProteinPeaks_Folder}
    
        done
        
        echo "Completed"
        
    done
done
