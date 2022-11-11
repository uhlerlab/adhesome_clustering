#!/bin/bash
resol=250000
MAPQ=30
num_chromosomes=22
for i in $(seq 1 22)
do
    for j in $(seq ${i} 22)
    do
        java -jar ../juicer_tools.jar dump observed NONE /home/louiscam/projects/gpcr/data/hic_data/GM12878/GSE63525_GM12878_insitu_primary_and_replicate_combined_${MAPQ}.hic ${i} ${j} BP ${resol} /home/louiscam/projects/gpcr/save/processed_hic_data/processed_hic_data_GM12878/normalized_BP${resol}/chr${i}_chr${j}_NONE.txt
        java -jar ../juicer_tools.jar dump observed KR /home/louiscam/projects/gpcr/data/hic_data/GM12878/GSE63525_GM12878_insitu_primary_and_replicate_combined_${MAPQ}.hic ${i} ${j} BP ${resol} /home/louiscam/projects/gpcr/save/processed_hic_data/processed_hic_data_GM12878/normalized_BP${resol}/chr${i}_chr${j}_KR.txt
        java -jar ../juicer_tools.jar dump observed INTER_KR /home/louiscam/projects/gpcr/data/hic_data/GM12878/GSE63525_GM12878_insitu_primary_and_replicate_combined_${MAPQ}.hic ${i} ${j} BP ${resol} /home/louiscam/projects/gpcr/save/processed_hic_data/processed_hic_data_GM12878/normalized_BP${resol}/chr${i}_chr${j}_INTERKR.txt
        java -jar ../juicer_tools.jar dump observed GW_KR /home/louiscam/projects/gpcr/data/hic_data/GM12878/GSE63525_GM12878_insitu_primary_and_replicate_combined_${MAPQ}.hic ${i} ${j} BP ${resol} /home/louiscam/projects/gpcr/save/processed_hic_data/processed_hic_data_GM12878/normalized_BP${resol}/chr${i}_chr${j}_GWKR.txt
        
        echo "Normalize Hi-C for chromosome pair ${i}, ${j} (NONE, KR, INTER_KR, GW_KR)"
    done
done