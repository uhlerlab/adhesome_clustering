#!/bin/bash
resol=250000
MAPQ=30

java -Xmx5g -jar ../hic_emt.jar excise -r ${resol} /home/louiscam/projects/gpcr/data/hic_data/CD3T/GSM2861709_HIC1255_aligned_inter_${MAPQ}.hic /home/louiscam/projects/gpcr/data/hic_data/CD3T
