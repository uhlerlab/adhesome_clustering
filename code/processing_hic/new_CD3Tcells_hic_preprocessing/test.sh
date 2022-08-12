#!/bin/bash
resol=250000

# MAPQ=30
java -Xmx5g -jar ../hic_emt.jar excise -r ${resol} /home/louiscam/projects/gpcr/data/hic_data/IMR90/GSE63525_IMR90_combined_30.hic /home/louiscam/projects/gpcr/data/hic_data/test
