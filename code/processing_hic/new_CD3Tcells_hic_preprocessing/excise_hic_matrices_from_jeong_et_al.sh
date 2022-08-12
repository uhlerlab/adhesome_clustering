#!/bin/bash
resol=250000

# MAPQ=0
java -Xmx5g -jar ../hic_emt.jar excise -r ${resol} https://s3.amazonaws.com/hicfiles/external/goodell/tcell.hic /home/louiscam/projects/gpcr/data/hic_data/CD3T

# MAPQ=30
java -Xmx5g -jar ../hic_emt.jar excise -r ${resol} https://s3.amazonaws.com/hicfiles/external/goodell/tcell_30.hic /home/louiscam/projects/gpcr/data/hic_data/CD3T
