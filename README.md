# Adhesome Receptor Clustering is Accompanied by the Co-localization of the Associated Genes in the Cell Nucleus

In the paper "Adhesome Receptor Clustering is Accompanied by the Co-localization of the Associated Genes in the Cell Nucleus" (in review, 2022) by Cammarata, Uhler, and Shivashankar, we:

* Show that genes coding for focal adhesion proteins that are clustered on the plasma membrane of fibroblast cells are also co-localized in the cell nucleus
* Establish that these adhesome genes are transcriptionally co-regulated
* Find that Transcription Factors (TFs) targeting adhesome genes are also co-clustered with these adhesome genes, facilitating transcriptional reinforcement
* Cluster adhesome genes and their TFs to identify important interacting genomic regions
* Validate our findings using genome-wide Fluorescence In Situ Hybridization (FISH) data

![Alt text](GraphicalAbstract.png?raw=true "Title")

The subdirectories contained in the code directory can be used to reproduce results and figures in the article. Data pre-processing steps are implemented in the following subdirectories:

* processing1_adhesome: load and explore adhesome network data from [adhesome.org](http://adhesome.org/)
* processing2_TFs: scrape and explore TF-target data from [hTFtarget](http://bioinfo.life.hust.edu.cn/hTFtarget#!/)
* processing3_hic: process in situ Hi-C data in .hic format from [Rao et al. (2014)](https://www.cell.com/cell/fulltext/S0092-8674(14)01497-4?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867414014974%3Fshowall%3Dtrue) using juicer
* processing4_regulatory_marks: process regulatory marks (DNA accessibility, TF and histone ChIP-seq) from different sources (see Methods section of the article) including [targetfinder](https://github.com/shwhalen/targetfinder)

Analysis steps are implemented in the following subdirectories:

* analysis1_preliminaries: prepare adhesome data, determine gene and locus activity, generate and save useful data, prepare TF-target data
* analysis2_adhesome_HiC: show the proximity of adhesome genes in Hi-C data
* analysis3_adhesome_coregulation: show that adhesome genes are co-regulated using regulatory marks
* analysis4_adhesome_tfs: study the localization and the significance of genes coding for TFs that target adhesome genes
* analysis5_FISH: assess the proximity of adhesome genes in DNA-MERFISH data from [Su et al. (2020)](https://www.cell.com/cell/fulltext/S0092-8674(20)30940-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867420309405%3Fshowall%3Dtrue)
* analysis6_LAS: cluster adhesome genes in the cell nucleus using the Large Average Submatrices (LAS) algorithm

No data is included in this repository due to storage limitations. To replicate results in the article, download the relevant data from the references indicated in the links above and in the article (see the Methods section). Additional programs may be required to run the code in this repository:
* numpy
* pandas
* scipy
* sklearn
* xlmhg
* networkx
* pybedtools
* gseapy
* OmicsIntegrator
* matplotlib
* seaborn
* itertools
* pickle
* joblib
