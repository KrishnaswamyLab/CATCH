# CATCH: Cellular Analysis with Topology and Condensation Homology
Learning cellular hierarchy from scRNAseq data

Cells occupy a hierarchy of transcriptional identities which is difficult to study in an unbiased manner when perturbed by disease. To identify, characterize, and compare clusters of cells, we present CATCH, a coarse graining framework that learns the cellular hierarchy by applying a deep cascade of manifold-intrinsic diffusion filters. CATCH includes a suite of tools based on the connection we forge between topological data analysis and data diffusion geometry to identify salient levels of the hierarchy, automatically characterize clusters and rapidly compute differentially expressed genes between clusters of interest. When used in conjunction with MELD (https://github.com/KrishnaswamyLab/MELD), CATCH has been shown to identify rare populations of pathogenic cells and create robust disease signatures.

# Overview of Algorithm:

The key to thoroughly identifying and characterizing populations of cells affected by disease across granularities lies in the accurate computation of the cellular hierarchy. Current hierarchical clustering approaches applied to single cell analysis enforce global granularity constraints and provide only a few salient levels at which cellular groups can be found. This not only limits the discovery of rare disease-associated populations, but also requires computationally expensive differential expression analysis tools that produce diluted signatures of disease from unrefined clusters of cells. To address these limitations, we developed a novel topologically-inspired machine learning suite of tools called Cellular Analysis with Topology and Condensation Homology (CATCH). 

At the center of this framework is diffusion condensation, a recently proposed data-diffusion based dynamic process for continuous graining of data through a deep cascade of graph diffusion filters. The algorithm iteratively pull points towards the weighted average of their graph diffusion neighbors, slowly eliminating variation until all data points converge:

![alt text](https://github.com/KrishnaswamyLab/CATCH/blob/main/Images/CATCH%20img1.png)


To aid in single cell analysis tasks, we build a suite of tools around this coarse graining process:
1. Visual summarization of condensation process through embedding of condensation homology (i);
2. Identification of stable granularities for downstream analysis with topological activity analysis (ii);
3. Integration with single-cell enrichment analysis tool MELD (Burkhardt et al. 2021) to identify disease-enriched populations of cells;
4. Efficient comparison of clusters of cells via condensed transport to identify differentially expressed genes.

![alt text](https://github.com/KrishnaswamyLab/CATCH/blob/main/Images/CATCH%20img2.png)

# Getting Started

To install please use:

`pip install git+https://github.com/KrishnaswamyLab/CATCH`

For overview of functionality, please review our [CATCH tutorial](https://github.com/KrishnaswamyLab/CATCH/blob/main/Tutorial/tutorial.ipynb).
