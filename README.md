# CATCH: Cellular Analysis with Topology and Condensation Homology
Learning cellular hierarchy from scRNAseq data

Cells occupy a hierarchy of transcriptional identities which is difficult to study in an unbiased manner when perturbed by disease. To identify, characterize, and compare clusters of cells, we present CATCH, a coarse graining framework that learns the cellular hierarchy by applying a deep cascade of manifold-intrinsic diffusion filters. CATCH includes a suite of tools based on the connection we forge between topological data analysis and data diffusion geometry to identify salient levels of the hierarchy, automatically characterize clusters and rapidly compute differentially expressed genes between clusters of interest. When used in conjunction with MELD (https://github.com/KrishnaswamyLab/MELD), CATCH has been shown to identify rare popultions of pathogenic cells and create robust disease signatures.


![alt text](https://github.com/KrishnaswamyLab/CATCH/blob/main/Images/CATCH%20img1.png)

![alt text](https://github.com/KrishnaswamyLab/CATCH/blob/main/Images/CATCH%20img2.png)


To install use:

`pip install git+https://github.com/KrishnaswamyLab/CATCH`


Tutorial in the repo [here](https://github.com/KrishnaswamyLab/CATCH/blob/main/Tutorial/tutorial.ipynb).
