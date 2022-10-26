README
===============================
NIHGCN:Predicting cancer drug response using parallel heterogeneous graph convolutional networks with neighborhood interactions
This document mainly introduces the python code of NIHGCN algorithm.

# Requirements
- pytorch==1.6.0
- tensorflow==2.3.1
- numpy==1.17.3+mkl
- scipy==1.4.1
- pandas==0.25.2
- scikit-learn=0.21.3
- pubchempy==1.0.4
- seaborn==0.10.0
- hickle==4.0.1
- keras==2.4.3

# Instructions
This project contains all the codes for NIHGCN and 8 comparison algorithms to experiment on the in vitro datasets (GDSC, CCLE) and in vivo datasets (PDX, TCGA), respectively.We only introduce the algorithm proposed in our paper, NIHGCN, and the introduction of other algorithms can be found in the corresponding paper.

# Model composition and meaning
NIHGCN is composed of common modules and experimental modules.

## Common module
- Data defines the data used by the model
	- GDSC
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- cell_exprs.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
		- null_mask.csv records the null values in the cell line-drug association matrix.
		- threshold.csv records the drug sensitivity threshold.
	- CCLE
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- cell_exprs.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
	- PDX
		- pdx_response.csv records the binary patient-drug association matrix.
		- pdx_exprs.csv records patient gene expression features.
		- pdx_null_mask.csv records the null values in the patient-drug association matrix.
		- drug_feature.csv records the fingerprint features of drugs.	
	- TCGA
		- patient_drug_binary.csv records the binary patient-drug association matrix.
		- tcga_exprs.csv records patient gene expression features.
		- tcga_null_mask.csv records the null values in the patient-drug association matrix.
		- drug_feature.csv records the fingerprint features of drugs.			
- load_data.py defines the data loading of the model.
- model.py defines the complete NIHGCN model.
- myutils.py defines the tool functions needed by the entire algorithm during its operation.
- sampler.py defines the sampling method of the model.

## Experimental module
- Internal performs the internal division verification experiment.
	- Random performs the random clearing cross-validation experiment.
	- New performs single row and single column clearing experiments.	
	- Single performs a single drug response prediction experiment.
	- Target performs targeted drug experiments.

- External performs the external validation experiments from in vitro to in vivo.

All *main*.py files can complete a single experiment. Because of the randomness of dividing test data and training data, we recorded the true value of the test data during the algorithm performance. Therefore, the output of the main file includes the true and predicted values of the test data that have been cross-validated many times. In the subsequent statistical analysis, we analyze the output of the main file. The myutils.py file contains all the tools needed for the performance and analysis of the entire experiment, such as the calculation of AUC, AUPRC,  ACC, F1 score, and MCC. All functions are developed using PyTorch and support CUDA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).

#PS
The file cell_exprs.csv for the ccle dataset locates at “ NIHGCN/Data/CCLE/cell_gene/”. It is too large to upload to github, so we divided it into several zip files. Simlarly, the file cell_exprs.csv for the GDSC dataset locates at “ NIHGCN/Data/GDSC/cell_gene/”

