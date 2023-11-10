# cancer_microbiome_CICB

Code for paper *A gut microbial signature for combination immune checkpoint blockade across cancer types*

The scripts provided are intended to be used with the file "supp_tables_v2.xlsx" included with the manuscript, which should be placed in this directory. They will allow you to understand and replicate the supervised machine learning analysis of pre-processed (centred-log-ratio transformed) gut microbiota abundances, as well as associated clinical metdata.

Specifically:
- *create_parameter_list.py* should be run first, to create a list of feature sets to iterate over. The 'K' parameter can be modified to an integer, which will select the top K strains (features starting with t__ must be selected).
- *hyperparam_tuning.py* should be run as an array-job on a high-performance compute cluster, to iterate over the list made by "create_parameter_list.py", and create a list of csv files with relevant scores for the top performing hyperparameters for each feature set, that can then be concatenated into a single csv file. By default, this will reproduce the provided supp. table sheet "3. hyperparam_tuning_all". 
To reproduce the supp. table sheet "4. hyperaparm_tuning_top22", the * *create_parameter_list.py* * script should be edited to have feats = "^t__", and K = 22.
- *figure_two.ipynb* is a juypter notebook to create the receiver operating characteristic curve figure panels (Figure 2C and Ext Figure 2A)
