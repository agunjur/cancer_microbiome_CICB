# cancer_microbiome_CICB

Machine learning code for paper *A gut microbial signature for combination immune checkpoint blockade across cancer types*.

These scripts provided are intended to be used with the file "Supp_tables.xlsx" included with the manuscript, which should be placed in this directory. They will allow you to understand and replicate the supervised machine learning analysis of the CA209-538 cohort microbiome + clinical data (relevant to Figure 2 and 3 of the manuscript).

Specifically:
- *create_parameter_list.ipynb* is a jupyter notebook that should be run first, to create the file "parameter_list.csv". It lists all combinations of 'feats' (the feature sets to be selected for model training and testing, denoted by the prefix (e.g. "t__" denotes strain-level clr-abundances)), 'target' (the target binary variable, e.g. 'R_vs_PD') and 'K' (either 'all' for all features, or an integer to select the top K strains based on the pre-defined 'strain_importance' supplementary table sheet; only to be used if feats = "^t__").

- *hyperparam_tuning.py* is a python script ideally run as an array-job on a high-performance compute cluster. It iterates over 'parameter_list.csv', selects relevant features and target, and performs 1000 randomly-selected iterations of 20-repeat stratified 5-fold cross validation over a large hyperparameter space. A csv file of the best hyperparameters (based on AUC score), and the cross-validation scores (mean, standard deviation, and each score for the 100-folds) is output for each feature-K combination, that can be concatenated to reproduce supp. table sheet "hyperparam_tuning_all".

- *figure_two.ipynb* is a juypter notebook to create ROC curves across histology cohort subgroups, either using leave-one-group-out cross-validation (Figure 2C), or out-of-bag predictions.

- *get_feature_importances.py* is a python script to determine the TreeSHAP feature importances of a tuned classifier trained on the whole CA209-538 cohort, using the specified features and target variable (default feats = '^t__', target = 'R_vs_PD'). Importance direction is inferred using a linear model of feature vs shap values. To account for model stochasticity, the procedure is repeated 1000x and averaged.
