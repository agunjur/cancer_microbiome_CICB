# Purpose: Script to create 'parameter_list.csv' for hyperparam search
# Author: Ashray Gunjur
# date: 2023-03-02

import itertools as it
import numpy as np
import pandas as pd

# set parameters

## featue glossary: t__ = sTrain, s__ = species, g__ = genus, f__ = family, c__ = clinical, e__ = tEchnical
feats = ["^t__","^s__","^g__","^f__","^c__","^t__|^c__","^s__|^c__","^g__|^c__","^f__|^c__",]
## define if using all, or subset of top features based on previously defined feature importance
K = "all"
all_combinations = list(it.product(feats, K))

# save
param_list = pd.DataFrame(all_combinations, columns=['feats','K'])
param_list.to_csv("parameter_list.csv")

