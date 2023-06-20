# Script to perform hyperparameter tuning procedure
# Best to run as an array job using a HPC with argument param_index determing which row of parameter_list.csv is used
# Author: Ashray Gunjur

# import packages
import scipy, sklearn, imblearn, argparse, time, datetime, os, warnings
import pandas as pd
import numpy as np
import itertools as it
warnings.filterwarnings("ignore", category=UserWarning)

## pre-processing
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# balancers
from imblearn.over_sampling import RandomOverSampler

## classifier
from sklearn.ensemble import RandomForestClassifier

## cross-val / evaluating
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

## parse args
parser = argparse.ArgumentParser()
parser.add_argument('-p', "--param_index", help='Suffix for output file')
args = parser.parse_args()

# read in params (create_parameter_list.py MUST be run first)
param_list = pd.read_csv("parameter_list.csv", index_col=0)
feats, K = param_list[param_list.index==int(args.param_index)-1].values[0].tolist()
target = "R_vs_P"

# ------------------
# create datasets
# ------------------

# import data
df=pd.read_csv("supp_tables.xlsx", sheet_name = "1. metadata_and_clr_abundances")

# if using a subset of top strains, this needs to be defined
if K != "all":
    K = int(K)
    top_taxa = pd.read_excel("supp_tables.xlsx", sheet_name = "6. strain_importance")
    top_taxa.columns = ["Strain","mean_importance_score","std_importance_score"]

    # reorder
    top_taxa.sort_values(by='mean_importance_score', key=abs, ascending = False, inplace=True)

    # add t__ prefix
    top_taxa_list = [f"t__{s}" for s in top_taxa['Strain'].to_list()]

    # select top K
    top_K = top_taxa_list[0:K]

# filter to data needed
train = df.filter(regex='%s|%s' % (feats, target), axis =1).dropna()

# split training data into X and y

if K == "all":
    X = train.filter(regex='%s'%(feats), axis = 1)
if K != "all":
    X = train.filter(top_K, axis = 1)
y = train[target].astype('bool').values

# ---------------------
# set hyperparam grid
# ---------------------

clf = RandomForestClassifier(n_jobs=-1)
grid = {
        'estimator__criterion': ["gini","entropy"],
        'estimator__n_estimators':[int(x) for x in np.arange(20, 501, 10)],
        'estimator__max_features': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1],
        'estimator__max_depth': [2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'estimator__min_samples_split': [int(x) for x in np.arange(2,21,2)],
        'estimator__min_samples_leaf': [int(x) for x in np.arange(2,21,2)],
        'estimator__bootstrap': [True]
       }

# --------------
#  pipeline
# --------------

# define column types
numericals = X.select_dtypes(include=['int64', 'float64']).columns
categoricals = X.select_dtypes(include=['category','object', 'bool']).columns

# create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericals),
        ("cat", OneHotEncoder(drop = 'first', handle_unknown ='ignore'), categoricals),
    ]
)

# create pipeline
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("balancer", RandomOverSampler()),
    ("estimator", clf)
])


# -----------------
# hyperparam search fit
# -----------------

# define cv strategy:
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20)

# execute grid search:
grid_search = RandomizedSearchCV(estimator=pipe, 
                                 param_distributions=grid, 
                                 n_jobs=-1, 
                                 n_iter = 1000,
                                 cv=cv,
                                 scoring= 'roc_auc',
                                 refit= False,
                                 error_score=0)

grid_result = grid_search.fit(X, y)

# ----------------
# process scores
# ----------------

# get best results
cvdict = grid_result.cv_results_
cvdf = pd.DataFrame(cvdict)
scores = cvdf.loc[cvdf['rank_test_score'] == 1]

# add vars
scores.loc[:,'feats'], scores.loc[:,'classifier'], scores.loc[:,'target'], scores.loc[:,'K'] = feats, classifier, target, K

# ------------#
# save data  #
# ------------#

# create project title
project=f"hyperparams_{K}"

# create new folder
os.makedirs(f"{project}", exist_ok=True)

# create paths
scores_path = f"{project}/hyperparams-feats_{feats}-K_{K}.csv"

# save as csv
scores.to_csv(scores_path)
