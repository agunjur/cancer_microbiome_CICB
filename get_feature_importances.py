# Purpose: get list of feature (e.g. Strain) importances, using TreeSHAP- averaged over N repeats
# Author: Ashray Gunjur
# Date: 2023-03-02

import scipy, sklearn, imblearn, os, ast, random, shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

## pre-processing
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.fixes import loguniform
from sklearn.feature_selection import SelectFromModel

# balancers
from imblearn.over_sampling import RandomOverSampler

## estimators
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

## cross-val / evaluating
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut, GroupShuffleSplit, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

# settings
feats = '^t__'
target = 'R_vs_P'

# load data
df=pd.read_excel("supp_tables.xlsx", sheet_name = "1. metadata_and_clr_abundances")

train = df.filter(regex='%s|%s' % (feats, target), axis =1).dropna()
X = train.filter(regex='%s'% (feats), axis = 1)
y = train[target].astype('bool').values

# load hyperparams (from concatenating results of hyperparam_tuning.py, or pre-made in supp_tables.xlsx)
hp_all = pd.read_excel("supp_tables.xlsx", sheet_name = "3. hyperparam_tuning_all")
string = hp_all[hp_all['feats']=='^t__']['params'].reset_index(drop=True)
hp = ast.literal_eval(string[0])

# set up classifier:
clf = RandomForestClassifier(n_jobs=-1,
    n_estimators= hp['estimator__n_estimators'],
    max_features= hp['estimator__max_features'],
    max_depth= hp['estimator__max_depth'],
    min_samples_split= hp['estimator__min_samples_split'],
    min_samples_leaf= hp['estimator__min_samples_leaf'],
    bootstrap= hp['estimator__bootstrap'], 
    criterion= hp['estimator__criterion']
                            )
    
# --------------
#  pipeline
# --------------

# define column types
numericals = X.select_dtypes(include=['int64', 'float64']).columns
categoricals = X.select_dtypes(include=['category','object', 'bool']).columns

# create preprocessor
num_transformer = make_pipeline(StandardScaler())
cat_transformer = make_pipeline(OneHotEncoder(drop = 'first', handle_unknown ='ignore'))
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numericals),
        ("cat", cat_transformer, categoricals),
    ]
)

# create pipeline
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("balancer", RandomOverSampler()),
    ("estimator", clf)
])
    
# reset lists
impscores_list = []

#------#
# LOOP #
#------#
N = 1000
for i in range(N):
    
    # fit pipe
    pipe.fit(X,y)
    
    # get feature names:
    feature_names = pipe[:-1].get_feature_names_out()
    def trim_names(s):
        return s[8:]
    trimmed_names = [trim_names(s) for s in feature_names]
    
    #-----------------------#
    # get importance scores #
    #-----------------------#
    
    shap.initjs() 

    #apply the preprocessing steps to whole of X
    X_pp = pipe['preprocessor'].fit_transform(X,y)

    #get Shap values for positive class [1]
    shap_values = shap.TreeExplainer(pipe['estimator']).shap_values(X_pp, y)[1]

    # get mean absolute shap values (masv) per feature
    masv = np.abs(shap_values).mean(axis=0)

    # store as df:
    d = pd.DataFrame({'feature': trimmed_names, 'masv': masv}).set_index('feature')

    # get direction of feature importance using linear regression of X_pp vs shap values
    Xbal_df = pd.DataFrame(X_pp)
    Xbal_df.columns = trimmed_names

    sv_df = pd.DataFrame(shap_values)
    sv_df.columns = trimmed_names

    coefs = {}
    for i in trimmed_names:
        Xc = Xbal_df[i].to_numpy().reshape(-1, 1)
        yc = sv_df[i]
        lm= LinearRegression().fit(Xc,yc)
        coef = lm.coef_[0]
        coefs[i]=coef

    coefs_df = pd.DataFrame.from_dict(coefs, orient="index")
    coefs_df.columns = ["lm_coef"]

    # merge coef back into d
    d = pd.merge(d, coefs_df, left_index=True, right_index=True)

    # change sign of mean impscore based on coef sign
    d['coef_sign'] = np.where(d['lm_coef'] > 0, 1, -1)
    d['imp'] = d['masv']*d['coef_sign']

    # select only importance score
    impscores_df = d[['imp']]

    # append
    impscores_list.append(impscores_df)

# get importance scores (mean and SD)

impscores_df = pd.concat(impscores_list, axis=1).fillna(0)
mean_impscore = impscores_df.mean(axis = 1)
std_impscore = impscores_df.std(axis = 1)

impscores_df['mean_importance_score']=mean_impscore
impscores_df['std_importance_score']=std_impscore
impscores_df = impscores_df[['mean_importance_score','std_importance_score']]

# save
impscores_df.to_csv(f"feats-{feats}_importance.csv")