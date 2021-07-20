########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from joblib import Parallel, delayed
import copy
import warnings
import time

# Scikit
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, check_cv, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin, clone  # ClassifierMixin

# ML
import xgboost as xgb
import lightgbm as lgbm
from itertools import product  # for GridSearchCV_xlgb


########################################################################################################################
# Parameter
########################################################################################################################

# Locations
dataloc = "../data/"
plotloc = "../output/"

# Number of cpus
n_jobs = 4

# Util
sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

# Other
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
manycol = np.delete(np.array(list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values()), dtype=object),
                    np.array([4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 26]))
colorblind = sns.color_palette("colorblind", as_cmap=True)
# sel = np.arange(50); fig, ax = plt.subplots(figsize=(5,15)); ax.barh(sel.astype("str"), 1, color=manycol[sel])

    
