########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages --------------------------------------------------------------------------

# General
import numpy as np 
import pandas as pd 
import swifter
import matplotlib.pyplot as plt
import pickle
from importlib import reload
import time
import hmsPM.plotting as hms_plot

# Special
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit, PredefinedSplit

# Custom functions and classes
#from tmp import my_utils as my
import my_utils as my


# --- Parameter ---------------------------------------------------------------------------------------------------

# Plot 
plot = True
%matplotlib
#%matplotlib qt / %matplotlib inline  # activate standard/inline window
plt.ioff()  # / plt.ion()  # stop/start standard window
#plt.plot(1, 1)

# Specific parameters
TARGET_TYPES = ["REGR", "CLASS"]
d_targets = {"REGR": "payment_cap_log", "CLASS": "claims_bin"}


########################################################################################################################
# ETL
########################################################################################################################

# --- Read data and adapt to be more readable --------------------------------------------------------------------------

# Read 
df_orig = (pd.read_csv(my.dataloc + "fremotor1prem0304a.csv")
           .merge(pd.read_csv(my.dataloc + "fremotor1freq0304a.csv"), how="left", on=["IDpol", "Year"])
           .merge(pd.read_csv(my.dataloc + "fremotor1sev0304a.csv", parse_dates=["OccurDate"])
                  .assign(Year=lambda x: x["OccurDate"].dt.year), how="left", on=["IDpol", "Year"]))
#df_orig.iloc[0, :].T.to_csv(my.dataloc + "tmp.csv", sep=";")
#df_num = pd.DataFrame(df_orig.describe().T)
#df_cat = my.value_counts(df_orig, topn=10).T
df_orig.OccurDate.describe()

'''
df_tmp = pd.read_csv(my.dataloc + "fremotor1sev0304a.csv")
df_tmp.drop_duplicates(subset=["IDpol", "OccurDate"])
df_orig = (pd.concat([pd.read_csv(my.dataloc + "fremotor1prem0304a.csv"),
                      pd.read_csv(my.dataloc + "fremotor1prem0304b.csv"),
                      pd.read_csv(my.dataloc + "fremotor1prem0304c.csv")])
           .merge(pd.concat([pd.read_csv(my.dataloc + "fremotor1freq0304a.csv"),
                             pd.read_csv(my.dataloc + "fremotor1freq0304b.csv"),
                             pd.read_csv(my.dataloc + "fremotor1freq0304c.csv")]),
                  how="left", on=["IDpol", "Year"])
           .merge((pd.concat([pd.read_csv(my.dataloc + "fremotor1sev0304a.csv", parse_dates=["OccurDate"]),
                              pd.read_csv(my.dataloc + "fremotor1sev0304b.csv", parse_dates=["OccurDate"]),
                              pd.read_csv(my.dataloc + "fremotor1sev0304c.csv", parse_dates=["OccurDate"])])
                   .assign(Year=lambda x: x["OccurDate"].dt.year)),
                  how="left", on=["IDpol", "Year"])
           .drop_duplicates(keep="first").reset_index(drop=True))
'''
# Create some artifacts
df_orig = df_orig.assign(VehAge=lambda x: np.where(np.random.randint(0, 10, len(df_orig)) == 0, np.nan, x["VehAge"]))

# Add targets
df_orig = df_orig.assign(claims=lambda x: x[["Damage", "Fire", "Other", "Theft", "TPL", "Windscreen"]].sum(axis=1),
                         claims_bin=lambda x: np.where(x["claims"] > 0, 1, 0),
                         claims_bin_weight=lambda x: np.where(x["claims"] > 0, x["claims"], 1),
                         severity=lambda x: x["Payment"] / x["claims"],
                         excess_flag=lambda x: np.where(x["Payment"] > 2500, "Y", "N"),
                         payment_cap=lambda x: np.where(x["excess_flag"] == "N", x["Payment"], np.nan),
                         severity_cap=lambda x: x["payment_cap"] / x["claims"],
                         payment_cap_log=lambda x: np.log(1+x["payment_cap"]),
                         severity_cap_log=lambda x: np.log(1+x["severity_cap"]),
                         payment_ex=lambda x: np.where(x["excess_flag"] == "Y", x["Payment"], np.nan),
                         severity_ex=lambda x: x["payment_ex"] / x["claims"])

# Adapt categories for sorting
df_orig = df_orig.assign(VehPower=lambda x: x["VehPower"].str.slice(0, 1) + x["VehPower"].str.slice(1).str.zfill(2))
df_orig = df_orig.assign(Area=lambda x: x["Area"].str.slice(0, 1) + x["Area"].str.slice(1).str.zfill(2))

# Filter on TPL
df = df_orig.query("claims == TPL").reset_index(drop=True)

'''
# Check some stuff
df.dtypes
df.describe()
my.value_counts(df, dtypes=["object"]).T
df["Payment"].describe()
(df["Payment"]==0).sum()
df["Payment"].value_counts()

%matplotlib inline
fig, ax = plt.subplots(3, 3, figsize=(15,15))
df["claims"].plot.hist(bins=50, ax=ax[0,0], title="claims")
np.log(1+df["Payment"]).plot.hist(bins=50, ax=ax[0,1], title="Payment log")
np.log(1+df["Payment"]).hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[0,2])
df["payment_cap"].plot.hist(bins=50, ax=ax[1, 0], title="cap")
np.log(1+df["payment_cap"]).plot.hist(bins=50, ax=ax[1, 1], title="cap log")
np.log(1+df["payment_cap"]).hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[1, 2])
df["payment_ex"].plot.hist(bins=50, ax=ax[2, 0], title="ex")
np.log(-2500+df["payment_ex"]).plot.hist(bins=50, ax=ax[2, 1], title="ex log")
np.log(-2500+df["payment_ex"]).hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[2, 2])
%matplotlib
'''


# --- Read metadata (Project specific) ---------------------------------------------------------------------------------

df_meta = pd.read_excel(my.dataloc + "datamodel_claims.xlsx", header=1, engine='openpyxl')

# Check
print(my.diff(df.columns, df_meta["variable"]))
print(my.diff(df_meta["variable"], df.columns))

# Filter on "ready"
df_meta_sub = df_meta.query("status in ['ready']").reset_index()


# --- Feature engineering ----------------------------------------------------------------------------------------------
'''
df["day_of_month"] = df['dteday'].dt.day.astype("str").str.zfill(2)

# Check again
print(my.diff(df_meta_sub["variable"], df.columns))
'''

# --- Define train/test/util-fold --------------------------------------------------------------------------------------

df["fold"] = np.random.permutation(
    pd.qcut(np.arange(len(df)), q=[0, 0.1, 0.8, 1], labels=["util", "train", "test"]))
#df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data



########################################################################################################################
# Numeric variables: Explore and adapt
########################################################################################################################

# --- Define numeric covariates ----------------------------------------------------------------------------------------

nume = df_meta_sub.loc[df_meta_sub["type"] == "nume", "variable"]
df[nume] = df[nume].apply(lambda x: pd.to_numeric(x))
df[nume].describe()


# --- Create nominal variables for all numeric variables (for linear models)  -----------------------------------------

df[nume + "_BINNED"] = (df[nume].swifter.apply(lambda x: (pd.qcut(x, 5, duplicates="drop")))
                        .apply(lambda x: (("q" + x.cat.codes.astype("str") + " " + x.astype("str")))))

# Convert missings to own level ("(Missing)")
df[nume + "_BINNED"] = df[nume + "_BINNED"].fillna("(Missing)")
print(my.value_counts(df[nume + "_BINNED"], 6))

# Get binned variables with just 1 bin (removed later)
onebin = (nume + "_BINNED")[(df[nume + "_BINNED"].nunique() == 1).values]
print(onebin)


# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove covariates with too many missings
misspct = df[nume].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending=False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
nume = my.diff(nume, remove)  # adapt metadata

# Check for outliers and skewness
df[nume].describe()
start = time.time()
for TARGET_TYPE in TARGET_TYPES:
    if plot:
        distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12,
                                                                     show_regplot=True)
                            .plot(features=df[nume],
                                  target=df[d_targets[TARGET_TYPE]],
                                  file_path=my.plotloc + "1__distr_nume_orig__" + TARGET_TYPE + ".pdf"))
    print(time.time() - start)
    
# Winsorize (hint: plot again before deciding for log-trafo)
df[nume] = my.Winsorize(lower_quantile=0, upper_quantile=0.99).fit_transform(df[nume])

# Log-Transform
tolog = np.array([], dtype="object")
if len(tolog):
    df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
    nume = np.where(np.isin(nume, tolog), nume + "_LOG_", nume)  # adapt metadata (keep order)
    df.rename(columns=dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace=True)  # adapt binned version


# --- Final variable information ---------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE = "REGR"
    
    # Univariate variable performances
    varperf_nume = df[np.append(nume, nume + "_BINNED")].swifter.apply(lambda x: (
        my.variable_performance(x, df[d_targets[TARGET_TYPE]],
                                splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                scorer=my.d_scoring[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" else "auc"])))
    print(varperf_nume.sort_values(ascending=False))
    
    # Plot
    if plot:
        distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(show_regplot=True,
                                                                     n_rows=2, n_cols=2, w=12, h=8)
                            .plot(features=df[np.column_stack((nume, nume + "_BINNED")).ravel()],
                                  target=df[d_targets[TARGET_TYPE]],
                                  varimps=varperf_nume.round(2),
                                  file_path=my.plotloc + "1__distr_nume__" + TARGET_TYPE + ".pdf"))


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
nume = my.diff(nume, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[nume].describe()
corr_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=8, h=6)
             .plot(features=df[nume], file_path=my.plotloc + "1__corr_nume.pdf"))
remove = ["xxx"]
nume = my.diff(nume, remove)


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varperf_nume_fold = df[nume].swifter.apply(lambda x: my.variable_performance(x, df["fold"],
                                                                             splitter=my.InSampleSplit(),
                                                                             scorer=my.d_scoring["CLASS"]["auc"]))


# Plot: only variables with with highest importance
nume_toprint = varperf_nume_fold[varperf_nume_fold > 0.53].index.values
if len(nume_toprint):
    if plot:
        distr_nume_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(show_regplot=True,
                                                                             n_rows=2, n_cols=3, w=18, h=12)
                                    .plot(features=df[nume_toprint],
                                          target=df["fold"],
                                          varimps=varperf_nume_fold,
                                          file_path=my.plotloc + "1__distr_nume_folddep.pdf"))


# --- Missing indicator and imputation (must be done at the end of all processing)--------------------------------------

miss = nume[df[nume].isnull().any().values]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
np.random.seed(123)
df[miss] = SimpleImputer(strategy="median").fit_transform(df[miss])
df[miss].isnull().sum()



########################################################################################################################
# Categorical  variables: Explore and adapt
########################################################################################################################

# --- Define categorical covariates ------------------------------------------------------------------------------------

# Categorical variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("str")
df[cate].describe()


# --- Handling factor values -------------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)").replace("nan", "(Missing)")
#df[cate] = df[cate].astype("object")
df[cate].describe()

# Create ordinal/binary-encoded features
ordi = np.array(["LicenceNb"], dtype="object")
df[ordi + "_ENCODED"] = df[ordi].apply(lambda x: pd.to_numeric(x))  # ordinal
yesno = "MISS_" + miss
df[yesno + "_ENCODED"] = df[yesno].apply(lambda x: x.map({"No": 0, "Yes": 1}))  # binary

# Create target-encoded features for nominal variables
nomi = my.diff(cate, np.concatenate([ordi, yesno]))
df_util = df.query("fold == 'util'").reset_index(drop=True)
df[nomi + "_ENCODED"] = target_encoder.TargetEncoder().fit(df_util[nomi], df_util["claims_bin"]).transform(df[nomi])
#df = df.query("fold != 'util'").reset_index(drop=True)  # remove utility data

# Get "too many members" columns and lump levels
topn_toomany = 10
levinfo = df[cate].nunique().sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = my.diff(toomany, ["hr", "mnth", "weekday"])  # set exception for important variables
if len(toomany):
    df[toomany] = my.Collapse(n_top=topn_toomany).fit_transform(df[toomany])


# --- Final variable information ---------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE="REGR"

    # Univariate variable importance
    #varperf_cate = my.variable_performance(df[np.append(cate, ["MISS_" + miss])], df["cnt_" + TARGET_TYPE],
    #                                       ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)).round(2)
    varperf_cate = df[np.append(cate, ["MISS_" + miss])].swifter.apply(lambda x: (
        my.variable_performance(x, df[d_targets[TARGET_TYPE]],
                                splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                scorer=my.d_scoring[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" else "auc"])))
    print(varperf_cate.sort_values(ascending=False))

    # Check
    if plot:
        distr_cate_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[np.append(cate, ["MISS_" + miss])],
                            #.plot(features=df[cate[:3]],
                                  target=df[d_targets[TARGET_TYPE]],
                                  varimps=varperf_cate.round(2),
                                  file_path=my.plotloc + "1__distr_cate__" + TARGET_TYPE + ".pdf"))


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage variables
cate = my.diff(cate, ["xxx"])
toomany = my.diff(toomany, ["xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
corr_cate_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=12, h=12)
                  .plot(features=df[np.append(cate, ["MISS_" + miss])],
                        file_path=my.plotloc + "1__corr_cate.pdf"))


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varperf_cate_fold = df[np.append(cate, ["MISS_" + miss])].swifter.apply(lambda x: (
    my.variable_performance(x, df["fold"],
                            splitter=my.InSampleSplit(),
                            scorer=my.d_scoring["CLASS"]["auc"])))

# Plot: only variables with with highest importance
cate_toprint = varperf_cate_fold[varperf_cate_fold > 0.52].index.values
if len(nume_toprint):
    if plot:
        distr_cate_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                                    .plot(features=df[cate_toprint],
                                          target=df["fold"],
                                          varimps=varperf_cate_fold,
                                          file_path=my.plotloc + "1__distr_cate_folddep.pdf"))



########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Add numeric target -----------------------------------------------------------------------------------------------------
#df["cnt_REGR_num"] = df["cnt_REGR"]
#df["cnt_CLASS_num"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
#df["cnt_MULTICLASS_num"] = df["cnt_MULTICLASS"].str.slice(0, 1).astype("int")


# --- Define final features --------------------------------------------------------------------------------------------

# Standard: for all algorithms
nume_standard = np.append(nume, toomany + "_ENCODED")
cate_standard = np.append(cate, "MISS_" + miss)

# Binned: for Lasso
cate_binned = np.append(my.diff(nume + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
nume_encoded = np.concatenate([nume, cate + "_ENCODED", "MISS_" + miss + "_ENCODED"])

# Check
all_features = np.unique(np.concatenate([nume_standard, cate_standard, cate_binned, nume_encoded]))
my.diff(all_features, df.columns.values.tolist())
my.diff(df.columns.values.tolist(), all_features)


# --- Remove burned data -----------------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop=True)


# --- Save image -------------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(my.dataloc + "1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "nume_standard": nume_standard,
                 "cate_standard": cate_standard,
                 "cate_binned": cate_binned,
                 "nume_encoded": nume_encoded},
                file)
