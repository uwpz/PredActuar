########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ------------------------------------------------------------------------------------

# General
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from importlib import reload 
import hmsPM.plotting as hms_plot

# Special
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgbm
import shap

# Custom functions and classes
import settings as sett
sys.path.append("../../PredAna/code")
import up_utils as uu
import up_plots as up


# --- Parameter --------------------------------------------------------------------------

# Main parameter
gbm = "lgbm"
target_name = "claims"  # claims, severity_cap, claims_bin, severity_cap_log
if target_name == "claims":
    TARGET_TYPE = "REGR"
    objective = "count:poisson" if gbm == "xgb" else "poisson"
    estimator = xgb.XGBRegressor if gbm == "xgb" else lgbm.LGBMRegressor
elif target_name == "claims_bin":
    TARGET_TYPE = "CLASS"
    objective = "binary:logistic" if gbm == "xgb" else "binary"
    estimator = xgb.XGBClassifier if gbm == "xgb" else lgbm.LGBMClassifier
elif target_name == "severity_cap":
    TARGET_TYPE = "REGR"
    objective = "reg:squarederror" if gbm == "xgb" else "rmse"
    estimator = xgb.XGBRegressor if gbm == "xgb" else lgbm.LGBMRegressor
elif target_name == "severity_cap_log":
    TARGET_TYPE = "REGR"
    objective = "reg:squarederror" if gbm == "xgb" else "rmse"
    estimator = xgb.XGBRegressor if gbm == "xgb" else lgbm.LGBMRegressor
elif target_name == "severity_cap_TOLOG":
    TARGET_TYPE = "REGR"
    objective = "reg:squarederror" if gbm == "xgb" else "rmse"
    estimator = xgb.XGBRegressor if gbm == "xgb" else lgbm.LGBMRegressor

else:
    raise ValueError('Wrong target')
metric = "spear" if TARGET_TYPE == "REGR" else "auc"
scoring = uu.d_scoring[TARGET_TYPE]
id_name = "IDpol"

# Plot
plot = False
%matplotlib
plt.ioff()
#%matplotlib {qt} / %matplotlib inline  # activate standard/inline window
#plt.ioff() / plt.ion()  # stop/start standard window
#plt.plot(1, 1)

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(sett.dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")
nume = nume_standard
cate = cate_standard

# Tuning parameter to use (for xgb) and classifier definition
if gbm == "xgb":
    param = dict(n_estimators=1100, learning_rate=0.01,
                 max_depth=3, min_child_weight=10,
                 colsample_bytree=0.7, subsample=0.7,
                 verbosity=0,
                 objective=objective,
                 n_jobs=sett.n_jobs)
else:
    param = dict(n_estimators=500, learning_rate=0.01,
                 num_leaves=32, min_child_weight=5,
                 objective=objective,
                 n_jobs=sett.n_jobs)
if objective == "reg:gamma":
    param["base_score"] = df[target_name].mean()


########################################################################################################################
# Prepare
########################################################################################################################

# --- Derive train/test ------------------------------------------------------------------------------------------------

# Duplicate for logtrafo-estimator
df["severity_cap_TOLOG"] = df["severity_cap"]
if objective == "gamma":
    df[target_name] = df[target_name] + 0.1

# Undersample only training data (take all but n_maxpersample at most)
if TARGET_TYPE == "CLASS":
    df.query("fold == 'train'")[target_name].value_counts()
    df_train, b_sample, b_all = uu.undersample(df.query("fold == 'train'"), target=target_name,
                                               n_max_per_level=3000)
    print(b_sample, b_all)
    if np.any(np.isclose(b_sample, b_all)):
        algo = estimator(**param)
    else:
        algo = uu.ScalingEstimator(estimator(**param), b_sample=b_sample, b_all=b_all)
        #alternative: algo = XGBClassifier_rescale(**xgb_param, b_sample = b_sample, b_all = b_all)
else:
    df_train = df[df[target_name].notna()].query("fold == 'train'").reset_index(drop=True)
    if target_name == "severity_cap_TOLOG":
        algo = uu.LogtrafoEstimator(estimator(**param))
    else:
        algo = estimator(**param)
    
# Test data
df_test = df[df[target_name].notna()].query("fold == 'test'").reset_index(drop=True)  

# Combine again
df_traintest = pd.concat([df_train, df_test]).reset_index(drop=True)

# Folds for crossvalidation and check
cv_5foldsep = uu.KFoldSep(5)
split_5foldsep = cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))
i_train, i_test = next(split_5foldsep)
print("TRAIN-fold:\n", df_traintest["fold"].iloc[i_train].value_counts(), i_train[:5])
print("TEST-fold:\n", df_traintest["fold"].iloc[i_test].value_counts(), i_test[:5])

    
# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
pipe = Pipeline(
    [('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume),
                                    ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate)]))),
     ('predictor', algo)])
features = np.append(nume, cate)
model = pipe.fit(df_train[features], df_train[target_name])

# Predict
if TARGET_TYPE == "CLASS":
    yhat_test = model.predict_proba(df_test[features])
else:
    yhat_test = model.predict(df_test[features])
print("spear: ", uu.spear(df_test[target_name].values, yhat_test))
print("auc: ", uu.auc(df_test[target_name].values, yhat_test))

if plot:
    d_calls = up.get_plotcalls_model_performance(y=df_test[target_name], yhat=yhat_test, target_type=TARGET_TYPE)
    uu.plot_function_calls(l_calls=d_calls.values(),
                           pdf_path=uu.plotloc + "3__performance__" + TARGET_TYPE + "_" + target_name + ".pdf")
    if objective in ["count:poisson", "poisson"]:
        d_calls = up.get_plotcalls_model_performance(y=np.where(df_test[target_name] > 0, 1, 0),
                                                     yhat=np.where(yhat_test > 1, 1, yhat_test),
                                                     threshold=df_train[target_name].mean())
        uu.plot_function_calls(l_calls=d_calls.values(),
                               pdf_path=uu.plotloc + "3__performance__" + TARGET_TYPE + "_" + target_name + 
                               "_asCLASS.pdf")
    
# Check performance for crossvalidated fits
d_cv = cross_validate(model, df_traintest[features], df_traintest[target_name],
                      cv=cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test")),  # special 5fold
                      scoring=scoring,
                      return_estimator=True,
                      n_jobs=1)
print(d_cv["test_" + metric], " \n", np.mean(d_cv["test_" + metric]), np.std(d_cv["test_" + metric]))


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------

# Variable importance (on train data!)
df_varimp_train = uu.variable_importance(model, df_train[features], df_train[target_name], features,
                                         target_type=TARGET_TYPE,
                                         scoring=scoring[metric],
                                         random_state=42, n_jobs=sett.n_jobs)
# Scikit's VI: permuatation_importance("same parameter but remove features argument and add n_repeats=1")

# Top features (importances sum up to 95% of whole sum)
features_top_train = df_varimp_train.loc[df_varimp_train["importance_cum"] < 95, "feature"].values

# Fit again only on features_top
pipe_top = Pipeline([
    ('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume[np.in1d(nume, features_top_train)]),
                                   ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"),
                                    cate[np.in1d(cate, features_top_train)])]))),
    ('predictor', clone(algo))])
model_top = pipe_top.fit(df_train[features_top_train], df_train[target_name])

# Plot performance of features_top model
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_top = model_top.predict_proba(df_test[features_top_train])
    print(uu.auc(df_test[target_name].values, yhat_top))
else:
    yhat_top = model_top.predict(df_test[features_top_train])
    print(uu.spear(df_test[target_name].values, yhat_top))
if plot:
    d_calls = up.get_plotcalls_model_performance(y=df_test[target_name], yhat=yhat_top, target_type=TARGET_TYPE)
    uu.plot_function_calls(l_calls=d_calls.values(),
                           pdf_path=sett.plotloc + "3__performance_top__" + TARGET_TYPE + "_" + target_name + ".pdf")



########################################################################################################################
# Diagnosis
########################################################################################################################

# ---- Check residuals fot top features --------------------------------------------------------------------------------------

# Residuals
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # "1 - yhat_of_true_class"
    df_test["residual"] = 1 - yhat_test[np.arange(len(df_test[target_name])), df_test[target_name]]
else:
    df_test["residual"] = df_test[target_name] - yhat_test
df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()

# For non-regr tasks one might want to plot it for each target level (df_test.query("target == 0/1"))
if plot:
    (hms_plot.MultiFeatureDistributionPlotter(target_limits=None if TARGET_TYPE == "REGR" else (0, 0.5),
                                              n_rows=2, n_cols=3, w=18, h=12)
        .plot(features=df_test[features_top_train],
              target=df_test["residual"],
              file_path=sett.plotloc + "3__diagnosis_residual__" + TARGET_TYPE + "_" + target_name + ".pdf"))

# Absolute residuals
if TARGET_TYPE == "REGR":
    if plot:
        (hms_plot.MultiFeatureDistributionPlotter(target_limits=None, n_rows=2, n_cols=3, w=18, h=12)
         .plot(features=df_test[features_top_train],
               target=df_test["abs_residual"],
               file_path=sett.plotloc + "3__diagnosis_absolute_residual__" + TARGET_TYPE + "_" + target_name + ".pdf"))


########################################################################################################################
# Variable Importance
########################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------

#xgb.plot_importance(model[1].estimator if type(model[1]) == uu.ScalingEstimator else model[1])


# --- Variable Importance by permuation argument ----------------------------------------------------------------------

# Importance (on test data!)
df_varimp_test = uu.variable_importance(model, df_test[features], df_test[target_name], features,
                                        target_type=TARGET_TYPE,
                                        scoring=scoring[metric],
                                        random_state=42, n_jobs=sett.n_jobs)
features_top_test = df_varimp_test.loc[df_varimp_test["importance_cum"] < 95, "feature"].values

# Compare variable importance for train and test (hints to variables prone to overfitting)
if plot:
    fig, ax = plt.subplots(1, 1)
    sns.barplot(x="score_diff", y="feature", hue="fold",
                data=pd.concat([df_varimp_train.assign(fold="train"),
                                df_varimp_test.assign(fold="test")], sort=False),
                ax=ax)

# Crossvalidate Importance (only for top features)
df_varimp_test_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))):
    df_tmp = df_traintest.iloc[i_train, :]
    df_varimp_test_cv = df_varimp_test_cv.append(
        uu.variable_importance(d_cv["estimator"][i], df_tmp[features], df_tmp[target_name], features_top_test,
                               target_type=TARGET_TYPE,
                               scoring=scoring[metric],
                               random_state=42, n_jobs=sett.n_jobs).assign(run=i))
df_varimp_test_se = (df_varimp_test_cv.groupby("feature")["score_diff", "importance"].agg("sem")
                     .pipe(lambda x: x.set_axis([col + "_se" for col in x.columns], axis=1, inplace=False))
                     .reset_index())

# Add other information (e.g. special category)
df_varimp_test["category"] = pd.cut(df_varimp_test["importance"], [-np.inf, 10, 50, np.inf],
                                    labels=["low", "medium", "high"])

# Plot Importance
df_varimp_plot = (df_varimp_test.query("feature in @features_top_test")
                  .merge(df_varimp_test_se, how="left", on="feature"))
l_calls = [(uu.plot_variable_importance,
            dict(features=df_varimp_plot["feature"],
                 importance=df_varimp_plot["importance"],
                 importance_cum=df_varimp_plot["importance_cum"],
                 importance_se=df_varimp_plot["importance_se"],
                 max_score_diff=df_varimp_plot["score_diff"][0].round(2),
                 category=df_varimp_plot["category"]))]
if plot:
    uu.plot_function_calls(l_calls, n_rows=1, n_cols=1, figsize=(8, 4), 
                           pdf_path=sett.plotloc + "3__vi__" + TARGET_TYPE + "_" + target_name + ".pdf")


########################################################################################################################
# Partial Dependance
########################################################################################################################

'''
# Scikit's partial dependence
from sklearn.inspection import permutation_importance, partial_dependence

# cate
cate_top_test = my.diff(features_top_test, nume)
tmp = partial_dependence(model, df_test[features],
                   features=cate_top_test[0],  # just one feature per call is possible!
                   grid_resolution=np.inf,  # workaround to take all members
                   kind="individual")
# nume
nume_top_test = my.diff(features_top_test, cate)
from joblib import Parallel, delayed
Parallel(n_jobs=my.n_jobs, max_nbytes='100M')(
    delayed(partial_dependence)(model, df_test[features], feature,
                                grid_resolution=5,  # 5 quantiles
                                kind="average")
     for feature in nume_top_test)
'''

# --- Standard PD --------------------------------------------------------------------------------------------------

# Dataframe based patial dependence which can use a reference dataset for value-grid defintion
d_pd = uu.partial_dependence(model, df_test[features], features_top_test, df_ref=df_train)

# Crossvalidate
d_pd_cv = {feature: pd.DataFrame() for feature in features_top_test}
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest,
                                                        test_fold=(df_traintest["fold"] == "test").values)):
    d_pd_run = uu.partial_dependence(model, df_traintest.iloc[i_test, :][features], features_top_test,
                                     df_ref=df_train)
    for feature in features_top_test:
        d_pd_cv[feature] = d_pd_cv[feature].append(d_pd_run[feature].assign(run=i)).reset_index(drop=True)
d_pd_err = {feature: df_tmp.drop(columns="run").groupby("value").std()
            for feature, df_tmp in d_pd_cv.items()}

# Plot it
l_calls = list()
for i, feature in enumerate(list(d_pd.keys())):
    i_col = {"REGR": 0, "CLASS": 1, "MULTICLASS": 2}
    l_calls.append((uu.plot_pd,
                    dict(feature_name=feature, feature=d_pd[feature]["value"],
                         yhat=d_pd[feature].iloc[:, i_col[TARGET_TYPE]].values,
                         yhat_err=d_pd_err[feature].iloc[:, i_col[TARGET_TYPE]].values,
                         feature_ref=df_test[feature],
                         refline=yhat_test[:, i_col[TARGET_TYPE]].mean() if TARGET_TYPE != "REGR" else yhat_test.mean(),
                         ylim=None, color=sett.colorblind[i_col[TARGET_TYPE]])))
if plot:
    uu.plot_function_calls(l_calls, pdf_path=sett.plotloc + "3__pd__" + TARGET_TYPE + "_" + target_name + ".pdf")
    

'''
# --- Shap based PD --------------------------------------------------------------------------------------------------

# Get shap for test data
explainer = shap.TreeExplainer(model[1].estimator if type(model[1]) is my.ScalingEstimator else model[1])
shap_values = my.agg_shap_values(explainer(model[0].transform(X=df_test[features])),
                                 df_test[features],
                                 len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_,
                                 round=2)

# Rescale due to undersampling
if TARGET_TYPE == "CLASS":
    shap_values.base_values = my.logit(my.scale_predictions(my.inv_logit(shap_values.base_values), b_sample, b_all))
if TARGET_TYPE == "MULTICLASS":
    shap_values.base_values = np.log(my.scale_predictions(np.exp(shap_values.base_values) /
                                                          np.exp(shap_values.base_values).sum(axis=1, keepdims=True),
                                                          b_sample, b_all))
# Aggregate shap
d_pd_shap = my.shap2pd(shap_values, features_top_test, df_ref=df_train)

# Plot it
l_calls = list()
for i, feature in enumerate(list(d_pd_shap.keys())):
    i_col = {"REGR": 0, "CLASS": 1, "MULTICLASS": 2}
    l_calls.append((my.plot_pd,
                    dict(feature_name=feature, 
                         feature=d_pd_shap[feature]["value"],
                         yhat=d_pd_shap[feature]["yhat"],
                         #feature_ref=df_test[feature],
                         #refline=yhat_test[:, i_col[TARGET_TYPE]].mean() if TARGET_TYPE != "REGR" else yhat_test.mean(),
                         ylim=None, color=my.colorblind[i_col[TARGET_TYPE]])))
if plot:
    my.uu.plot_function_calls(l_calls, pdf_path=my.plotloc + "3__pd_shap__" + TARGET_TYPE + "_" + target_name + ".pdf")
'''

########################################################################################################################
# Explanations
########################################################################################################################

# ---- Explain bad predictions ------------------------------------------------------------------------------------

# Filter data
n_select = 6
i_worst = df_test.sort_values("abs_residual", ascending=False).iloc[:n_select, :].index.values
i_best = df_test.sort_values("abs_residual", ascending=True).iloc[:n_select, :].index.values
i_random = df_test.sample(n=n_select).index.values
i_explain = np.concatenate([i_worst, i_best, i_random])
df_explain = df_test.iloc[i_explain, :].reset_index(drop=True)
if TARGET_TYPE == "CLASS":
    yhat_explain = model.predict_proba(df[features].iloc[i_explain, :])
else:
    yhat_explain = model.predict(df[features].iloc[i_explain, :])

# Get shap
explainer = shap.TreeExplainer(model[1].estimator if type(model[1]) == uu.ScalingEstimator else model[1])
X_explain = model[0].transform(X=df_explain[features])
if gbm == "lgbm":
    X_explain = X_explain.toarray()
shap_values = explainer(X_explain)
shap_values = uu.agg_shap_values(shap_values,
                                 df_explain[features],
                                 len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_,
                                 round=2)  # aggregate onehot

# Rescale due to undersampling
if TARGET_TYPE == "CLASS":
    shap_values.base_values = uu.logit(uu.scale_predictions(uu.inv_logit(shap_values.base_values), b_sample, b_all))
if TARGET_TYPE == "MULTICLASS":
    shap_values.base_values = np.log(uu.scale_predictions(np.exp(shap_values.base_values) /
                                                          np.exp(shap_values.base_values).sum(axis=1, keepdims=True),
                                                          b_sample, b_all))

# Check
shaphat = shap_values.values.sum(axis=1) + shap_values.base_values
if TARGET_TYPE == "REGR":
    if objective.endswith("poisson"):
        print(sum(~np.isclose(np.exp(shaphat), model.predict(df_explain[features]))))
    else:
        print(sum(~np.isclose(shaphat, model.predict(df_explain[features]))))
elif TARGET_TYPE == "CLASS":
    if gbm == "lgbm":
        print(sum(~np.isclose(uu.inv_logit(shaphat)[:, 1], model.predict_proba(df_explain[features])[:, 1])))
    else:
        print(sum(~np.isclose(uu.inv_logit(shaphat), model.predict_proba(df_explain[features])[:,1])))

# Plot default waterfall
fig, ax = plt.subplots(1, 1)
i = 1
i_col = {"CLASS": 1, "MULTICLASS": df_explain[target_name].iloc[i]}
y_str = (str(df_explain[target_name].iloc[i]) if TARGET_TYPE != "REGR" 
         else format(df_explain[target_name].iloc[i], ".2f"))
yhat_str = (format(yhat_explain[i, i_col[TARGET_TYPE]], ".3f") if TARGET_TYPE != "REGR" 
            else format(yhat_explain[i], ".2f"))
ax.set_title("id = " + str(df_explain[id_name].iloc[i]) + " (y = " + y_str + ")" + r" ($\^ y$ = " + yhat_str + ")")
if TARGET_TYPE != "MULTICLASS":
    shap.plots.waterfall(shap_values[i], show=True)  # TDODO: replace "00"
else:
    shap.plots.waterfall(shap_values[i][:, df_explain[target_name].iloc[i]], show=True)

# Plot it
l_calls = list()
for i in range(len(df_explain)):
    y_str = (str(df_explain[target_name].iloc[i]) if TARGET_TYPE != "REGR"
             else format(df_explain[target_name].iloc[i], ".2f"))
    i_col = {"CLASS": 1, "MULTICLASS": df_explain[target_name].iloc[i]}
    yhat_str = (format(yhat_explain[i, i_col[TARGET_TYPE]], ".3f") if TARGET_TYPE != "REGR"
                else format(yhat_explain[i], ".2f"))
    l_calls.append((uu.plot_shap,
                    dict(shap_values=shap_values, 
                         index=i, 
                         id=df_explain[id_name][i],
                         y_str=y_str,
                         yhat_str=yhat_str,
                         multiclass_index=None if TARGET_TYPE != "MULTICLASS" else i_col[TARGET_TYPE])))
if plot:
    uu.plot_function_calls(l_calls, pdf_path=sett.plotloc + "3__shap__" + TARGET_TYPE + "_" + target_name + ".pdf")


