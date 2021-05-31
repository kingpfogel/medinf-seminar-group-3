import h5py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import statsmodels.api as sm

# LOAD DATA
# region
pd.set_option('display.max_columns', None)

DATAFILE = "all_hourly_data.h5"
static = pd.read_hdf(DATAFILE, 'patients')
# endregion

# PROCESS DATA
# region
X = static[["age", "gender"]]
one_hot = pd.get_dummies(X["gender"], prefix="gender")
X = X.drop("gender", axis=1)
X = X.join(one_hot)

y = static["mort_icu"].values.reshape(-1, 1).ravel()
# endregion

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# clf = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train, y_train)
# pred = clf.predict(X_test)
# fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
# print(metrics.auc(fpr, tpr))

logit = sm.Logit(y_train, X_train)
result = logit.fit()
print(result.summary())



