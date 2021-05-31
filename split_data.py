import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

X = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Final_shortened.csv")
id_list = X["hadm_id"].unique()

pre_inds, val_inds = next(GroupShuffleSplit(test_size=.25, n_splits=2, random_state=42).split(X, groups=X['hadm_id']))

pre = X.iloc[pre_inds]
val = X.iloc[val_inds]

train_inds, test_inds = next(GroupShuffleSplit(test_size=.25, n_splits=2, random_state=42).split(pre, groups=pre['hadm_id']))

train = X.iloc[train_inds]
test = X.iloc[test_inds]

train.to_csv("Train.csv")
test.to_csv("Test.csv")
val.to_csv("Validation.csv")
