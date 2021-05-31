import pandas as pd
import datetime as dt

X = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MIMIC3\foo.csv", sep=",")
detail = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MIMIC3\local_mimic\views\icustay_detail.csv")

detail = detail[["hadm_id", "age", "admittime"]]

col_list = ["subj_id", "hadm_id", 'heartrate', 'sysbp', 'diasbp', 'tempc', 'resprate', 'spo2', 'glucose',
             'albumin', 'bun', 'creatinine', 'sodium', 'bicarbonate', 'platelet', 'inr',
             'potassium', 'calcium', 'ph', 'pco2', 'lactate']

X.columns = col_list

X = pd.merge(X, detail, how="left")

X["admittime"] = pd.to_datetime(X["admittime"])
X["admittime"] = X["admittime"].apply(lambda x: x.replace(minute=0, second=0))

# Add Hours
for idx, row in X.iterrows():
    print(idx)
    X.loc[idx, "admittime"] = row["admittime"] + dt.timedelta(hours=int(idx) % 48)

# X1 = X.iloc[0:700000]
# X2 = X.iloc[700000:1426800]
# X1.to_csv("Output1.csv", sep=",", header=True)
# X2.to_csv("Output2.csv", sep=",", header=True)

X.to_csv("Admittime_fixed.csv", sep=",", header=True)
