import numpy as np
import pandas as pd

X1 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\alt\Sepsis_Output1.csv", sep=";")
X2 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\alt\Sepsis_Output2.csv", sep=";")
X = X1.append(X2, ignore_index=True)

X["row_number"] = X.groupby("hadm_id").cumcount()

grouped_df = X.groupby('hadm_id')
new_table = []
for name, group in grouped_df:
    try:
        idx = group.loc[group['sepsis_onset'] == 1, "row_number"]
        idx = idx.reset_index(drop = True)
        idx = idx[0]
        group1 = group.iloc[0:(idx+1), ]
        new_table.append(group1)
    except KeyError:
        new_table.append(group)

new_table = pd.concat(new_table).reset_index(drop=True)
new_table = new_table.drop("row_number", 1)
new_table.to_csv("Final_shortened.csv", index=False)
