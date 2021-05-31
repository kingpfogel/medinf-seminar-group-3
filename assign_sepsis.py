import numpy as np
import pandas as pd

X1 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Output1d.csv")
X2 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Output2.csv")
sepsis = pd.read_csv(r"C:\Users\Kai\Desktop\outputs\python_generated_csv\19-06-12-sepsis_onsets.csv")

sepsis = sepsis[["hadm_id", "sepsis_time"]]

sepsis["sepsis_time"] = pd.to_datetime(sepsis["sepsis_time"])
sepsis["sepsis_time"] = sepsis["sepsis_time"].apply(lambda x: x.replace(minute=0, second=0))

X2 = pd.merge(X, sepsis, how="left")

X2["sepsis_onset"] = np.where(X2["sepsis_time"] == X2["admittime"], 1, 0)

X2.to_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Test_v2.csv")

