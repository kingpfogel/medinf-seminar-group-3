import numpy as np
import pandas as pd

X1 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\alt\Output1.csv")
X2 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\alt\Output2.csv")
X = X1.append(X2, ignore_index=True)

sepsis = pd.read_csv(r"C:\Users\Kai\Desktop\sepsis3-mimic-master\data\sepsis3-df.csv")

sepsis = sepsis[["hadm_id", "suspected_infection_time_poe"]]

sepsis["suspected_infection_time_poe"] = pd.to_datetime(sepsis["suspected_infection_time_poe"])
sepsis["suspected_infection_time_poe"] = sepsis["suspected_infection_time_poe"].apply(lambda x: x.replace(minute=0, second=0))

X2 = pd.merge(X, sepsis, how="left")

X2["sepsis_onset"] = np.where(X2["suspected_infection_time_poe"] == X2["admittime"], 1, 0)


X2.to_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Final.csv")

