import numpy as np
import pandas as pd

X1 = pd.read_csv(r"C:\Users\Kai\PycharmProjects\MedInf\Final.csv")
print(X1["sepsis_onset"].sum())