import pandas as pd
import numpy as np

# Load datasets
cf = pd.read_csv("data/cystfibr.csv")
nhanes = pd.read_csv("data/nhanes_clean.csv")

print("CF Shape:", cf.shape)
print("NHANES Shape:", nhanes.shape)

print("\nCF Columns:", cf.columns)
print("\nNHANES Columns:", nhanes.columns)
