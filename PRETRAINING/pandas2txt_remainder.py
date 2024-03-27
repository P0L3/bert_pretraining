"""
Script to prepair data for NSP and MLM: from pandas pickle to sentence txt and full text for remainder
"""

import pandas as pd
from tqdm import tqdm
from time import time
import spacy
from os import listdir


DIR = "/PRETRAINING/DATASET/ED4RE_2503/ED4RE_2603_tc.csv"
FINAL_DIR =  "/PRETRAINING/DATASET/ED4RE_2503/TOKENIZED/ED4RE_2603_tc_tokenized_"
BATCH_SIZE = 200
TOTAL = 172000

with open("left2process.txt", 'r') as f:
    remaining = f.read().split("\n")

# print(remaining)

remaining_rows = []

# Add all spans that had exceptions
for r in remaining:
    remaining_rows.extend(range(int(r), int(r)+BATCH_SIZE))

# Add all numbers that were missed due to bad code logic
for i in range(0, TOTAL, BATCH_SIZE):
    if i > 0:
        remaining_rows.append(i-1)
    else:
        remaining_rows.append(i)

# Make a set
remaining_rows = set(remaining_rows)
print("Rows remaining for tokenizetion: ", len(remaining_rows))

# Create a set of all rows
all_rows = set(range(0, TOTAL))

# Skiprows
skiprows = all_rows - remaining_rows

print("Rows that will be skipped:      ", len(skiprows))

df = pd.read_csv(DIR, skiprows=list(skiprows))
df.to_csv(DIR.replace(".csv", "_remaining.csv"))
print(df)