from datasets import *
from transformers import *
from tokenizers import *
import os
import pandas as pd
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
BATCH_SIZE = 10000000

# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
DIR = "/PRETRAINING/DATASET/ED4RE_2503/ED4RE_2603_tc.csv"
df = pd.read_csv(DIR, nrows=BATCH_SIZE)

print(df[df["Content"] == "no_content"])