"""
Script to prepair data for NSP and MLM: from pandas pickle to sentence txt and full text 
"""

import pandas as pd
from tqdm import tqdm
from time import time

DIR = "./DATASET/full_10ksamplecsv"

df = pd.read_csv(DIR, nrows=500)
start = time()
# df.to_csv(DIR.replace(".pickle", ".csv"))
# exit()
# print(df["Content"])
print("Importing SpaCy ...")
import spacy
nlp = spacy.load('./DATASET/en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4/', disable=["ner"])

# def text_to_sentence(content):
#     return [sent for sent in nlp(content).sents]
# tqdm.pandas()

def gen_to_list(gen):
    return [sent.text for sent in gen.sents]

def print_sentence_len(sent_list):
    print(len(sent_list))
    return sent_list

print("Tagging sentences ...")
# df["Sentences"] = df["Content"].progress_apply(text_to_sentence)
df["Sentences"] = list(nlp.pipe(df['Content'], batch_size=10, n_process=12))

df["Sentences"] = df["Sentences"].apply(gen_to_list)
print(df["Sentences"])

df["Sentences"].apply(print_sentence_len)
print("Done ... Total time: ", time() - start)