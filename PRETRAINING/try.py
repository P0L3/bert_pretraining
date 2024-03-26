import spacy
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from time import time

DIR = "/PRETRAINING/DATASET/ED4RE_2503/ED4RE_2603_tc.csv"
FINAL_DIR =  "/PRETRAINING/DATASET/ED4RE_2503/TOKENIZED/ED4RE_2603_tc_tokenized_"
BATCH_SIZE = 150
RANGE = 200000 // BATCH_SIZE
processed_rows = 0

def gen_to_list(gen):
    return [sent.text for sent in gen.sents]

def print_sentence_len(sent_list):
    # print(len(sent_list))
    return sent_list


for i in tqdm(range(RANGE)):
    df = pd.read_csv(DIR, nrows=BATCH_SIZE, skiprows=range(1, processed_rows))
    df["Content"] = df["Content"].apply(lambda x: True if pd.isna(x) or x == "" else False)

    if len(df[df["Content"] != False]["Title"]) != 0:
        print(df)
    # df["Content"] = df["Content"].apply(lambda x: type(x))
    # a = df[df["Content"] == type(str("aa")) or df["Content"]]["Content"]
    # if len(a) != 0:
    #     print(a)

    # start = time()


    # print("Importing SpaCy ...")
    # nlp = spacy.load('./DATASET/en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4/', disable=["ner"])

    # print("Tagging sentences ...")
    # # df["Sentences"] = df["Content"].progress_apply(text_to_sentence)
    # df["Sentences"] = list(nlp.pipe(df['Content'], batch_size=10, n_process=1))
    
    # print("Saving to list ...")
    # df["Sentences"] = df["Sentences"].apply(gen_to_list)
    # # print(df["Sentences"])

    # print("Saving to pickle ...")
    # #  df["Sentences"].apply(print_sentence_len)
    # df.to_pickle(FINAL_DIR+str(processed_rows)+".pickle")

    # print("Done ... Total time: ", time() - start)
    # processed_rows += BATCH_SIZE