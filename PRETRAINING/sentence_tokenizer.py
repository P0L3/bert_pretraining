"""
This script processes a large CSV dataset by tokenizing the text content in batches using SpaCy, 
and saves the tokenized content to a series of pickle files for further processing or analysis.

The script reads the specified CSV file in batches to manage memory usage efficiently. For each batch, 
it utilizes SpaCy to perform sentence tokenization on the text content contained within a specific column of the dataset. 
The resulting tokenized sentences are then saved to a new column within the dataframe. Finally, each processed batch 
is saved to a separate pickle file, allowing for incremental processing and the ability to pause or stop the process 
without losing all progress.

Parameters:
- DIR (str): The directory path to the input CSV file containing the text to be tokenized.
- FINAL_DIR (str): The directory path where the tokenized batches will be saved as pickle files.
- BATCH_SIZE (int): The number of rows to be processed in each batch. -> Smaller size recommended
- RANGE (int): The total number of batches to process, calculated as the total number of rows divided by the batch size.

Output:
- Series of pickle files each containing a dataframe with the original content and the corresponding tokenized sentences.

Notes:
- The script assumes the existence of a SpaCy language model at a specified path relative to the dataset's directory.
- Error handling includes printing the number of processed rows if an exception occurs, but more detailed error reporting or logging may be beneficial for troubleshooting.
- The script's performance and efficiency may vary based on the size of the dataset, the specific SpaCy model used, and the hardware it runs on.
- Reiterate on the ranges that did not succesfully tokenize with a smaller BATCH_SIZE 
"""

import pandas as pd
from tqdm import tqdm
from time import time
import spacy
from os import listdir


DIR = "/PRETRAINING/DATASET/ED4RE_2503/ED4RE_2603_tc_remaining.csv"
FINAL_DIR =  "/PRETRAINING/DATASET/ED4RE_2503/TOKENIZED/ED4RE_2603_tc_tokenized_remaining_"
BATCH_SIZE = 50
processed_rows = 0

RANGE = 42500 // BATCH_SIZE

def gen_to_list(gen):
    return [sent.text for sent in gen.sents]

for i in tqdm(range(RANGE)):
    df = pd.read_csv(DIR, nrows=BATCH_SIZE+1, skiprows=range(1, processed_rows)) # Fix with BATC_SIZE+1 to mitigate skipping last row
    try:
        start = time()

        print("Importing SpaCy ...")
        nlp = spacy.load('./DATASET/en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4/', disable=["ner"])

        print("Tagging sentences ...")
        # df["Sentences"] = df["Content"].progress_apply(text_to_sentence)
        df["Sentences"] = list(nlp.pipe(df['Content'], batch_size=10, n_process=1)) # Sadly doesn't work with n_process > 1
        
        print("Saving to list ...")
        df["Sentences"] = df["Sentences"].apply(gen_to_list)
        # print(df["Sentences"])

        print("Saving to pickle ...")
        #  df["Sentences"].apply(print_sentence_len)
        df.to_pickle(FINAL_DIR+str(processed_rows)+".pickle")

        print("Done ... Total time: ", time() - start)
    except:
        print(processed_rows)
    processed_rows += BATCH_SIZE