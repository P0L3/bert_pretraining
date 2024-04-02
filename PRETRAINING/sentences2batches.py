"""
Batch sentences from tokenized data into sequences with a maximum sequence length.

    Parameters:
        DIR (str): Directory containing tokenized data.
        AVG_SENT_LEN (int): Average sentence length.
        MAX_SEQ_LEN (int): Maximum sequence length.

    Returns:
        pd.DataFrame: DataFrame containing the batched sentences.
"""

import pandas as pd
from os import listdir
from tqdm import tqdm

# Directory containing tokenized data
DIR = "DATASET/ED4RE_2503/TOKENIZED"
# Average sentence length (adjust accordingly)
AVG_SENT_LEN = 50
# Maximum sequence length (adjust accordingly)
MAX_SEQ_LEN = 512

# Calculate the number of sentences per sequence
n_sentences = MAX_SEQ_LEN // AVG_SENT_LEN

# List files in the directory
files = listdir(DIR)[0:10]

# Print the list of files
print(files)

# List to store new data
new_data = []

# Iterate through each file
for file in tqdm(files):
    # Read the pickle file into a DataFrame
    df = pd.read_pickle(DIR + "/" + file)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the list of sentences from the 'Sentences' column
        sentences = row['Sentences']

        # Iterate through the list of sentences, with a step size of n_sentences
        for i in range(0, len(sentences), n_sentences):
            # Create a new row with 'text' containing a chunk of n_sentences joined together
            new_row = {'Text': ' '.join(sentences[i:i+n_sentences])}
            # Append the new row to the new_data list
            new_data.append(new_row)

# Create a new DataFrame from new_data
new_df = pd.DataFrame(new_data)

# Write the new DataFrame to a CSV file
new_df.to_csv(f"./DATASET/BATCHED/ED4RE_MSL{MAX_SEQ_LEN}_ASL{AVG_SENT_LEN}_S{len(new_df)}")
