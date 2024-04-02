import pandas as pd
from transformers import BertTokenizer, BertForPreTraining
import torch

DIR = "DATASET/ED4RE_2503/TOKENIZED/ED4RE_2603_tc_tokenized_800.pickle"

df = pd.read_pickle(DIR)

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# def sentence_tokenize(list_sentence):


# sentence = df["Sentences"][10][6]
# print(sentence)
# print(tokenizer.tokenize(sentence))

# # Tokenizes a sentence and returns the tokens
# def tokenize_sentence(sentence):
#     return tokenizer.tokenize(sentence)

# # Flatten the list of lists of sentences
# all_sentences = [sentence for sublist in df["Sentences"] for sentence in sublist]

# # Tokenize all sentences and calculate the length of each token list
# token_lengths = [len(tokenize_sentence(sentence)) for sentence in all_sentences]

# # Calculate the average number of tokens per sentence
# average_tokens = sum(token_lengths) / len(token_lengths)

# print(f"Average number of tokens per sentence: {average_tokens}")

# Tokenize sentence
def tokenize_sentence(sentence):
    return tokenizer.tokenize(sentence)

# Flatten the list of lists of sentences
all_sentences = [sentence for sublist in df["Sentences"] for sentence in sublist]

# Tokenize all sentences and calculate the length of each token list
token_lengths = [len(tokenize_sentence(sentence)) for sentence in all_sentences]

# Sort the token lengths to prepare for quartile calculations
sorted_token_lengths = sorted(token_lengths)

# Calculate the index for the first and fourth quartiles
q1_index = len(sorted_token_lengths) // 4
q4_index = 3 * len(sorted_token_lengths) // 4

# Calculate the average token length for the first and fourth quartiles
average_tokens_q1 = sum(sorted_token_lengths[:q1_index]) / q1_index
average_tokens_q4 = sum(sorted_token_lengths[q4_index:]) / (len(sorted_token_lengths) - q4_index)

# General average for comparison
average_tokens = sum(sorted_token_lengths) / len(sorted_token_lengths)

print(f"Average number of tokens per sentence (overall): {average_tokens:.2f}")
print(f"Average number of tokens per sentence (Q1): {average_tokens_q1:.2f}")
print(f"Average number of tokens per sentence (Q4): {average_tokens_q4:.2f}")