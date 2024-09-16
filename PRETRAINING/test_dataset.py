from datasets import load_dataset
from os import listdir
from transformers import BertTokenizer, BertTokenizerFast

DIR = "DATASET/BATCHED/ED4RE_MSL512_ASL50_S3592675_clirebert_clirevocab_uncased_train/*.arrow"

test = load_dataset("arrow", data_files=DIR)

i = 3

print(test["train"][i].keys())
# print(test["train"][0]["input_ids"])

tokenizer = BertTokenizerFast(tokenizer_file="LOCAL_MODELS/CliReBERT/tokenizer.json")
# tokenizer = BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/tokenizer.json")

# Convert tokens back to string
# Convert input_ids back to tokens
tokens = tokenizer.convert_ids_to_tokens(test["train"][i]["input_ids"])

print("Original: ", test["train"][i]["Text"])
print("Tokens: ", tokens)
print("Input Ids", test["train"][i]["input_ids"])

