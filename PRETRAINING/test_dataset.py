from datasets import load_dataset
from os import listdir
from transformers import BertTokenizer, BertTokenizerFast

DIR = "DATASET/BATCHED/ED4RE_MSL512_ASL50_S3592675_test/*.arrow"

test = load_dataset("arrow", data_files=DIR)

print(test["train"][0].keys())
# print(test["train"][0]["input_ids"])

# tokenizer = BertTokenizerFast(tokenizer_file="LOCAL_MODELS/CliReBERT/tokenizer.json")
tokenizer = BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/tokenizer.json")

# Convert tokens back to string
# Convert input_ids back to tokens
tokens = tokenizer.convert_ids_to_tokens(test["train"][0]["input_ids"])

print("Original: ", test["train"][0]["Text"])
print("Tokens: ", tokens)

