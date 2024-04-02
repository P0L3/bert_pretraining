"""

"""

from datasets import load_dataset
from transformers import BertTokenizer, BertForPreTraining
import torch



## DATA LOADING
DIR = "DATASET/BATCHED/ED4RE_MSL512_ASL50_S3592675"
MAX_SEQ_LEN = 512

# Load the data from csv
dataset = load_dataset("csv", data_files=DIR)

# Create train-test split
train_test_split = dataset['train'].train_test_split(test_size=0.05, seed=3005)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print(train_dataset)
print(test_dataset)

# # See how it looks
# for t in train_dataset["Text"][:3]:
#   print(t)
#   print("="*50)



## TOKENIZATION
truncate_longer_samples = True
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["Text"], truncation=True, padding="max_length",
                   max_length=MAX_SEQ_LEN, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["Text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset_encoded = train_dataset.map(encode, batched=True)

# tokenizing the testing dataset
test_dataset_encoded = test_dataset.map(encode, batched=True)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset_encoded.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset_encoded.set_format(type="torch", columns=["input_ids", "attention_mask"])

print(len(train_dataset_encoded), len(test_dataset_encoded))



## SAVE DATASETS
train_dataset_encoded.save_to_disk(DIR+"_train")
test_dataset_encoded.save_to_disk(DIR+"_test")

