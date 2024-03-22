from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')

