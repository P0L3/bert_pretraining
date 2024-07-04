""" 
Script for exploring BERT masked prediction.

This script loads different pre-trained BERT and RoBERTa models for masked language 
modeling tasks, tokenizes input text, and predicts masked tokens.

Configuration is provided for different models, which includes the model type, 
directory, and tokenizer path.

Usage:
- Set the desired model name in the `MODEL_NAME` variable.
- Use the `predict_masked_sent` function to predict masked tokens in a given text.

"""

from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
import torch

# Configuration for each model "model_name" : (type, model_directory, model_tokenizer)
config = {
    "CliReBERT" : ("BERT", "MODELS/p0l3__clirebert_clirevocab_uncased_ED4RE_MSL512_ASL50_S3592675_4/checkpoint-177000", "LOCAL_MODELS/CliReBERT/tokenizer.json"),
    "CliReRoBERTa" : ("ROBERTA", "MODELS/p0l3__clireroberta_clirevocab_cased_ED4RE_MSL512_ASL50_S3592675_4/checkpoint-177000_CliReRoBERTa", "LOCAL_MODELS/CliReRoBERTa"),
    "CliSciBERT" : ("BERT", "MODELS/allenai__scibert_scivocab_uncased_ED4RE_MSL512_ASL50_S11369/checkpoint-177000", "allenai/scibert_scivocab_uncased"),
    "SciClimateBERT" : ("ROBERTA", "MODELS/climatebert__distilroberta-base-climate-f_ED4RE_MSL512_ASL50_S3592675_24/checkpoint-177000_ClimateBERT_177", "climatebert/distilroberta-base-climate-f"),
    "SciBERT" : ("BERT", "allenai/scibert_scivocab_uncased", "allenai/scibert_scivocab_uncased"),
    "ClimateBERT_f" : ("ROBERTA", "climatebert/distilroberta-base-climate-f", "climatebert/distilroberta-base-climate-f")
}

# Change this as per config
MODEL_NAME = "ClimateBERT_f"


DIR_MODEL = config[MODEL_NAME][1]
DIR_TOKENIZER = config[MODEL_NAME][2]
MODEL_TYPE = config[MODEL_NAME][0] 

if MODEL_TYPE == "BERT":
    model = BertForMaskedLM.from_pretrained(DIR_MODEL)
    tokenizer = BertTokenizer.from_pretrained(DIR_TOKENIZER)
elif MODEL_TYPE == "ROBERTA":
    model = RobertaForMaskedLM.from_pretrained(DIR_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(DIR_TOKENIZER)

print(model)
print(tokenizer)
print("Mask token:", tokenizer.mask_token)
print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)
print("PAD token:", tokenizer.pad_token)
print("UNK token:", tokenizer.unk_token)

def predict_masked_sent(text, top_k=5, tokenizer=tokenizer):
    # Tokenize input
    text = f"{tokenizer.cls_token} {text} {tokenizer.sep_token}"
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index(f"{tokenizer.mask_token}")  
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    print(text)
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

        
predict_masked_sent(f"Climate change causes {tokenizer.mask_token}", top_k=10, tokenizer=tokenizer)




