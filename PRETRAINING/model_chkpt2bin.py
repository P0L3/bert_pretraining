import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BertTokenizer, BertConfig, BertForMaskedLM
from datasets import load_dataset
from os import mkdir

DATA = "ED4RE_MSL512_ASL50_S3592675"
DIR_TRAIN = f"DATASET/BATCHED/{DATA}_train/*.arrow"
DIR_TEST = f"DATASET/BATCHED/{DATA}_test/*.arrow"
MODEL = "p0l3/clireroberta_clirevocab_cased" # "p0l3/clirebert_clirevocab_uncased"
BATCH = 4
NAME = "CliReRoBERTa_142" # "CliReBERT_142" # Add a name of the folde where you want to save the model


CHKPT = "MODELS/{}_{}_{}".format(MODEL.replace("/", "__"), DATA, BATCH)
LOGS = CHKPT + "/LOGS"

print("Checking for CUDA: ", torch.cuda.is_available())

# Load the saved dataset
print("Loading data ...")
train = load_dataset("arrow", data_files=DIR_TRAIN, split="train[:1%]") # Load only 1% of training data
test = load_dataset("arrow", data_files=DIR_TEST)

# Access the loaded dataset
print(train)
print(test)

### FOR ROBERTA PRETRAINED
# # Model and tokenizer load
# print("Loading tokenizer and model ...")
# tokenizer = RobertaTokenizer.from_pretrained(MODEL)
# model = RobertaForMaskedLM.from_pretrained(MODEL)
# Model and tokenizer load

### FOR BERT FROM SCRATCH
# print("Loading tokenizer and model ...")
# tokenizer = BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/tokenizer.json")
# config = BertConfig.from_json_file("LOCAL_MODELS/CliReBERT/config.json")
# model = BertForMaskedLM(config)

### FOR ROBERTA FROM SCRATCH
print("Loading tokenizer and model ...")
tokenizer = RobertaTokenizer(vocab_file="LOCAL_MODELS/CliReRoBERTa/vocab.json", merges_file="LOCAL_MODELS/CliReRoBERTa/merges.txt")
config = RobertaConfig.from_json_file("LOCAL_MODELS/CliReRoBERTa/config.json")
model = RobertaForMaskedLM(config)

print("Mask token:", tokenizer.mask_token)
print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)
print("PAD token:", tokenizer.pad_token)
print("UNK token:", tokenizer.unk_token)

# MLM data prep
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

# Training args
training_args = TrainingArguments(
    output_dir=CHKPT,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=0,            # number of training epochs, feel free to tweak, put to 0 to save the model
    per_device_train_batch_size=BATCH, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=BATCH,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,                # Save steps
    logging_dir=LOGS,               # Where logging steps go
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

# Initialize traininer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    eval_dataset=test,
)

# train the model
trainer.train(resume_from_checkpoint=True,)

# Saving model
try:
    mkdir(CHKPT+"/"+NAME)
    print("Final folder created: ", CHKPT+"/"+NAME)
except:
    print("Final folder already exists: ", CHKPT+"/"+NAME)

trainer.model.save_pretrained(CHKPT+"/"+NAME, safe_serialization=False) # Use safe_serialization=False when loading model in older versions of transformers (eg. 3.4.0)