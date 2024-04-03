import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from os import mkdir

DATA = "ED4RE_MSL512_ASL50_S3592675"
DIR_TRAIN = f"DATASET/BATCHED/{DATA}_train/*.arrow"
DIR_TEST = f"DATASET/BATCHED/{DATA}_test/*.arrow"
MODEL = "allenai/scibert_scivocab_uncased"
CHKPT = "MODELS/{}_{}".format(MODEL.replace("/", "__"), DATA)

print("Checking for CUDA: ", torch.cuda.is_available())
try:
    mkdir(CHKPT)
    print("Checkpoint folder created: ", CHKPT)
except:
    print("Checkpoint folder already exists: ", CHKPT)

# Load the saved dataset
print("Loading data ...")
train = load_dataset("arrow", data_files=DIR_TRAIN)
test = load_dataset("arrow", data_files=DIR_TEST)

# Access the loaded dataset
print(train)
print(test)

# Modle and tokenizer load
print("Loading tokenizer and model ...")
tokenizer = BertTokenizer.from_pretrained(MODEL)
model = BertForMaskedLM.from_pretrained(MODEL)

# MLM data prep
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

# Training args
training_args = TrainingArguments(
    output_dir=CHKPT,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=1,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=24, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=24,  # evaluation batch size
    logging_steps=300,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=300,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

# Initialize traininer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train['train'],
    eval_dataset=test['train'],
)

# train the model
trainer.train()