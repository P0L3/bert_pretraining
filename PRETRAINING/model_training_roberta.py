import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from os import mkdir


MODEL = "climatebert/distilroberta-base-climate-f" # Change dependaning opn the model you want to train
model_name = MODEL.split("/")[-1]

DATA = f"ED4RE_MSL512_ASL50_S3592675_{model_name}"
DIR_TRAIN = f"DATASET/BATCHED/{DATA}_train/*.arrow"
DIR_TEST = f"DATASET/BATCHED/{DATA}_test/*.arrow"
BATCH = 24

CHKPT = "MODELS/{}_{}_{}".format(MODEL.replace("/", "__"), DATA, BATCH)
LOGS = CHKPT + "/LOGS"

print("Checking for CUDA: ", torch.cuda.is_available())
try:
    mkdir(CHKPT)
    print("Checkpoint folder created: ", CHKPT)
except:
    print("Checkpoint folder already exists: ", CHKPT)

try:
    mkdir(LOGS)
    print("Logs folder created: ", LOGS)
except:
    print("Logs folder already exists: ", LOGS)

# Load the saved dataset
print("Loading data ...")
train = load_dataset("arrow", data_files=DIR_TRAIN)
test = load_dataset("arrow", data_files=DIR_TEST)

# Access the loaded dataset
print(train)
print(test)

# Modle and tokenizer load
print(f"Loading tokenizer and model: {model_name}")
tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = RobertaForMaskedLM.from_pretrained(MODEL)

# MLM data prep
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

# Training args
training_args = TrainingArguments(
    output_dir=CHKPT,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs, feel free to tweak
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
    train_dataset=train['train'],
    eval_dataset=test['train'],
)

# train the model
trainer.train()

# Saving model
try:
    mkdir(CHKPT+"/pretrained_final")
    print("Final folder created: ", CHKPT+"/pretrained_final")
except:
    print("Final folder already exists: ", CHKPT+"/pretrained_final")

trainer.model.save_pretrained(CHKPT+"/pretrained_final_10epochs", safe_serialization=False) # Use safe_serialization=False when loading model in older versions of transformers (eg. 3.4.0)