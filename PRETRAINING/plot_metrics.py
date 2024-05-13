"""
Plot loss, eval loss and simmilar based on the checkpoint
"""

from matplotlib import pyplot as plt
import json
from random import randint
import re

DIR = "MODELS/allenai__scibert_scivocab_uncased_ED4RE_MSL512_ASL50_S11369/checkpoint-177000/trainer_state.json" # Path to trainer state
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

with open(DIR, 'rb') as f:
    trainer_state = json.loads(f.read())

log_history = trainer_state["log_history"]
# print(log_history.keys())

# Extracting data from the checkpoint log history
epochs_checkpoint = [entry.get("epoch", None) for entry in log_history]
losses_checkpoint = [entry.get("loss", None) for entry in log_history]
eval_losses_checkpoint = [entry.get("eval_loss", None) for entry in log_history]
learning_rates_checkpoint = [entry.get("learning_rate", None) for entry in log_history]

# Extracting data from the previous log history
epochs_previous = [entry["epoch"] for entry in log_history]
grad_norms_previous = [entry.get("grad_norm", None) for entry in log_history]
eval_samples_per_sec_previous = [entry.get("eval_samples_per_second", None) for entry in log_history]

# Plotting
plt.style.use('seaborn-v0_8-paper')
plt.figure(figsize=(12, 10))

i = lambda a: randint(0, 6)

# Plotting loss and eval loss from checkpoint
plt.subplot(2, 2, 1)
plt.plot(epochs_checkpoint, losses_checkpoint, marker='o', label='Training Loss (Checkpoint)', color=colors[i(1)])
plt.plot(epochs_checkpoint, eval_losses_checkpoint, marker='o', label='Evaluation Loss (Checkpoint)', color=colors[i(1)])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Evaluation Loss (Checkpoint)')
plt.legend()

# Plotting learning rate from checkpoint
plt.subplot(2, 2, 2)
plt.plot(epochs_checkpoint, learning_rates_checkpoint, marker='o', label='Learning Rate (Checkpoint)', color=colors[i(1)])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate (Checkpoint)')
plt.legend()

# Plotting gradient norm and eval samples per second from previous log history
plt.subplot(2, 2, 3)
plt.plot(epochs_previous, grad_norms_previous, marker='o', label='Gradient Norm (Previous)', color=colors[i(1)])
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm (Previous)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs_previous, eval_samples_per_sec_previous, marker='o', label='Eval Samples/Sec (Previous)', color=colors[i(1)])
plt.xlabel('Epoch')
plt.ylabel('Eval Samples/Sec')
plt.title('Eval Samples/Sec (Previous)')
plt.legend()

plt.tight_layout()

plt.savefig("{}_stats".format(re.search(r'checkpoint-\d*', DIR).group(0)))
plt.show()

# # Extracting data
# epochs = [entry.get("epoch", None) for entry in log_history]
# losses = [entry.get("loss", None) for entry in log_history]
# eval_losses = [entry.get("eval_loss", None) for entry in log_history]
# learning_rates = [entry.get("learning_rate", None) for entry in log_history]


# print(plt.style.available)

# # Plotting
# plt.style.use('seaborn-v0_8-colorblind')
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(epochs, losses, marker='o', label='Training Loss')
# plt.plot(epochs, eval_losses, marker='o', label='Evaluation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Evaluation Loss')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(epochs, learning_rates, marker='o', label='Learning Rate')
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.title('Learning Rate')
# plt.legend()

# plt.tight_layout()
# plt.show()

