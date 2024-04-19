"""
Loss visualization for first model - Hopefully loaded directly in next training
"""

from matplotlib import pyplot as plt
import ast

with open("./eval.txt", 'r') as f:
    evals = f.read().splitlines()

# print(evals)

eval_loss = [ast.literal_eval(s) for s in evals if s.startswith("{'e")]
loss = [ast.literal_eval(s) for s in evals if s.startswith("{'l")]

epochs_eval = [entry['epoch'] for entry in eval_loss]
eval_losses = [entry['eval_loss'] for entry in eval_loss]

epochs_loss = [entry['epoch'] for entry in loss]
losses = [entry['loss'] for entry in loss]

plt.plot(epochs_eval, eval_losses, label='Evaluation Loss', marker='o')
plt.plot(epochs_loss, losses, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Visualization')
plt.legend()
plt.grid(True)
plt.show()



