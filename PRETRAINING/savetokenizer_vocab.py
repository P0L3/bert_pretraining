# from transformers import PreTrainedTokenizerFast

# DIR = "MODELS/p0l3__clirebert_clirevocab_uncased_ED4RE_MSL512_ASL50_S3592675_4/CliReBERT_142/"

# tokenizer = PreTrainedTokenizerFast(tokenizer_file=DIR+"tokenizer.json")
# # tokenizer = AutoTokenizer.from_pretrained(DIR)

# test_sentence = "This is a test sentence! ENSO rocks!"

# print(test_sentence)
# print(tokenizer.tokenize(test_sentence))

# print("\n\nSaving tokenizer vocab ...")
# print(tokenizer.get_vocab())

from transformers import PreTrainedTokenizerFast

DIR = "./LOCAL_MODELS/CliReBERT/"

# Specify the path to your tokenizer.json file
tokenizer_path = DIR+"tokenizer.json"

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# Extract the vocabulary
vocab = tokenizer.get_vocab()

# Sort the vocabulary by the token ID to ensure the order is correct
sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

# Specify the path to save the vocabulary
vocab_save_path = DIR+"vocab.txt"

# Save the vocabulary to a text file
with open(vocab_save_path, 'w') as vocab_file:
    for token, idx in sorted_vocab:
        vocab_file.write(f"{token}\n")

print(f"Vocabulary saved to {vocab_save_path}")