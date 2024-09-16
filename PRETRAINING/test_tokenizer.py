from transformers import BertTokenizer, RobertaTokenizer, BertTokenizerFast


tokenizers = []

# CliReBERT
tokenizers.append((BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/tokenizer.json"), "CliReBERT"))
# CliReRoBERTa
tokenizers.append((RobertaTokenizer(vocab_file="LOCAL_MODELS/CliReRoBERTa/vocab.json", merges_file="LOCAL_MODELS/CliReRoBERTa/merges.txt"), "CliReRoBERTa"))
# CliSciBERT
tokenizers.append((BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased"), "CliSciBERT"))
# SciClimateBERT
tokenizers.append((RobertaTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f"), "SciClimateBERT"))
# CliReBERT
tokenizers.append((BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/tokenizer.json"), "CliReBERT_fixed"))
# CliReBERT_fixed_1
tokenizers.append((BertTokenizerFast(tokenizer_file="LOCAL_MODELS/CliReBERT/tokenizer.json"), "CliReBERT_fixed_1"))
# CliReBERT_fixed_2
tokenizers.append((BertTokenizer("LOCAL_MODELS/CliReBERT/tokenizer.json"), "CliReBERT_fixed_2"))
# CliReBERT_fixed_4real
tokenizers.append((BertTokenizer(vocab_file="LOCAL_MODELS/CliReBERT/vocab.txt"), "CliReBERT_fixed_4real"))



test_sentence = "Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural, due to changes in the sun’s activity or large volcanic eruptions."


print("Sentence: ")
print(test_sentence)
print("---"*10)

for tokenizer in tokenizers:
    print(tokenizer[1])
    print(tokenizer[0].tokenize(test_sentence))
    print()