

test_sentence = "Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural, due to changes in the sun’s activity or large volcanic eruptions."

def test_tokenizer(tokenizer, sentence=test_sentence):
    print("Sentence: ")
    print(sentence)
    print("---"*10)
    print(tokenizer.tokenize(sentence))