import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

text = open("taylorswift.txt", "r", encoding="utf-8").read()
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")