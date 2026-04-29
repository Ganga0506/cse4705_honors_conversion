import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

# ---- Load intents ----
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

# ---- Step 1: Tokenize & Stem ----
def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word)

# ---- Step 2: Build vocabulary & training data ----
all_words = []
tags = []
xy = []  

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Remove punctuation and stem all words
ignore = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))  # remove duplicates
tags = sorted(set(tags))

print("Vocabulary:", all_words)
print("Tags:", tags)

# ---- Step 3: Bag of Words ----
def bag_of_words(tokenized_sentence, vocab):
    stemmed = [stem(w) for w in tokenized_sentence]
    bow = np.zeros(len(vocab), dtype=np.float32)
    for idx, word in enumerate(vocab):
        if word in stemmed:
            bow[idx] = 1.0
    return bow

# ---- Step 4: Build X (inputs) and Y (labels) ----
X_train = []
Y_train = []

for (tokens, tag) in xy:
    bow = bag_of_words(tokens, all_words)
    X_train.append(bow)
    Y_train.append(tags.index(tag)) 

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("X_train shape:", X_train.shape) 
print("Y_train shape:", Y_train.shape)  