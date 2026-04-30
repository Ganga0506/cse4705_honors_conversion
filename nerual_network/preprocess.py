import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, vocab):
    stemmed = [stem(w) for w in tokenized_sentence]
    bow = np.zeros(len(vocab), dtype=np.float32)
    for idx, word in enumerate(vocab):
        if word in stemmed:
            bow[idx] = 1.0
    return bow

def build_vocab(intents_path='nerual_network/nn_data/intents.json'):
    with open(intents_path, 'r') as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    ignore = ['?', '!', '.', ',']

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    all_words = sorted(set(stem(w) for w in all_words if w not in ignore))
    tags = sorted(set(tags))

    X_train = np.array([bag_of_words(tokens, all_words) for tokens, _ in xy])
    Y_train = np.array([tags.index(tag) for _, tag in xy])

    return all_words, tags, X_train, Y_train

if __name__ == "__main__":
    all_words, tags, X_train, Y_train = build_vocab()
    print("Vocabulary size:", len(all_words))
    print("Tags:", tags)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)