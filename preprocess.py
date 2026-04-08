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
