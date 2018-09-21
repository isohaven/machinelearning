import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
	lexicon = []
	for file in [pos, neg]:
		with open(file, 'r') as f:
			content = f.readlines()
			for l in contends[:hm_lines]
			all_words = word_tokenize(l)
			lexicon += list(all_words)

