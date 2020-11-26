import os
import pickle
from string import punctuation
import re
import pymorphy2

comp = re.compile(r"\b\d*\b")

morph = pymorphy2.MorphAnalyzer(lang="ru")

class Word_Tokenizer():
    def tokenize(self, text):
        for element in punctuation.replace("+", ""):
            text = text.replace(element, "")
        return text.split()

word_tokenize = Word_Tokenizer()

def get_tokens(x):
    out = []
    for element in word_tokenize.tokenize(x):
        tokens = morph.parse(element)
        if type(tokens) == list:
            tokens = tokens[0]
        out.append(tokens.tag.POS)
    return out

class name_checker():
    def __init__(self, stop_words, list_of_pos):
        self.stop_words = stop_words
        self.list_of_pos = list_of_pos

    def is_correct_input_name(self, x):
        try:
            x_clear = x.strip().lower()
            n = len(x_clear.split())
            if x_clear in self.stop_words:
                return 0
            if (len(set(x_clear)) == 1) or (x_clear == ""):
                return 0
            if ((x.count("%") >= 1) or (x.count(",") >= 1) or (x.count(".") >= 1)) and (n >= 3):
                return 1
            poses = get_tokens(x)
            if (str(poses) in self.list_of_pos) and (n >= 3):
                return 1
            dates = comp.search(x)
            if (dates.span() != (0, 0)) and (n >= 3):
                return 1
            return 0
        except: return 0

def load_name_checker(path=""):
    with open(os.path.join(path, "stopwords.pkl"), "rb") as f:
            stopwords = pickle.load(f)

    with open(os.path.join(path, "list_of_pos.pkl"), "rb") as f:
            list_of_pos = pickle.load(f)

    return name_checker(stopwords, list_of_pos)