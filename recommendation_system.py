from itertools import chain
from string import punctuation
import pymorphy2

class Word_Tokenizer():
    def tokenize(self, text):
        for element in punctuation.replace("+", ""):
            text = text.replace(element, "")
        return text.split()

word_tokenize = Word_Tokenizer()

#  tagggers
morph = pymorphy2.MorphAnalyzer(lang="ru")

#  get chosen tokens
def get_format(x):
    tags = [i[0] for token in x for i in morph.parse(token) if i.tag.POS == None]

    tags_sorted = sorted([i[0] for token in x for i in morph.parse(token)
                        if i.tag.POS == "NOUN"], reverse=True)[:3]

    concat_tages = chain(tags, tags_sorted)
    return set(concat_tages)

#  get the number of values of each item in the list
def get_list_counts_books(x: list, data_temp):
    temp = [[i["id"], 0] for i in data_temp]
    for chars in x:
        for i, element in enumerate(data_temp):
            count1 = str(element["title"]).lower().count(chars) * 2
            count2 = str(element["genres"]).count(chars) * 4
            temp[i][1] += count1 + count2
    return temp

#  get top course-block for books
def get_top_blocks_books(x, data_temp):
    list_counts = get_list_counts_books(x, data_temp)
    list_counts.sort(key=lambda y: y[1],
                                reverse=True)
    top_five = list_counts[:5]
    names_out = []
    for counts in top_five:
        for element in data_temp:
            if element["id"] == counts[0]:
                names_out.append(element)
    return names_out

#  get result
def get_recommendations(books_info, data):
    x = get_format(word_tokenize.tokenize(data.lower()))
    return get_top_blocks_books(x, books_info)