import pandas as pd
import os
import pymorphy2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pickle

PATH = 'models/'
with open(os.path.join(PATH, 'tfidf.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

time_related = ['лет', 'год ', 'меся', 'недел', 'дне', 'года']
specials_to_remove = [
    '.', '"', "'", '?', '(', ')', '`',
]
specials_to_replace = [
    '-', '\\', '/', ','
]
key_pos = ['NOUN', 'VERB', 'NUMR', 'ADJF', 'ADJS', 'INFN']
morph = pymorphy2.MorphAnalyzer()

def you_know_first_steps(x):
    if x:
        if 'не знаю' in x.lower():
            return 0
        else:
            return 1
    else:
        return 0

def is_time_certain(x):
    if x:
        x = str(x).lower()
        for i in time_related:
            if i in x:
                return 1
            else:
                continue
        return 0
    else:
        return 0

def certainly_imagined(x):
    if x:
        if ' четко' in x.lower():
            return 1
        else:
            return 0
    else:
        return 0

def are_obstackles_expected(x):
    if x:
        if 'не вижу преград' in str(x).lower() or 'нет' in str(x).lower():
            return 0
        else:
            return 1
    else:
        return 1

def remove_special(x):
    for special in specials_to_remove:
        if special in x:
            x =  x.replace(special, '').strip()
        else:
            pass
    return x

def replace_special(x):
    for special in specials_to_replace:
        if special in x:
            x =  x.replace(special, ' ').strip()
        else:
            pass
    return x

def create_list_of_words(x):
    return x.split(' ')

def clean_LoW_nv(x):
    clean_LoW_nv = []
    for word in x:
        if word.isdigit() == True:
            clean_LoW_nv.append(word)
        else:
            p = morph.parse(word)[0]
            normal_form = p.normal_form
            pos = p.tag
            stop = 0
            for s_pos in key_pos:
                if s_pos in pos:
                    clean_LoW_nv.append(normal_form)
                else:
                    continue
    return ' '.join(clean_LoW_nv)

def word_counter(x):
    return len(x)

def letters_counter(x):
    counter = int()
    for word in x:
        counter += len(word)
    return counter

def pos_counter(x, pos_to_comp):
    pos_counter = int()
    for word in x:
        p = morph.parse(word)[0]
        pos = p.tag
        for pos_ in pos_to_comp:
            if pos_ in pos:
                pos_counter += 1
            else:
                pass
    return pos_counter

def digit_counter(x):
    digit_counter = int()
    for word in x:
        if word.isdigit() == True:
            digit_counter += 1
    return digit_counter

def create_featured_datasets(input:dict):
    input_df = pd.DataFrame.from_dict(input, orient='index').T
    input_df['are_first_steps_known'] = input_df.loc[:, 'goal_first_step'].apply(lambda x: you_know_first_steps(x))
    input_df['is_time_certain'] = input_df['goal_time'].apply(lambda x: is_time_certain(x))
    input_df['is_certainly_imagined'] = input_df['goal_result'].apply(lambda x: certainly_imagined(x))
    input_df['are_obstackles_expected'] = input_df['goal_obstacle'].apply(lambda x: are_obstackles_expected(x))
    input_df.drop(columns=['goal_result', 'goal_first_step', 'goal_obstacle', 'goal_time'], inplace=True)
    input_df['space'] = ' '
    input_df['name_type'] = input_df['goal_name'] + input_df['space'] + input_df['goal_type']
    input_df.drop(columns=['goal_name', 'goal_type', 'space'], inplace=True)
    input_df['goal_domain'] = input_df['goal_domain'].apply(lambda x: str(x).lower())
    input_df['name_type'] = input_df['name_type'].apply(lambda x: str(x).lower())
    input_df['goal_domain'] = input_df['goal_domain'].apply(lambda x: remove_special(x))
    input_df['name_type'] = input_df['name_type'].apply(lambda x: remove_special(x))
    input_df['goal_domain'] = input_df['goal_domain'].apply(lambda x: replace_special(x))
    input_df['name_type'] = input_df['name_type'].apply(lambda x: replace_special(x))
    input_df['goal_domain_LoW'] = input_df['goal_domain'].apply(lambda x: create_list_of_words(x))
    input_df['name_type_LoW'] = input_df['name_type'].apply(lambda x: create_list_of_words(x))
    input_df['goal_domain_clean_NV_LoW'] = input_df['goal_domain_LoW'].apply(lambda x: clean_LoW_nv(x))
    input_df['name_type_clean_NV_LoW'] = input_df['name_type_LoW'].apply(lambda x: clean_LoW_nv(x))
    input_df['topic_words'] = input_df['goal_domain_LoW'].apply(lambda x: word_counter(x))
    input_df['goal_words'] = input_df['name_type_LoW'].apply(lambda x: word_counter(x))
    input_df['topic_letters'] = input_df['goal_domain_LoW'].apply(lambda x: letters_counter(x))
    input_df['goal_letters'] = input_df['name_type_LoW'].apply(lambda x: letters_counter(x))
    input_df['topic_aver_word_len'] = round(input_df['topic_letters'].div(input_df['topic_words']), 2)
    input_df['goal_aver_word_len'] = round(input_df['goal_letters'].div(input_df['goal_words']), 2)
    input_df['goal_verbs_counter'] = input_df['name_type_LoW'].apply(lambda x: pos_counter(x, ['VERB', 'INFN']))
    input_df['goal_nouns_counter'] = input_df['name_type_LoW'].apply(lambda x: pos_counter(x, ['NOUN']))
    input_df['goal_numr_counter'] = input_df['name_type_LoW'].apply(lambda x: pos_counter(x, ['NUMR']))
    input_df['goal_adj_counter'] = input_df['name_type_LoW'].apply(lambda x: pos_counter(x, ['ADJF', 'ADJS']))
    input_df['goal_digit_counter'] = input_df['name_type_LoW'].apply(lambda x: digit_counter(x))

    # Features
    df_features = input_df[['are_first_steps_known', 'is_time_certain',
               'is_certainly_imagined', 'are_obstackles_expected',
                'topic_words', 'goal_words', 'topic_letters',
               'goal_letters', 'topic_aver_word_len', 'goal_aver_word_len',
               'goal_verbs_counter', 'goal_nouns_counter', 'goal_numr_counter',
               'goal_adj_counter', 'goal_digit_counter']]
    mms = MinMaxScaler()
    features = mms.fit_transform(df_features.values)

    # Vectors
    df_vectors = input_df[['name_type_clean_NV_LoW']] # only name-type for now
    text = df_vectors['name_type_clean_NV_LoW']
    vectors = tfidf.transform(text)

    return features, vectors, input_df