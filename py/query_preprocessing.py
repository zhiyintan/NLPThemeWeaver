import re
from nltk.corpus import wordnet as wn
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()

# Converts a query into a dictionary, where 
# keys: order indexes to the tokens in the query, values: tokens
# "a b (c1|c2) d (e1|e2|e3)" -> 
# {1:'a', 2:'b', 3:{c1, c2}, 4:'d', 5:{e1, e2, e3}, ..}
def standardized(query):
    token_list = query.strip("'\"").replace(', ', ' ').replace(',', ' ').split(" ")
    token_order_dic = {}; token_order_list = []
    for i in range(len(token_list)):
        #print("index: ", i)
        token = token_list[i].strip()
        #print("token: ", token)
        if token[0] == '(' and token[-1] == ")" and '|' in token: # "(cat|British short hair|British long hair)"
            manual_sys_set = set(token.strip('()').split('|')) # ("cat", "British short hair", "British long hair")
            token_order_dic[i] = manual_sys_set
            token_order_list.append(list(manual_sys_set))
        else:
            token_order_dic[i] = token
            token_order_list.append([token])
    return token_order_list # token_order_dic

# Get a synonyms list for a token
# token ->  [token, syn1, syn2, ..]
def get_synonyms_list(token):
    try:
        wn_token = wn.synsets(token)[0]
        synsets_list = [str(lemma.name()) for lemma in wn_token.lemmas()]
        for i in range(len(synsets_list)):
            if '_' in synsets_list[i]:
                synsets_list[i] = synsets_list[i].replace('_', ' ')
        return synsets_list
    except:
        return [token]
    

def add_synonyms(query):
    if type(query) is str:
        query_list = standardized(query)
    else:
        query_list = query
    # standardized_query format: [[],[], ..]

    result_list = []
    for token_list in query_list:
        synsets_set = set()
        for token in token_list:
            synsets_set.update(get_synonyms_list(token))
        result_list.append(list(synsets_set))
    return result_list # [[],[],..]


def add_stemming_token(query):
    if type(query) is str:
        query_list = standardized(query)
    else:
        query_list = query
    # standardized_query format: [[],[], ..]

    result_list = []
    for token_list in query_list:
        stem_set = set()
        for token in token_list:
            stem_set.add(token) # original token
            stem_set.add(stemmer.stem(token)) # stem token
        result_list.append(list(stem_set))
    return result_list # [[],[],..]


def remove_symbols(query):
    if type(query) is str:
        query_list = standardized(query)
    else:
        query_list = query
    # standardized_query format: [[],[], ..]

    result_list = []
    for token_list in query_list:
        rm_sym_set = set()
        for token in token_list:
            rm_sym_set.add(token)
            rm_sym_set.add(re.sub(r'[^a-zA-Z]', '', token))
        result_list.append(list(rm_sym_set))
    return result_list