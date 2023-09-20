import numpy as np
import pandas as pd

import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()

import spacy
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")


# take a list of input token, output each token's stemming form
def get_related_topic(query_list:list, split_by_hyphen=1): 
    extension_list = []
    for token in query_list:
        extension_list.append(stemmer.stem(token))
        if split_by_hyphen == 1:
            sub_token_list = token.split('-') # further split, might use regular expression
            if len(sub_token_list) > 1:
                extension_list.append(''.join(sub_token_list))
                for sub_token in sub_token_list:
                    extension_list.append(sub_token)
                    extension_list.append(stemmer.stem(sub_token))
    extension_list += query_list
    return list(set(extension_list))


class searchTopic:

    def __init__(self, query, top_n, i2t_filename):
        self.query = query
        self.top_n = top_n
        self.i2t_filename = i2t_filename
        return

    # get a topic to topic id dictionary
    def get_topic_id_dic(self):
        df_i2t = pd.read_csv(self.i2t_filename) # topic id <-> topic
        t2i_dic = {}; i2t_dic = {}
        for i in range(1, len(df_i2t)):  # Skip line 1 as its topic id = -1
            topic_id = df_i2t.loc[i, 'Topic']
            topic_list = df_i2t.loc[i, 'Representation'].strip("['\"]").split("', '")
            doc = nlp(' '.join(topic_list))
            topic_set = set([token.lemma_ for token in doc])

            i2t_dic[topic_id] = topic_list

            for t in topic_set:
                if t not in t2i_dic:
                    t2i_dic[t] = [topic_id]
                else:
                    t2i_dic[t].append(topic_id)
        return t2i_dic, i2t_dic

    # get top N topics for the given query
    def query_to_topic(self):
        query = self.query.replace(',', ' ').replace(', ', ' ').strip("'\"").split(" ") # or use tokenizer
        print("Your entered query: ", query)
        extened_query_list = get_related_topic(query)
        print("The extened query: ", extened_query_list)

        t2i_dic, i2t_dic = self.get_topic_id_dic()
        
        counts_dic = {}
        for topic in t2i_dic:
            topic_id_list = t2i_dic[topic]

            for token in extened_query_list:
                if token in topic:
                    #print("topic:", topic)
                    #print("topic_id_list:", topic_id_list)

                    for topic_id in topic_id_list:
                        if topic_id not in counts_dic:
                            #print("topic_id: ", topic_id)
                            #print("counts_dic: ", counts_dic)
                            counts_dic[topic_id] = 1
                            #print("counts_dic: ", counts_dic)
                        else:
                            counts_dic[topic_id] += 1
        #print("counts_dic: ", counts_dic)
        for topic_id in counts_dic:
            counts_dic[topic_id] = counts_dic[topic_id]/len(i2t_dic[topic_id])
        #print("counts_dic: ", counts_dic)
        top_n_topic_id_list = sorted(counts_dic.items(), key=lambda kv: kv[1], reverse=True)[:self.top_n]
        #print("top_n_topic_id_list: ", top_n_topic_id_list)

        topic_result = "Topic ID \t Score \t Topic Keywords \n"
        for topic_id, counts in top_n_topic_id_list:
            topic = i2t_dic[topic_id]
            topic_result += str(topic_id) + '\t' + str(counts) + '\t' + str(topic) + '\n'
        print(topic_result)
        return


class searchPaper:
    def __init__(self, query_topic, topic_id, paper_number, i2p_filename, origin_filename):
        self.query_topic = query_topic
        self.topic_id = topic_id
        self.paper_number = paper_number
        self.i2p_filename = i2p_filename
        self.origin_filename = origin_filename
        return

    def check_topic_id(self):
        # Check if the topic is delineated
        topic_id_list = []
        for topic_id in self.topic_id.split(','):
            try:
                topic_id = int(topic_id.strip())
                topic_id_list.append(topic_id)
            except:
                print(topic_id, " is not a valid topic id. It should be an integer.")
        if len(topic_id_list) != 0:
            return topic_id_list
        else:
            return None

    def similarity(self, query, document):
        score_list = []
        extened_query_list = query #get_related_topic(query, split_by_hyphen=0)
        
        
        for q in extened_query_list:
            score_list.append(document.lower().count(q))
            #if document.lower().count(q) != 0:
                #print("extened_query_list: ", extened_query_list)
                #print("query: ", q)
                #print("paper title: ", document.lower())
                #print("counts: ", document.lower().count(q))

        basic_score = sum(score_list)/len(score_list)
        tanh_basic_score = np.tanh(basic_score) # in range (0, 1)
        attendance_rate = (len(score_list) - score_list.count(0))/len(score_list)
        if attendance_rate > 0.5:
            result = tanh_basic_score + attendance_rate *2
        else:
            result = tanh_basic_score

        #if result > 2.99999:
        #    print("score_list: ", score_list)
        #    print("basic_score: ", basic_score, '\t', "tanh_basic_score: ", tanh_basic_score)
        #    print("attendance_rate: ", attendance_rate)
        #    print("result: ", result)
        #    print(extened_query_list)
        #    print(document)
        return result


    def get_paperIndex_score_dic(self):
        query = self.query_topic.replace(',', ' ').replace(', ', ' ').strip("'\"").split(" ") # or use tokenizer

        df_i2p = pd.read_csv(i2p_filename) # topic id <-> paper
        paperIndex_score_dic = {}; paperIndex_paper_key_dic = {}

        topic_id_list = self.check_topic_id()
        if topic_id_list is not None:
            for i in range(len(df_i2p)):
                paper_index = i
                topic_id = df_i2p.loc[i, 'Topic']
                paper = df_i2p.loc[i, 'Document'].strip("['\"]")

                
                score = self.similarity(query, paper)
                paperIndex_score_dic[paper_index] = score
                paperIndex_paper_key_dic[paper_index] = paper[:8]
        else:
            for i in range(len(df_i2p)): 
                paper_index = i
                topic_id = df_i2p.loc[i, 'Topic']
                if topic_id in topic_id_list:
                    paper = df_i2p.loc[i, 'Document'].strip("['\"]")
                    score = self.similarity(self.query_topic, paper)
                    paperIndex_score_dic[paper_index] = score
                    paperIndex_paper_key_dic[paper_index] = paper[:8]
        return paperIndex_score_dic, paperIndex_paper_key_dic


    def get_related_paper(self):
        paperIndex_score_dic, paperIndex_paper_key_dic = self.get_paperIndex_score_dic()
        paperIndex_score_dic_sorted = sorted(paperIndex_score_dic.items(), key=lambda kv: kv[1], reverse=True)[:self.paper_number]

        df = pd.read_csv(self.origin_filename, index_col=None) 

        for paper_index, score in paperIndex_score_dic_sorted:
            paper_key = paperIndex_paper_key_dic[paper_index]
            df_index_list = df[df['Key'].str.contains(paper_key, case=False)].index.tolist()
            if len(df_index_list) == 1:
                index = df_index_list[0]
                print("Score: ", score, "paper_key: ", '\t', paper_key, '\t', "paper_index: ", index, '\n', df.loc[index, "Title"] , ', ',\
                df.loc[index, "Publication Year"], '\t', df.loc[index, "Author"], '\n', df.loc[index, "Url"], '\n')
            else:
                print("Miss match happend. Index error: ", df_index_list, '\n'
                    "paper key: ", paper_key)
        return

    

print('\n')
query_topic = input("Enter the topic you want: ")#"fine-grained entity"
top_n = int(input("How many relevant topic categories you want to check: "))

i2t_filename = 'id_topic.csv'
i2p_filename = 'document_topic.csv'
origin_filename = 'ACL_Anthology_(092023).csv'


print('\n')
querying = searchTopic(query_topic, top_n, i2t_filename)
querying.query_to_topic()

print("Next, enter topic and topic id for paper searching. \n Topic id could be a integer number, e.g. '1' \n also could be serval integer number separate by comma(,), e.g. '1, 2'\n")
query_topic = input("Enter the topic to get the related papers: ")
topic_id = input("Any known relevant topic id? If, yes, please enter, otherwise press 'Enter': ")
paper_number = int(input("How many documents to display: "))
print('\n')

searching = searchPaper(query_topic, topic_id, paper_number, i2p_filename, origin_filename)
searching.get_related_paper()

'''
df_i2p = pd.read_csv('document_topic.csv')  # topic id 2 paper

print(t2i_dic['entity'])
get_related_topic()



print(df_id2topic.loc[0:3, :])
print(df_topic2paper.loc[0:3, :])'''