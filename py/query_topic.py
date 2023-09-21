from py.query_preprocessing import add_synonyms, add_stemming_token, remove_symbols

import numpy as np
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")
tokenizer = nlp.tokenizer


def get_relevant_score(query_list, document):
    score_list = []
    for token_list in query_list:
        counts = 0
        for token in token_list:
            counts += document.lower().count(token)
        score_list.append(np.tanh(counts)) # in range (0, 1)
        #if document.lower().count(q) != 0:
            #print("query_list: ", query_list)
            #print("query: ", q)
            #print("paper title: ", document.lower())
            #print("counts: ", document.lower().count(q))

    basic_score = sum(score_list)/len(score_list)
    tanh_basic_score = np.tanh(basic_score) # in range (0, 1)
    attendance_rate = (len(score_list) - score_list.count(0))/len(score_list)
    if attendance_rate > (len(score_list)-1)/len(score_list):
        result = tanh_basic_score + attendance_rate * 2
    else:
        result = tanh_basic_score

    #if result > 0.5:
    #    print("score_list: ", score_list)
    #    print("basic_score: ", basic_score, '\t', "tanh_basic_score: ", tanh_basic_score)
    #    print("attendance_rate: ", attendance_rate)
    #    print("result: ", result)
    #    print(query_list)
    #    print(document)
    return result


class searchTopic:

    def __init__(self, query_topic, top_n, i2t_filename):
        self.query_topic = query_topic
        self.top_n = top_n
        self.i2t_filename = i2t_filename
        self.df_result = pd.DataFrame()
        return

    def clean_topic(self, topic_list):
        result_list = []
        doc = nlp(' '.join(topic_list))
        for token in doc:
            result_list.append(token.lemma_)
        return list(set(result_list))
    
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
            t2i_dic[', '.join(topic_list)] = topic_id
            '''for t in topic_set:
                if t not in t2i_dic:
                    t2i_dic[t] = [topic_id]
                else:
                    t2i_dic[t].append(topic_id)'''
        return t2i_dic, i2t_dic

    # get top N topics for the given query
    def query_to_topic(self):
        revised_query = remove_symbols(add_stemming_token(self.query_topic)) # add stem token, remove all non [a-z]
        print("Your entered query: ", self.query_topic)
        print("The extened query: ", revised_query, '\n') # [[],[], ..]

        t2i_dic, i2t_dic = self.get_topic_id_dic()
        
        score_dic = {}
        for topic in t2i_dic:
            score = get_relevant_score(revised_query, topic)
            topic_id = t2i_dic[topic]
            score_dic[topic_id] = score

        if self.top_n > len(score_dic):
            top_n = len(score_dic)
            print("As the total number of relevant topics is ", len(score_dic), ",", len(score_dic), "topics will be presented.")
        else:
            top_n = self.top_n

        top_n_topic_id_list = sorted(score_dic.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        #print("top_n_topic_id_list: ", top_n_topic_id_list)

        self.df_result = pd.DataFrame(columns=["Topic ID", "Topic Keywords", "Score"])
        for topic_id, score in top_n_topic_id_list:
            topic = self.clean_topic(i2t_dic[topic_id])
            self.df_result.loc[len(self.df_result.index)] = [topic_id, str(topic), str(score)] 
        #print(topic_result)
        return self.df_result


class searchPaper:
    def __init__(self, query_topic, topic_id, paper_number, i2p_filename, origin_filename):
        self.query_topic = query_topic
        self.topic_id = topic_id
        self.paper_number = paper_number
        self.i2p_filename = i2p_filename
        self.origin_filename = origin_filename
        self.df_result = pd.DataFrame()
        return

    def check_topic_id(self):
        # Check if the topic is delineated
        topic_id_list = []
        for topic_id in self.topic_id.split(','):
            try:
                topic_id = int(topic_id.strip())
                topic_id_list.append(topic_id)
            except:
                pass
        if len(topic_id_list) != 0:
            return topic_id_list
        else:
            print("No valid topic id inserted. All papers will be retrieved.")
            return None


    def get_paperIndex_score_dic(self):
        revised_query = add_stemming_token(add_synonyms(self.query_topic)) # add token's synonyms, then add stem token

        df_i2p = pd.read_csv(self.i2p_filename) # topic id <-> paper
        paperIndex_score_dic = {}; paperIndex_paper_key_dic = {}

        topic_id_list = self.check_topic_id()
        if topic_id_list is None:
            #print("topic_id_list is None")
            for i in range(len(df_i2p)):
                paper_index = i
                topic_id = df_i2p.loc[i, 'Topic']
                paper = df_i2p.loc[i, 'Document'].strip("['\"]")
                score = get_relevant_score(revised_query, paper)
                paperIndex_score_dic[paper_index] = score
                paperIndex_paper_key_dic[paper_index] = paper[:8]
        else:
            #print("topic_id_list is not None")
            for i in range(len(df_i2p)): 
                paper_index = i
                topic_id = df_i2p.loc[i, 'Topic']
                if topic_id in topic_id_list:
                    paper = df_i2p.loc[i, 'Document'].strip("['\"]")
                    score = get_relevant_score(revised_query, paper)
                    paperIndex_score_dic[paper_index] = score
                    paperIndex_paper_key_dic[paper_index] = paper[:8]
        return paperIndex_score_dic, paperIndex_paper_key_dic


    def get_related_paper(self):
        paperIndex_score_dic, paperIndex_paper_key_dic = self.get_paperIndex_score_dic()

        if self.paper_number > len(paperIndex_score_dic):
            paper_number = len(paperIndex_score_dic)
            print("As the total number of relevant paper is ", len(paperIndex_score_dic), ",", len(paperIndex_score_dic), "papers will be presented.\n")
        else:
            paper_number = self.paper_number
        paperIndex_score_dic_sorted = sorted(paperIndex_score_dic.items(), key=lambda kv: kv[1], reverse=True)[:paper_number]

        df = pd.read_csv(self.origin_filename, index_col=None) 

        self.df_result = pd.DataFrame(columns=["Title", "Publication Year", "Url", "Score"])
        for paper_index, score in paperIndex_score_dic_sorted:
            paper_key = paperIndex_paper_key_dic[paper_index]
            df_index_list = df[df['Key'].str.contains(paper_key, case=False)].index.tolist()
            if len(df_index_list) == 1:
                index = df_index_list[0]
                self.df_result.loc[len(self.df_result.index)] = [df.loc[index, "Title"], str(df.loc[index, "Publication Year"]), str(df.loc[index, "Url"]), str(score)] 
                #"Score: ", score, "paper_key: ", '\t', paper_key, '\t', "paper_index: ", index, '\n', df.loc[index, "Author"], '\n',
            else:
                print("Miss match happend. Index error: ", df_index_list, '\n'
                    "paper key: ", paper_key)
        #print(self.df_result.to_string())
        return self.df_result

    def sort_result(self, sort_by="1", ascending='1'):
        if ascending == 'yes':
            ascending = True
        elif ascending == 'no':
            ascending = False

        if sort_by == "1": # sort by Title
            #print(self.df_result.sort_values(by='Title', ascending=ascending).to_string())
            return self.df_result.sort_values(by='Title', ascending=ascending)
        elif sort_by == "2": # sort by Publication Year
            #print(self.df_result.sort_values(by='Publication Year', ascending=ascending).to_string())
            return self.df_result.sort_values(by='Publication Year', ascending=ascending)
        elif sort_by == "3": # sort by Score
            #print(self.df_result.sort_values(by='Score', ascending=ascending).to_string())
            return self.df_result.sort_values(by='Score', ascending=ascending)
        
    
if __name__ == "__main__":
    print('\n')
    query_topic = input("Enter the topic you want: ")#"fine-grained entity"
    top_n = int(input("How many relevant topic categories you want to check: "))

    i2t_filename = '../document/id_topic.csv'
    i2p_filename = '../document/id_paper.csv'
    origin_filename = '../document/ACL_Anthology_(092023).csv'


    print('\n')
    querying = searchTopic(query_topic, top_n, i2t_filename)
    print(querying.query_to_topic())

    print("Next, enter topic and topic id for paper searching. \n")

    print("Organize query statements using commas('word, word') and spaces('word word'). \n \
    For synonyms, enclose them in parentheses and separate with a vertical bar(word (word|word)).")
    query_topic = input("\nEnter the topic to get the related papers: ")

    print("\nTopic id could be a integer number, e.g. '1' \nalso could be serval integer number separate by comma(,), e.g. '1, 2'\n")
    topic_id = input("Any known relevant topic id? If, yes, please enter, otherwise press 'Enter': ")
    paper_number = int(input("How many documents to display: "))
    print('\n')

    searching = searchPaper(query_topic, topic_id, paper_number, i2p_filename, origin_filename)
    result = searching.get_related_paper()
    print(result)
    print('\n')

    sort_by = input("Which column would you like to sort? \n 1 - Title; \n 2 - Publication Year; \n 3 - Score \n :")
    ascending = input("Would you prefer to sort in ascending order? yes/no: ")
    sort_result = searching.sort_result(sort_by=sort_by, ascending=ascending)
    print(sort_result)