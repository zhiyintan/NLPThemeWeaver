import pandas as pd
from tqdm import tqdm
import spacy
import spacy_fastlang
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"



#----------------------------------------DATA------------------------------------
file_path = "../document/english_sentence.txt"
if os.path.isfile(file_path):
    pass
else:
    def is_english_word(sentence, fix_percentage):
        sentence = " ".join(sentence.split(" ")[-10:-1])
        doc = nlp(sentence)
        #print(sentence)
        #print(doc._.language)
        #print(doc._.language_score)
        if doc._.language == 'en' and doc._.language_score > fix_percentage:
            return True
        else:
            return False


    df = pd.read_csv('ACL_Anthology_(092023).csv') 
    #print(list(df.columns.values))

    key_2_content = {}; other_lang = 0; content_str = ""
    for i in tqdm(range(len(df))):
        key = df.loc[i, 'Key']
        title = str(df.loc[i, 'Title'])
        abstract = str(df.loc[i, 'Abstract Note']).strip("nan")
        content = title + ". " + abstract
        #print(is_english_word(content, fix_percentage=0.3))
        if is_english_word(content, fix_percentage=0.3):
            key_2_content[key] = content
            content_str += key + "\t" + key + ' ' + content + "\n"
            
        else:
            other_lang += 1
    #print(content_str)
    english_sentence = open("english_sentence.txt", "w")
    english_sentence.write(content_str)
    english_sentence.close()



#-----------------------------------------MODEL-------------------------------------
content_list = []
with open("english_sentence.txt", "r") as es:
    for line in es:
        content_list.append(line.strip("\n").split("\t")[1])

docs = content_list #fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']


# Fine-tune your topic representations
from bertopic.representation import MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance()
topic_model = BERTopic(representation_model=representation_model, verbose=True)
topics, probs = topic_model.fit_transform(docs)

id_topic = topic_model.get_topic_info()
print(topic_model.get_topic_info(0))
print(topic_model.get_topic_info)
id_topic.to_csv('../document/id_topic.csv') 

document_topic = topic_model.get_document_info(docs)
print(document_topic)
document_topic.to_csv('../document/document_topic.csv')

document_topic = pd.read_csv('../document/document_topic.csv') 
id_paper = document_topic.loc[:, ['Document', 'Topic']]
id_paper.to_csv('../document/id_paper.csv')

hierarchical_topics = topic_model.hierarchical_topics(docs)
print(hierarchical_topics)
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)



# topic_model = BERTopic() #BERTopic.load("MaartenGr/BERTopic_ArXiv")
