from query_topic import searchTopic, searchPaper
from html_custom import set_background

import streamlit as st
import pandas as pd
import numpy as np

set_background('background.png')

st.header('NLPThemeWaver')
st.write("Search one NLP topic and see how it connects with others.")

tab1, tab2 = st.tabs(["Topic Searching", "Paper Searching"])

# --------Topic Searching--------
with tab1:
    st.subheader("Topic Searching")

    query_topic = st.text_input(
        "Enter the topic you want: ",\
        placeholder="\"text-based emotion analysis\" | \"text-based, emotion analysis\"",\
        help="Enter the number of relevant topic returned."
        )
    top_n = st.text_input(
        "Number of topics to display: ",\
        placeholder="< 842",\
        help="Total 842 topics."
        )
                        

    if st.button('Submit', key="topic_query"): # type="primary"
        with st.spinner("Querying..."):
            i2t_filename = '../id_topic.csv'
            querying = searchTopic(query_topic, int(top_n), i2t_filename)
            topic_result = querying.query_to_topic()

            st.session_state['topic_result'] = topic_result

    if 'topic_result' in st.session_state:
        st.write(st.session_state['topic_result'])




# --------Paper Searching--------
with tab2:
    st.subheader("Paper Searching")

    query_topic = st.text_input(
        "Enter the topic you want: ",\
        placeholder="\"text-based emotion analysis\" | \"text-based, emotion analysis\"",\
        help="Adding synonyms by using the '(word|word)' format."
        )
    topic_id = st.text_input(
        "Topic ID: ",\
        placeholder="\" \", \"1\" or \"1, 2, ...\"",\
        help="Obtain Topic IDs from 'Topic Searching' to search specific papers group. \n Without any ID, the system searches the entire database."
        )
    paper_number = st.text_input(
        "Number of papers to display: ",\
        placeholder="any number")


    if st.button('Submit', key="paper_query"):
        with st.spinner("Querying..."):
            i2p_filename = '../document_topic.csv'
            origin_filename = '../ACL_Anthology_(092023).csv'

            searching = searchPaper(query_topic, topic_id, int(paper_number), i2p_filename, origin_filename)
            paper_result = searching.get_related_paper()

            st.session_state['result'] = paper_result

    if 'result' in st.session_state:    
        st.write(st.session_state['result'])
