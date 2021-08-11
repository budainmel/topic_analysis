"""
module of utility functions to do topic modelling
"""

import re 
import numpy as np
import pandas as pd

import scipy.sparse as ss
from boxx import g
from corextopic import corextopic as ct
from corextopic import vis_topic as vt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def remove_lead_trail_spaces(df_in):
    """
    Remove leading and trailing whitespace
    """

    return df_in.apply(lambda x: x.str.strip())

def remove_stopwords(comment_column, stopwords):
    """
    remove stopwords
    """
    
    return comment_column.apply(lambda x:" ".join([word for word in str(x).split() if word not in (stopwords)]))


def convert_to_lower(x):
    """
    Convert to lower case
    """
    return str(x).lower()



def text_pre_processor(lemmatizer, x):
    """
    Pre-process text
    """     
    # remove punctuations
    x = re.sub(r'[^\w\s$/]', '', x)
    # replace '/' and '-' with space
    x = re.sub(r'[/-]',' ', x)

    x_new = []
    for w in x.split():
        if len(w)>3:
            pass
        w = lemmatizer.lemmatize(lemmatizer.lemmatize(w, 'v'), 'n')
        x_new.append(w)
    return ' '.join(x_new)


def prepare_data(df, wnl, stop_words):
    df_processed = (
        df.set_axis(['comment_raw','rate'], axis=1, inplace=False)
        .dropna(axis=0, subset=['comment_raw'])
        .assign(comment=lambda x: x.comment_raw.astype(str).apply(str.lower))
        .assign(comment=lambda x: x.comment.apply(lambda y: text_pre_processor(wnl, y)))
        .assign(comment=lambda x: remove_stopwords(x.comment, stop_words))
    )

    return df_processed

def create_features(vectorizer, list_comments):
    doc_word = vectorizer.fit_transform(list_comments)
    doc_word = ss.csr_matrix(doc_word)

    # get words that label the columns
    words = list(np.asarray(vectorizer.get_feature_names()))

    # remove all integers from our set of words
    not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
    doc_word = doc_word[:, not_digit_inds]
    words = [word for ind,word in enumerate(words) if not word.isdigit()]

    return (doc_word, words)

def extract_topic(topic_model, df_processed):
    df_processed['top_topic_id'] = topic_model.p_y_given_x.argmax(1)
    df_processed['top_topic_prob'] = topic_model.p_y_given_x.max(1)
    
    df_labels = (
        pd.DataFrame(topic_model.p_y_given_x, index=df_processed.index)
        .rename(columns = lambda x: f'topic_{x}')
        .reset_index(drop=True)
    )

    df_processed = pd.concat([df_processed, df_labels], 1)

    topics = topic_model.get_topics(n_words=10)
    topics = [', '.join([e[0] for e in t]) for t in topics]
    df_processed['top_topic'] = df_processed.top_topic_id.apply(lambda x: topics[x])
    df_topics = (
        df_processed.groupby('top_topic')
        .top_topic_id.first().to_frame()
        .sort_values(by=['top_topic_id'])
    )

    df_processed = df_processed.reset_index(drop=True)
    column_ordered = (
        [e for e in df_processed.columns if not e.startswith('topic')]+
        [e for e in df_processed.columns if e.startswith('topic')]
    )
    df_processed = df_processed.reindex(columns=column_ordered)

    top_docs_id = df_processed.groupby('top_topic_id').top_topic_prob.nlargest(10).reset_index()['level_1']


    df_topics_top_docs = (
        df_labels.apply(lambda x: x.sort_values().tail(10).index).T.stack()
        .reset_index(level=1,drop=True).to_frame('comment_raw')
        .applymap(lambda x: df_processed.loc[x].comment_raw)
        .assign(top_topic_id=lambda x: x.index.str.split('_').str[-1].astype(int))
        .assign(top_topic=lambda x : x.top_topic_id.apply(lambda x: topics[x]))
        .sort_values(by='top_topic_id')
        [['top_topic_id', 'top_topic','comment_raw']]
    )

    return (df_processed, df_topics_top_docs, df_labels, topics, df_topics)