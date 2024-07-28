
import re
import json
import pandas as pd
import numpy as np

# Plotly
import plotly
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

# Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore


def LDACreateModel(df, num_topics, processed_col_name):
    '''
    Creates Gensim LDA model
    Input: 
        df: pd.DataFrame -> pandas DataFrame
        num_topics: int -> number of topics for the LDA model to create
        processed_col_name: string -> column in df with processed text (each cell should be a list of strings)
    '''
    print("From lda_functions.py: Running LDACreateModel...")
    # create_lda_model
    # Create Dictionary
    id2word = corpora.Dictionary(df[processed_col_name])
    id2word.filter_extremes(no_below=2, no_above=0.5) # filter extremes
    # Create Corpus
    texts = df[processed_col_name]
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Instantiating a Base LDA model 
    base_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                num_topics=num_topics, 
                                                id2word=id2word, 
                                                update_every=1,
                                                chunksize=100,
                                                passes=10, 
                                                alpha='auto',
                                                per_word_topics=True,
                                                random_state=99)
    # Filtering for words 
    words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics(num_topics)]
    # Create Topics
    topics = [' '.join(t[0:10]) for t in words]
    
    print(f'\tLDA model created with {len(topics)} topics')

    return base_model, corpus

def LDACreatePlotly(df, model, corpus, num_topics, processed_col_name):
    print("From lda_functions.py: Running LDACreatePlotly...")
    '''
    Input:
        df: pd.DataFrame -> pandas DataFrame
        model: Gensim LDA model
        corpus:
        num_topics:
        processed_col_name:

    Returns:
        graphJSON:
        topics:
    '''
    # Filtering for words 
    words = [re.findall(r'"([^"]*)"',t[1]) for t in model.print_topics(num_topics)]
    # Create Topics
    topics = [' '.join(t[0:10]) for t in words]
    
    top_labels = {}
    # Create topic labels
    for id, t in enumerate(topics): 
        displayId = id + 1
        top_labels[id] = str(displayId) + " " + t

    # TSNE
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(model[corpus]):
        weights = [0] * num_topics
        for index, weight in row_list[0]:
            weights[index] = weight
        topic_weights.append(weights)
    print(len(topic_weights))

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    topic_num = np.argmax(arr, axis=1)
    topic_num = pd.DataFrame(topic_num)
    df['winning_topic_num'] = topic_num

    temp = df['winning_topic_num'].to_numpy().tolist()
    winning_topic = []
    for i in temp:
        winning_topic.append(top_labels[i])
    winning_topic = pd.DataFrame(winning_topic)
    df['winning_topic'] = winning_topic

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    df['X_tsne'] =tsne_lda[:, 0]
    df['Y_tsne'] =tsne_lda[:, 1]
    df['size'] = df[processed_col_name].apply(lambda x: len(x))


    # fig = px.scatter(df, x="X_tsne", y="Y_tsne", color="winning_topic",
    #                 size="size", hover_data=['title_manual', 'authors_manual', 'year_manual'])
    fig = px.scatter(df, 
                    x="X_tsne", 
                    y="Y_tsne", 
                    color="winning_topic",
                    size="size", 
                    hover_data=['dc_contributor_author', 'dc_title', 'dc_date_published'],
                    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, topics


# Helper function(s)
def text_to_sentence(word_list):
    sentence = ' '.join(x for x in word_list)
    return sentence
