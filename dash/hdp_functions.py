
import re
import json
import pandas as pd
import numpy as np

# topic modeling
import tomotopy as tp
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

# plotly
import plotly
import plotly.express as px
from sklearn.manifold import TSNE

def HDPCreateModel(df, 
                     processed_col_name, 
                     termweight='IDF', 
                     min_cf=0, 
                     min_df=0,
                     rm_top=0, 
                     initial_k=20,
                     alpha=0.1,
                     eta=0.01,
                     gamma=0.1,
                     seed=999,
                     burn_in=10000):
    """
    Creates tomotopy HDP model.
    Wrapper function for creating a tomotopy HDPModel.
    https://bab2min.github.io/tomotopy/v0.12.2/en/#tomotopy.HDPModel

    ** Inputs **
    df: pd.DataFrame -> pandas DataFrame
    processed_col_name: str -> column in df with processed text (each cell should be a list of strings)
    termweight: str -> term weighting scheme in TermWeight.
    min_cf: int, optional -> minimum collection frequency of words
    rm_top: int, optional -> the number of top words to be removed. 
        If you want to remove too common words from model, you can set this value to 1 or more. 
    initial_k: int, optional -> the initial number of topics between 2 ~ 32767 The number of topics will be adjusted for data during training.
    alpha: float, optional -> concentration coeficient of Dirichlet Process for document-table
    eta: float, optional -> hyperparameter of Dirichlet distribution for topic-word
    gamma: float, optional -> concentration coeficient of Dirichlet Process for table-topic
    seed: int, optional -> random seed
    burn_in: the burn-in iterations for optimizing parameters

    ** Returns **
    1. tomotopy.HDPModel -> description here
    2. topic_labels: dictionary -> dictionary of key value (int, string) pairs; key=topic number, value=topic from HDP model (concatenated list of string)
    3. top_labels_array: list -> list of strings of each topic number and topic
    """
    if termweight == 'IDF':
        term_weight = tp.TermWeight.IDF
    elif termweight == 'ONE':
        term_weight = tp.TermWeight.ONE
    else:
        term_weight = tp.TermWeight.IDF
    
    # Initialize tomotopy HDPModel
    mdl = tp.HDPModel(tw=term_weight, 
                      min_cf=min_cf, 
                      min_df=min_df,
                      rm_top=rm_top, 
                      initial_k=initial_k,
                      alpha=alpha,
                      eta=eta,
                      gamma=gamma,
                      seed=seed)
    
    # List of lists
    # [['word1', 'word2', ...],
    # ['word1', 'word2', ...]],
    word_list_lemmatized  = df[processed_col_name].tolist()
    
    # Add each document to HDP mdl
    tick = 0
    for vec in word_list_lemmatized:
        if len(vec) > 0:
            mdl.add_doc(vec)
        else:
            mdl.add_doc([''])
            print(tick)
        tick = tick + 1
    
    # Initiate sampling burn-in  (i.e. discard N first iterations)
    mdl.burn_in = burn_in
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs,
          ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    
    # Train model
    for i in range(0, 2000, 100):
        mdl.train(10) # 100 iterations at a time
        print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, mdl.ll_per_word, mdl.live_k))
    
    # Get topics
    topics = get_hdp_topics(mdl, top_n=10) # changing top_n changes no. of words displayed
    
    # Get topic labels
    top_labels_array = []
    for key, value in topics.items():
        top_labels_array.append(str(key) + ": " + ' '.join(tup[0] for tup in value))
    
    # Top labels dict
    top_labels = {}
    counter = 0
    for key, value in topics.items():
        top_labels[counter] = str(key) + ": " + ' '.join(tup[0] for tup in value)
        counter = counter+1

    #Evaluate coherence score
    coherence_scores = eval_coherence(topics, word_list_lemmatized)
    print("Coherence score: " + str(coherence_scores) + '\n')
    
    #print(mdl.summary())
    
    return mdl, top_labels, top_labels_array


def HDPCreatePlotly(df, hdpModel, top_labels, processed_col_name):
    """
    Create plotly scatter plot from tomotopy HDP model.

    ** Inputs **
    df: pd.DataFrame -> pandas DataFrame
    hdpModel:obj -> tomotopy HDPModel trained model
    top_labels: the outputted topic_labels from HDPCreateModel
    processed_col_name: string -> str -> column in df with processed text (each cell should be a list of strings)

    ** Returns **
    1. graphJSON: a plotly express figure (fig) in the form of JSON dump to render it on a webpage
    2. top_labels: same as input
    """
    # Convert tomotopy HDPModel to tomotopy LDAModel
    # to graph it on Plotly
    new_lda_model, new_topic_id = hdpModel.convert_to_lda()

    # For each row in the df, call LDAModel.get_topic_dist() to get a distribution of topics in the document,
    # and add that array to list_of_weights
    list_of_weights = []
    for i, row in df.iterrows():
        weights = new_lda_model.docs[i].get_topic_dist()
        list_of_weights.append(weights)

    # Convert list_of_weights to numpy.ndarray...
    # arr will have a shape (num_documents, num_topics)
    # each row will correspond to a document
    # and the row will be a distribution of topics
    arr = pd.DataFrame(list_of_weights).fillna(0).values
    # Get winning topic number for each list_of_weights
    topic_num = np.argmax(arr, axis=1)
    # Add winning topic as a column in df
    topic_num = pd.DataFrame(topic_num)
    df['winning_topic_num'] = topic_num
    
    # Create winning topic label (concatenate strings) and add to dataframe
    temp = df['winning_topic_num'].to_numpy().tolist()
    winning_topic = []
    for i in temp:
        winning_topic.append(top_labels[i])
    winning_topic = pd.DataFrame(winning_topic)
    df['winning_topic'] = winning_topic
    
    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Add output of tsne model to dataframe
    df['X_tsne'] =tsne_lda[:, 0]
    df['Y_tsne'] =tsne_lda[:, 1]
    df['size'] = df[processed_col_name].apply(lambda x: len(x))

    # Create plotly express figure
    fig = px.scatter(df, 
                    x="X_tsne", 
                    y="Y_tsne", 
                    color="winning_topic",
                    size="size", 
                    symbol=df['winning_topic'],
                    symbol_sequence= ['circle', 'square', 'diamond'],
                    hover_data=['dc_contributor_author', 'dc_title', 'dc_date_published']
                    #hover_data=['title_manual', 'year_manual']
                    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, top_labels

# Get HDP topics (https://github.com/ecoronado92/towards_data_science/blob/master/hdp_example/scripts/model_funcs.py)
def get_hdp_topics(hdp, top_n=10):
    '''HELPER FUNCTION
    Wrapper function to extract topics from trained tomotopy HDP model 
    
    ** Inputs **
    hdp:obj -> tomotopy HDPModel trained model
    top_n: int -> top n words in topic based on frequencies
    
    ** Returns **
    topics: dict -> per topic, an arrays with top words and associated frequencies 
    '''
    
    # Get most important topics by # of times they were assigned (i.e. counts)
    sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x:x[1], reverse=True)]

    topics = dict()
    
    # For topics found, extract only those that are still assigned
    for k in sorted_topics:
        if not hdp.is_live_topic(k): continue # remove un-assigned topics at the end (i.e. not alive)
        topic_wp = []
        for word, prob in hdp.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp # store topic word/frequency array
        
    return topics

# Get coherence score (https://github.com/ecoronado92/towards_data_science/blob/master/hdp_example/scripts/model_funcs.py)
def eval_coherence(topics_dict, word_list, coherence_type='c_v'):
    '''Wrapper function that uses gensim Coherence Model to compute topic coherence scores
    
    ** Inputs **
    topic_dict: dict -> topic dictionary from train_HDPmodel function
    word_list: list -> lemmatized word list of lists
    coherence_typ: str -> type of coherence value to comput (see gensim for opts)
    
    ** Returns **
    score: float -> coherence value
    '''
    
    # Build gensim objects
    vocab = corpora.Dictionary(word_list)
    corpus = [vocab.doc2bow(words) for words in word_list]
    
    # Build topic list from dictionary
    topic_list=[]
    for k, tups in topics_dict.items():
        topic_tokens=[]
        for w, p in tups:
            topic_tokens.append(w)
            
        topic_list.append(topic_tokens)
            

    # Build Coherence model
    print("Evaluating topic coherence...")
    cm = CoherenceModel(topics=topic_list, corpus=corpus, dictionary=vocab, texts=word_list, 
                    coherence=coherence_type)
    
    score = cm.get_coherence()
    print ("Done\n")
    return score