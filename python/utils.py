__author__ = 'ssktotoro2'

from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import re
from gensim import corpora, models
import pandas as pd
import bokeh.plotting as bp
from bokeh.objects import HoverTool

def apply_lda(text, num_topics):
    corpus_dictionary = corpora.Dictionary(text)
    corpus = [corpus_dictionary.doc2bow(words) for words in text]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus, id2word=corpus_dictionary, num_topics=num_topics, passes=10)
    corpus_lda = lda[corpus_tfidf]
    scipy_lda = gensim.matutils.corpus2csc(corpus_lda)
    ## check if sklearn accepts sparse vectors
    dense_lda = scipy_lda.todense().T
    test_Lda_doc = []
    
    for doc in corpus_lda:
        test_lda_doc.append(doc)
    max_prob_topic = []
    for doc in test_lda_doc:
        if len(doc) > 0:
            max_prob_topic.append(max(doc))
        else:
            max_prob_topic.append((0,0))
    max_topic_probability = pd.DataFrame(max_prob_topic)
    return max_topic_probability, dense_lda

def get_topics(topic):
    le = preprocessingLabelEncoder()
    le.fit(list(topic.unique()))
    topic_indices = le.tranform(topic.unique())
    return topic_indices

def apply_scaled_tsne(array, verbosity):
    tsne_embedding = TSNE(verbose=verbosity).fit_transform(array)
    scaled_tsne_embed = StandardScaler().fit_transform(tsne_embedding)
    

def show_tsne_viz(x, y, topics, words):
    """

    :param x:
    :param y:
    :param topics: should change to topic indices to match earlier function
    :param words:
    :return:
    """
    bp.output_notebook()
    bp.figure(plot_width=900, plot_height=700, title="Map by t-SNE", tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave"
              , x_axis_type=None, y_axis_type=None, min_border=1)
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])
    bp.scatter(
        x = X_embed_scaled[:,0],
        y = X_embed_scaled[:,1],
        radius= .01,
        color = colormap[topics],
        palette=["Spectral-11"],
        source=bp.ColumnDataSource({"line_descriptions": words})
    ).select(dict(type=HoverTool)).tooltips = {"rap_lyrics":"@words"}
    bp.show()


def preprocess_lyrics(lyrics):
    """

    :param lyrics:
    :return:
    """
    nlyrics = lyrics.str.lower()
    nlyrics = nlyrics.apply(lambda a: re.sub(r'\b\w{1,3}\b', '', a))
    nlyrics = nlyrics.apply(lambda a: (porter_stemmer.stem(wordnet_lemmatizer.lemmatize(a))))
    for word in punctuation:
        nlyrics = nlyrics.str.replace(word, '')
    nlyrics = nlyrics.apply(lambda a: re.sub(' +', ' ', a))
    glyrics_list = '\n'.join(nlyrics)
    glyrics_list = glyrics_list.split('\n')
    glyrics_list = [[word for word in verse.split()] for verse in glyrics_list]
    return glyrics_list
