import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem import *
import re
import nltk
import string
import numpy as np

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def remove_punctuation(text):
    try:
        return text.translate(str.maketrans('','',string.punctuation))
    except:
        print(text)

def remove_number(text):
    return re.sub(r'\d+', '', text)

def preprocess(text,pos_tag=False):
    result = str()
    text = remove_punctuation(text)
    text = remove_number(text)
    tokens = nltk.word_tokenize(text)
    if pos_tag:
        tokens = nltk.pos_tag(tokens)
        # for token in gensim.utils.simple_preprocess(text,deacc=True): # lowercase, tokenize, de-accent
        for token, pos in tokens:
            if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2 and pos not in pos_tag:
                result += ' ' + lemmatize_stemming(token)
    else:
        for token in tokens:
            if token not in gensim.parsing.preprocessing.STOPWORDS \
                    and len(token) > 2:
                result += ' ' + lemmatize_stemming(token)
    return result

# preprocess to remove weird character


def plot_top_10_words_from_count(count,count_vectorizer):
    '''
    count is a spare matrix
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    words = count_vectorizer.get_feature_names()
    total_count = np.zeros(len(words))
    for t in count:
        total_count += t.toarray()[0]

    count_dict = (zip(words,total_count))
    count_dict = sorted(count_dict,key=lambda x:x[1],reverse=True)[:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]

    xx = np.arange(0,len(words),1)

    plt.bar(xx,counts)
    plt.xticks(xx,words,rotation=15)
    plt.xlabel('word')
    plt.ylabel('count')
    plt.show()

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d: %s" % (topic_idx,", ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])))

def tsne_word_embedding(model,subsample=100,perplexity=15):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    w2v_model = model
    corpus_w2v = []
    list_vocab = list(w2v_model.wv.vocab.keys())
    n_sample = subsample

    list_vocab = np.random.choice(list_vocab, replace=False, size=n_sample)

    for word in list_vocab:
        corpus_w2v.append(w2v_model.wv.__getitem__([word]))

    corpus_w2v = np.array(corpus_w2v).reshape((n_sample, 300))

    pca_components = 50
    pca_reduced = PCA(pca_components).fit_transform(corpus_w2v)

    tsne_embedded = TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(pca_reduced)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in tsne_embedded[:, 0]],
                       'y': [y for y in tsne_embedded[:, 1]],
                       'words': list_vocab})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40})

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                weight='normal'
                ).set_size(15)

    plt.xlim(tsne_embedded[:, 0].min() - 50, tsne_embedded[:, 0].max() + 50)
    plt.ylim(tsne_embedded[:, 1].min() - 50, tsne_embedded[:, 1].max() + 50)

    plt.show()


