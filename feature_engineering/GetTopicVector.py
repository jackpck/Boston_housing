import pandas as pd
import numpy as np
from nlp_functions import preprocess,plot_top_10_words_from_count,print_topics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF


property_type = 'condo'
df = pd.read_csv('./data/' + 'Boston_%s_remarks.csv'%property_type,index_col=0)

print(df['REMARKS'].iloc[0])

df_processed = df['REMARKS'].apply(lambda x: preprocess(x,pos_tag=['NNP','VBP']))

ngram = 1
tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                   analyzer='word',
                                   strip_accents='ascii',
                                   ngram_range=(ngram,ngram))

corpus_tfidf = tfidf_vectorizer.fit_transform(df_processed.astype(str))

n_topic = 10

nmf = NMF(n_components=n_topic, random_state=101).fit(corpus_tfidf)
nmf.fit(corpus_tfidf)

n_top_words = 10
print_topics(nmf, tfidf_vectorizer, n_top_words)

words = np.array(tfidf_vectorizer.get_feature_names())

topic_word_vectors = []
for i in range(n_topic):
    topic_word_vectors.append(words[nmf.components_[i].argsort()[:-n_top_words - 1:-1]])


def topic_similarity(remark):
    return ' '.join('%.2f'%(len(set(remark.split(' ')).intersection(set(topic_word_vectors[i]))) / n_top_words)
            for i in range(n_topic))


df_topic_vector = df_processed.apply(topic_similarity)

df_topic_vector.to_frame().to_csv('./data/' + 'Boston_%s_topic_scores.csv'%property_type)

