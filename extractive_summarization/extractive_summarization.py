# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import hdbscan
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.summarization.summarizer import summarize as gensum
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min

import re
name_sent_skip = re.compile(r'\b([A-Za-z])(\.)')
sentence_encode_hub = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
sess = tf.Session()
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None,))
embed = hub.Module(sentence_encode_hub)
encoding_tensor = embed(similarity_input_placeholder)
init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
sess.run(init_op)

try:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
except:
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import sys


reload(sys)
sys.setdefaultencoding('UTF8')

import unicodedata


def process_summarize(text, sents, sess=sess):
    if sents:
        sents = int(sents)
    text = unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore')
    text = name_sent_skip.sub(r'\1', text)
    sent_collection = [process(text)]
    if len(sent_collection[0]) >= 2:
        embeddings = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: sent_collection[0]})
        results = []
        start, end = 0, 0
        for i in sent_collection:
            results.append(embeddings[start:start + len(i)])
            start += len(i)
        summaries_kmeans, summaries_hdb = summarize(results, sent_collection, sents)
        summary_kmeans, summary_hdb, spl = summaries_kmeans[0], summaries_hdb[0], None
        return {'summary_kmean': summary_kmeans, 'summary_hdb': summary_hdb,
                'summary_text_rank': gensum(text=text, ratio=0.1) if gensum(text=text, ratio=0.1)
                else gensum(text=text, ratio=0.0333) if gensum(text=text, ratio=0.0333)
                else gensum(text=text, ratio=0.333)}
    else:
        return {'summary_kmean': None, 'summary_hdb': None,
                'summary_text_rank': None}


def process(text):
    text = unicode(text.lower())
    sents = sent_detector.tokenize(text)
    sents = [e.strip() for sent in sents for e in sent.split('.') if e]
#     sents = [e for sent in sents for e in sent.split(',') if e]
    return sents


def summarize(results, sent_collection, sents):
    summaries_kmeans = [None] * len(results)
    summaries_hdb = [None] * len(results)
    for inx, i in enumerate(results):
        n_clusters = min(int(np.ceil(len(i) ** 0.5)), 5, sents)
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=max(int(np.ceil(len(i)/10.0)), 2))
        distance = pairwise_distances(i, metric='cosine')
        db = clusterer.fit(distance.astype('float64'))
        print db.labels_.max()
        hdb_clusters = min(db.labels_.max()+1, n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(i)
        avg_kmeans = []
        closest_hdb = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg_kmeans.append(np.mean(idx))
        if db.labels_.max() == -1:
            closest_hdb = [(0, 1.0)]
        else:
            for j in range(hdb_clusters):
                ix = np.where(db.labels_ == j)[0]
                closest = sorted(enumerate(ix), key=lambda (k, v): db.probabilities_[ix][k], reverse=True)
                closest_hdb.append((closest[0][1], db.probabilities_[closest[0][1]]))
        closest_kmeans, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, i)
        ordering = sorted(range(n_clusters), key=lambda k: avg_kmeans[k])
        summaries_kmeans[inx] = '. '.join([sent_collection[inx][closest_kmeans[indx]] for indx in ordering])
        closest_hdb = sorted(closest_hdb, key=lambda x: x[1], reverse=True)
        summaries_hdb[inx] = '. '.join(list(set([sent_collection[inx][closest_i[0]] for closest_i in closest_hdb])))
    return summaries_kmeans, summaries_hdb