from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
import pickle as pkl
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import tqdm
# from model.dataset.data_reader import TextActionModelData
from model.dataset.data_reader import SimpleData
from model.dataset.utils import format_sent_with_args
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def doc2vec_model(dataset: SimpleData):
    sent_list = [
        (i[0], dataset.decode_sentence(i[1].tolist())) for i in dataset
    ]
    idx = [
        '{}_{}_{}'.format(i[0][0], i[0][1], i[0][2]) for i in sent_list #i[0] : [plan_idx, state_idx, predicate_idx]
    ]
    sent_list1 = [
        i[1].split(' ') for i in sent_list
    ]
    tagged_sent_list = [
        TaggedDocument(sent, [idx]) for idx, sent in enumerate(sent_list1)
    ]
    model = Doc2Vec(tagged_sent_list, vector_size=128)
    model.train(tagged_sent_list, epochs=4)
    return model


def get_sentences_predicate_pair_from(file_in) -> dict:
    sents = []
    preds = []
    with open(file_in, 'rb') as f:
        while True:
            try:
                data = pkl.load(f)
                text_trace = data['text_trace']
                state = data['state']
                state[0] = data['initial_state']
                for each_text, each_state in zip(text_trace, state):
                    t = each_text.split('. ')[:-1]
                    p = [
                        st.replace('(', '').replace(')', '').split(' ')[0] for st in each_state
                    ]
                    sents += t
                    preds += p
            except:
                break
    assert len(sents)==len(preds)
    return {
        'sentences': sents,
        'predicates': preds
    }
                

def tf_idf(doc, corpus):
    tokens_set = list(set(corpus.tokens))
    dim = len(set(corpus.tokens))
    vec = np.zeros(dim)
    for idx, token in enumerate(tokens_set):
        vec[idx] = corpus.tf_idf(token, doc)
    return vec

def dimention_reduce_and_visualize(vectors, preds, labels):
    colors = 'bcgkmrwy'        # 蓝色 蓝绿色 绿色 黑色 紫红色 红色 白色 黄色
    color_idx = 0
    fig = plt.figure(1, figsize=(6, 4))
    new_vectors = TSNE(n_components=2).fit_transform(vectors)
    dim1 = new_vectors[:, 0]
    dim2 = new_vectors[:, 1]
    for label in labels:
        idx = np.argwhere(preds==label).reshape(-1).tolist()
        xs = dim1[idx]
        ys = dim2[idx]
        
        plt.scatter(xs, ys, c=colors[color_idx])
        color_idx += 1
    plt.show()
def bow(docs):
    pass


def get_corpus(file_in, args_type):
    sents = []
    with open(file_in, 'rb') as f:
        while True:
            try:
                data = pkl.load(f)
                text_trace = data['text_trace']
                for each in text_trace[:-1]:
                    sents += each.split('.')
            except:
                break 
    sents = [
        word_tokenize(format_sent_with_args(sent, args_type)) for sent in sents
    ]
    corpus = TextCollection(sents)
    return corpus
