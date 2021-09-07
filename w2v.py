from gensim.models import word2vec
from model.dataset.data_reader import SimpleData, BaseData, StateData
import os
from config.config import args_type
from torchtext.vocab import Vectors
import torch.nn as nn
import torch


data = SimpleData('data/Newblocks/test.pkl', args_type=args_type)
sents = [data.data[i][1].lower().split(' ') for i in range(len(data.data))]
for idx, sent in enumerate(sents):
    if sent[0] == '':
        sents[idx].remove('')


model = word2vec.Word2Vec(sents, vector_size=128)
model.save('tmp/wv/wv.model')
model.wv.save_word2vec_format('tmp/wv/wv.txt')
wv = model.wv

wv_model_dir = 'tmp/wv'
os.makedirs(wv_model_dir, exist_ok=True)

wv_txt = os.path.join(wv_model_dir, 'wv.txt')
vectors = Vectors(wv_txt)

embd = nn.Embedding.from_pretrained(vectors.vectors, padding_idx=0)
vocab_size = len(data.id2word.keys())
weight = torch.randn((vocab_size, wv.vector_size))
for word, idx in wv.key_to_index.items():
    weight[data.word2id[word]] = torch.tensor(wv.vectors[idx])
embd = nn.Embedding.from_pretrained(weight, padding_idx=data.pad_id)
d = BaseData('data/Newblocks/test.pkl', args_type=args_type)
sd = StateData('data/Newblocks/test.pkl', args_type=args_type)
pass