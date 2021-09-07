from operator import is_
import os
import nltk
import torch
import pickle as pkl
import logging
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from collections import Counter
import tqdm
from gensim.corpora import Dictionary
from functools import reduce

class BaseData(Dataset):
    def __init__(self, file_in, cache='tmp/dataset/', args_type=None) -> None:
        super().__init__()
        os.makedirs(cache, exist_ok=True)
        _, file_in_suffix = os.path.split(file_in)
        # cache_file = os.path.join(cache, f'cache_data_{file_in_suffix}')
        self.data = []
        self.pad_id = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.parameter_tokens = list(args_type.keys()) if args_type is not None else None
        with open(file_in, 'rb') as f:
            self.raw = pkl.load(f)
        
        
        label_file = os.path.join(cache, f'cache_label_{file_in_suffix}')
        if os.path.exists(label_file):
            logger.info(f'loading labels from {label_file}.')
            with open(label_file, 'rb') as f:
                self.labels = pkl.load(f)
        else:
            _states = [i['initial_state'] + reduce(lambda x,y: x+y, i['state'][1:]) for i in self.raw]
            _states = reduce(lambda x,y: x+y, _states)
            _labels = [i.split(' ')[0][1:] for i in _states]
            self.labels = list(set(_labels))
            logger.info(f'dumping labels to {label_file}.')
            with open(label_file, 'wb') as f:
                pkl.dump(self.labels, f, protocol=pkl.HIGHEST_PROTOCOL)
        self.n_catagory = len(self.labels) 
        self.sents = []
        for data in self.raw:
            for t in data['text_trace']:
                self.sents += self.pre_process_raw_text(t)

        
        corpus_file = os.path.join(cache, f'cache_corpus_{file_in_suffix}')
        if os.path.exists(corpus_file):
            self.dictionary = Dictionary.load(corpus_file)
        else:
            sents = [nltk.word_tokenize(i) for i in self.sents]
            self.dictionary = Dictionary(sents)
            special_tokens = {self.pad_token: self.pad_id}
            self.dictionary.patch_with_special_tokens(special_tokens)
            self.dictionary.add_documents([[self.unk_token]])
            self.dictionary.save(corpus_file)
        self.word2id = self.dictionary.token2id
        self.id2word = {}
        for k, v in self.word2id.items():
            self.id2word[v] = k
        self.vocab_size = len(self.id2word.items())
        
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.sents[index]

    def extact_params_from_sentence(self, sent):
        tokens = nltk.word_tokenize(sent)
        params = []
        if self.parameter_tokens is None:
            return params
        for token in tokens:
            for param in self.parameter_tokens:
                if token == param:
                    params.append(token)
        return params
  
    def decode_sentence(self, tokens):
        res = [self.id2word[idx] for idx in tokens]
        sent = ' '.join(res)
        return sent

    def batch_decode_sentence(self, batch):
        res = [
            [self.id2word[idx] for idx in tokens] for tokens in batch
        ]
        sents = [
            ' '.join(tokens) for tokens in res
        ]
        return sents

    def pre_process_raw_text(self, text):
        '''
        将原始数据中的一个text（包含多个句子，句子间由. 隔开）
        :param text: 一条文本
        :return: list
        '''
        ret = []
        t = text.split('.')[:-1]
        for sent in t:
            sent = sent.lower()
            if sent[0] == ' ':
                sent = sent[1:]
            ret.append(sent)
        return ret

class SimpleData(BaseData):
    '''
    一次返回一条处理后的句子及相关标签下标等
    '''

    def __init__(self, file_in, cache='tmp/dataset/', args_type=None) -> None:
        super().__init__(file_in, cache, args_type)
        _, file_in_suffix = os.path.split(file_in)
        cache_file = os.path.join(cache, f'cache_simpledata_{file_in_suffix}')
        self.data = []

        if os.path.exists(cache_file):
            logger.info(f'Loading data from cache {cache_file}')
            with open(cache_file, 'rb') as f:
                self.data = pkl.load(f)
        else:
            for data in self.raw:
                plan_idx = data['id']
                actions = data['plan']
                goal_state = data['goal_state']
                data['state'][0] = data['initial_state']
                # t for text, s for state
                for trace_idx, t, s in zip(range(len(data['text_trace'])), data['text_trace'], data['state']):
                    t = self.pre_process_raw_text(t)
                    is_last_state = True if trace_idx == len(data['text_trace'])-1 else False
                    
                    for predicate_idx, sent, pred in zip(range(len(t)), t, s):
                        
                        
                        is_goal = True if is_last_state and pred in goal_state else False
                        pred = pred.replace('(', '')
                        pred = pred.replace(')', '')
                        gold_params = pred.split(' ')[1:]
                        pred = pred.split(' ')[0]
                        
                        extract_params = self.extact_params_from_sentence(sent)
                        self.data.append(([plan_idx, trace_idx, predicate_idx, gold_params, extract_params, is_goal], sent, pred))
    
            with open(cache_file, 'wb') as f:
                logger.info(f'Dumping dataset to {cache_file}')
                pkl.dump(self.data, f, protocol=pkl.HIGHEST_PROTOCOL)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, x, y = self.data[index]
        tokens = nltk.word_tokenize(x)
        ids = [self.word2id[tok] for tok in tokens]
        y = self.labels.index(y)
        return idx, torch.tensor(ids), torch.tensor(y)

class StateData(BaseData):
    def __init__(self, file_in, cache='tmp/dataset', args_type=None) -> None:
        super().__init__(file_in, cache=cache, args_type=args_type)
        _, file_in_suffice = os.path.split(file_in)
        cache_file = os.path.join(cache, f'cache_statedata_{file_in_suffice}')

        if os.path.exists(cache_file):
            logger.info(f'Loading data from cache {cache_file}')
            with open(cache_file, 'rb') as f:
                self.data = pkl.load(f)
        else:
            for data in self.raw:
                plan_id = data['id']
                goal_state = data['goal_state']
                data['state'][0] = data['initial_state']
                for trace_idx, t, s in zip(range(len(data['text_trace'])), data['text_trace'], data['state']):
                    t = self.pre_process_raw_text(t)
                    is_last_state = True if trace_idx == len(data['text_trace'])-1 else False
                    sents = []
                    labels = []
                    gold_param_list = []
                    extract_param_list = []
                    for predicate_idx, sent, pred in zip(range(len(t)), t, s):
                        is_goal = True if is_last_state and pred in goal_state else False
                        pred = pred.replace('(', '')
                        pred = pred.replace(')', '')
                        gold_params = pred.split(' ')[1:]
                        pred = pred.split(' ')[0]
                        extract_params = self.extact_params_from_sentence(sent)
                        sents.append(sent)
                        labels.append(pred)
                        gold_param_list.append(gold_params)
                        extract_param_list.append(extract_params)
                    self.data.append(
                        {
                            'plan_idx': plan_id,
                            'trace_idx': trace_idx,
                            'gold_params_list': gold_param_list,
                            'extract_params_list': extract_param_list,
                            'is_goal': is_goal,
                            'sentences': sents,
                            'predicates': labels
                        }
                    )
            with open(cache_file, 'wb') as f:
                logger.info(f'Dumping dataset to {cache_file}')
                pkl.dump(self.data, f, protocol=pkl.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        tokens = [nltk.word_tokenize(sent) for sent in data['sentences']]
        ids = [
            torch.tensor([self.word2id[tok] for tok in j]) for j in tokens
        ]
        y = [
            torch.tensor(self.labels.index(pred)) for pred in data['predicates']
        ]
        others = {}
        for k, v in data.items():
            if k in ['sentences', 'predicates']:
                continue
            others[k] = v
        params = others['extract_params_list']
        params = [torch.tensor([self.word2id[word] for word in i]) for i in params]
        others['extract_params_logits'] = params
        others['gold_params_logits'] = [torch.tensor([self.word2id[word]for word in i]) for i in others['gold_params_list']]
        return ids, y, others



if __name__ == '__main__':
    dataset = SimpleData(file_in='data/Newblocks/train.pkl')
    print(dataset[0])
    print(len(dataset))