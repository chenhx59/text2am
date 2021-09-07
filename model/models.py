from unicodedata import bidirectional
import torch
import torch.nn as nn
from functools import reduce, partial
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence, pack_sequence
import torch.nn.functional as F
import itertools
from gensim.models import word2vec




class ModelOutputForPlan():
    '''
    生成solution文件所需要的输入
    '''
    def __init__(self, data: dict):
        '''self.data = data
        self.id = self.data['id']
        self.predicate = self.data['predicate']
        self.extract_params = self.data['extract_params']
        self.gold_params = self.data['gold_params']
        self.label = self.data['label']
        self.sentence = self.data['sentence']
        self.is_goal = data['is_goal']'''

        self.data = data
        self.id = self.data.get('id')
        self.predicate = self.data.get('predicate')
        self.extract_params = self.data.get('extract_params')
        self.gold_params = self.data.get('gold_params')
        self.label = self.data.get('label')
        self.sentence = self.data.get('sentence')
        self.is_goal = data.get('is_goal')





class MappingModel(torch.nn.Module):
    '''
    文本映射到词表.
    
    '''

    def __init__(self, args, from_w2v=True) -> None:
        super().__init__()
        self.args = args
        if from_w2v:
            self.embedding = self.get_embedding_layer_from_w2v('tmp/wv/wv.model', args.dataset.word2id, freeze=True)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embd_hid)
        # self.param_embdding = nn.Embedding(args.params_num, args.embd_hid)
        self.encoder = self.get_encoder('gru')
        self.pred_net = self.get_pred_net()
        self.sort_net = self.get_sort_net('o')

    def get_embedding_layer_from_w2v(self, w2v_model_file, dataset_word2id, freeze=False):
        model = word2vec.Word2Vec.load(w2v_model_file)
        wv = model.wv
        vocab_size = len(dataset_word2id.keys())
        weight = torch.randn((vocab_size, wv.vector_size))
        for word, idx in wv.key_to_index.items():
            weight[dataset_word2id[word]] = torch.tensor(wv.vectors[idx])
        
        return nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=freeze)

    def params_forward(self, params, hid):
        # pad_params = pad_sequence(params, batch_first=True)
        # pad_params = self.embedding(pad_params)
        # pack_padded_params = pack_padded_sequence(pad_params, lengths=list(map(len, params)), batch_first=True, enforce_sorted=False)
        # hid, _ = self.sort_net(pack_padded_params)
        # unpack_hid = pad_packed_sequence(hid, batch_first=True)

        ret = {
            'rep': [],
            'idx': []
        }
        for p, h in zip(params, hid):
            embd = self.embedding(p)
            out, _ = self.sort_net(embd)
            
            out = out[:, -1, :]
            logits = out.matmul(h)
            logits = F.softmax(logits)
            val, idx = logits.topk(1)
            ret['rep'].append(out[idx.item()])# MARK
            ret['idx'].append(p[idx.item()])# MARK
            s = out.size()
            h_s = _.size()
            # out = [i.dot(hid) for i in out]
        ret['rep'] = torch.stack(ret['rep'])# MARK
        ret['idx'] = pad_sequence(ret['idx'], batch_first=True)
        return ret
            
    

    def forward(self, batch, params=None, return_dict=False):
        batch_size = batch.size()[0]
        embd = self.embedding(batch)
        hid, _ = self.encoder(embd)
        hid = hid[:, -1, :]
        # MARK
        param_candidates = [self.get_params_candidate(i) for i in params]
        param_candidates = [torch.tensor(j) for j in param_candidates]
        param_candidates = [i.to(self.args.device) for i in param_candidates]
        # param_candidates = [torch.concat(i, hid) for i in param_candidates]
        # param_candidates = [self.sort_net(i) for i in param_candidates]
        ret = self.params_forward(param_candidates, hid)
        param_pred = ret['idx']
        param_hid = ret['rep']

        out = self.pred_net(hid)
        pred_hid = torch.mm(out, self.embedding.weight)

        if return_dict:
            return {
                'pred_out': out,
                'pred_hid': pred_hid,
                'params_hid': param_hid,
                'params_out': param_pred
            }
        return out, hid
    
    def get_params_candidate(self, params):
        ret = itertools.permutations(params, len(params))
        return list(ret)

    def get_encoder(self, enc_type='gru'):
        enc = None
        if enc_type == 'gru':
            enc = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=True
            )
        return enc
    
    def get_pred_net(self):
        pred_net = nn.Sequential(
            nn.Linear(self.args.hidden*2, self.args.vocab_size), 
            nn.Softmax(dim=1)
        )

        return pred_net

    def get_sort_net(self, key):
        if key == 'pn':
            return EncDec(self.args)
        rnn = nn.GRU(
            self.args.embd_hid,
            self.args.hidden * 2,
            # num_layers=self.args.encoder_num,
            batch_first=True,
            dropout=self.args.dropout,
            # bidirectional=True
        )
        
        return rnn

class StateModel(nn.Module):
    '''
    sentence to triplet
    multi triplet to state
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mm = MappingModel(self.args)
        self.state_encoder = self.get_state_encoder()
        
        self.pred_param_encoder = self.get_pred_param_encoder()

    def get_pred_param_encoder(self):
        return nn.Sequential(
            nn.Linear(self.args.hidden*3, self.args.hidden*2),
            nn.Linear(self.args.hidden*2, self.args.hidden)
        )
        

    
        

    def get_state_encoder(self):
        return F.adaptive_avg_pool2d

    def state_forward(self, hid, lengths):
        '''
        :param hid: tensor, pred_len * hid_size
        :param lengths: length of each state
        '''
        count = 0
        hid_size = hid.size()[1]
        ret = []
        for l in lengths:
            ret.append(self.state_encoder(hid[count: count+l].unsqueeze(0), (1, hid_size)).squeeze())
            count += l
        ret = torch.stack(ret)
        return ret

    def forward_once(self, batch, params, batch_type='sentence'):
        ret = {}
        params = reduce(lambda x, y: x + y, params)# MARK
        state_lens = [len(i) for i in batch]
        batch_sent = reduce(lambda x, y: x + y, batch)
        
        if batch_type == 'sentence':
            
            pad_batch_sent = pad_sequence(batch_sent, batch_first=True)
            sent_enc = self.mm(pad_batch_sent, params=params, return_dict=True)
            pred_hid, params_hid, pred_out, params_out = sent_enc['pred_hid'], sent_enc['params_hid'], sent_enc['pred_out'], sent_enc['params_out']
            ret['pred_out'] = pred_out
            ret['params_out'] = params_out
        elif batch_type == 'predicate':
            pred_hid = self.mm.embedding(torch.stack(batch_sent))
            
            pad_params = pad_sequence(params, batch_first=True)
            pad_params_embd = self.mm.embedding(pad_params)
            pack_padded_params = pack_padded_sequence(pad_params_embd, lengths=[len(i) for i in params], batch_first=True, enforce_sorted=False)
            packed_params_hid, _ = self.mm.sort_net(pack_padded_params)
            pad_packed_params_hid = pad_packed_sequence(packed_params_hid, batch_first=True)
            params_hid, idx = pad_packed_params_hid
            # MARK
            params_hid = torch.stack([params_hid[i, j, :] for i, j in zip(range(len(params_hid)), idx-1)])
        else:
            raise NotImplementedError()
            
        
        pred_params_cat = torch.cat((pred_hid, params_hid), dim=1)
        pred_params_hid = self.pred_param_encoder(pred_params_cat)
        state_hid = self.state_forward(pred_params_hid, state_lens)
        ret['state_hid'] = state_hid
        return ret

    def forward(self, batch_x, batch_y=None, x_params=None, y_params=None, **kw):
        x_len = kw.get('x_len')
        y_len = kw.get('y_len')
        x_state_ = self.forward_once(batch_x, x_params, batch_type='sentence')
        x_state, pred_out, params_out = x_state_['state_hid'], x_state_['pred_out'], x_state_['params_out']
        y_state = self.forward_once(batch_y, y_params, batch_type='predicate')['state_hid']
        return x_state, y_state, pred_out, params_out

    
    def calculate_loss(self, x, y):
        ret = torch.pow(F.pairwise_distance(x, y), 2)
        return torch.sum(ret)

class SimpleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embd_hid)
        self.ln_embd = nn.LayerNorm(self.args.embd_hid)
        self.encoder = nn.GRU(
            args.embd_hid, 
            args.hidden, 
            num_layers=args.encoder_num, 
            batch_first=True, 
            dropout=args.dropout, 
            bidirectional=True
        )
        self.ln_encoder = nn.LayerNorm(self.args.hidden*2)
        self.mlp = nn.Linear(
            args.hidden*2,
            args.n_catagory
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        embd = self.embedding(batch)
        embd = self.ln_embd(embd)
        hid, _ = self.encoder(embd)
        hid = hid[:, -1, :]
        hid = self.ln_encoder(hid)
        hid = self.mlp(hid)
        out = self.softmax(hid)
        return out



class EncDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc = self.get_encoder()
        self.dec = self.get_decoder()


    def forward(self, batch):
        pack_enc_out, enc_hid = self.enc(batch)
        dec_hid, _ = self.dec(batch)
        out = F.linear(dec_hid)
        out = F.softmax(out)
        return out

    def get_encoder(self):
        enc = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                # num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=False
            )
        return enc

    def get_decoder(self):
        dec = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                # num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=False
            )

        dec = nn.GRUCell(
            self.args.embd_hid,
            self.args.hidden
        )
        
        return dec

    


class AEModel(nn.Module):
    def __init__(self):
        pass