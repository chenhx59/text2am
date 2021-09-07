from model.models import SimpleModel
from model.dataset.data_reader import SimpleData
from torch.utils.data import DataLoader, Sampler
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import argparse
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

log_steps = 100
current_step = 0
epoch = 4

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

dataset = SimpleData('data/Newblocks/train.pkl')
parser = argparse.ArgumentParser()

args = parser.parse_args()

args.vocab_size = len(dataset.id2word.keys())
args.embd_hid = 128
args.hidden = 128
args.encoder_num = 4
args.dropout = 0.0
args.n_catagory = dataset.n_catagory

def collate_fn(data):
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    xs = [i[1] for i in data]
    ys = [i[2] for i in data]
    
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    seq_len = [i.size(0) for i in xs]
    # xs = pack_padded_sequence(xs, seq_len, batch_first=True)
    return xs, torch.tensor(ys)

loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


model = SimpleModel(args)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
loss_fn = nn.CrossEntropyLoss()


for _ in range(epoch):
    for xs, ys in loader:
        xs = xs.to(device)
        ys = ys.to(device)
        optimizer.zero_grad()
        preds = model(xs)
        loss = loss_fn(preds, ys)
        loss.backward()
        optimizer.step()
        if current_step % log_steps == 0:
            logger.info(f'step:{current_step}, loss:{loss}')
            logger.info(f'sentence: {dataset.decode_sentence(xs[2].tolist())}, predicate:{dataset.labels[int(preds[2].topk(1)[1])]}, gold: {dataset.labels[int(ys[2])]}')
            # logger.info(f'sentence:{}')
        current_step += 1

print('ok')

