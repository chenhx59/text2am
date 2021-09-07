import torch
from torch import optim
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import pickle as pkl
from model.models import ModelOutputForPlan, StateModel
from functools import reduce
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def eval(args, model, dataset):
    model.eval()
    eval_loader = DataLoader(dataset, shuffle=False)
    pass


def predict(args, model, dataset):
    result = []
    current_step = 0

    model.eval()
    def collate_fn(data):
        xs = [i[1] for i in data]
        ys = [i[2] for i in data]
        idx = [i[0] for i in data]
        xs = pad_sequence(xs, batch_first=True, padding_value=dataset.pad_id)
        return idx, xs, torch.tensor(ys)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=collate_fn
    )
    with torch.no_grad():
        
        for idx, xs, ys in loader:
            xs = xs.to(args.device)
            ys = ys.to(args.device)
            predictions = model(xs)
            for id, x, y, pred in zip(idx, xs, ys, predictions[0]):
                gold_params = id[3]
                extract_params = id[4]
                is_goal = id[5]
                id = id[:3]
                nx = dataset.decode_sentence(x.tolist())
                y = dataset.labels[y.item()]
                pred = [i for i in pred.topk(1)[1].tolist()]
                pred = dataset.decode_sentence(pred)
                
                data =    {
                        'id': id,
                        'predicate': pred,
                        'extract_params': extract_params,
                        'gold_params': gold_params,
                        'label': y,
                        'sentence': nx, 
                        'is_goal': is_goal
                    }
                result.append(ModelOutputForPlan(data))
                
        result.sort(key=lambda x: x.id[0])
        with open(args.prediction_out_file, 'wb') as f:
            pkl.dump(result, f)
            
            
            

def train(args, model, dataset):
    
    # TODO tensorboard

    current_step = 0
    total_loss = 0.0
    logging_loss = 0.0

    def collate_fn(data):
        xs = [i[1] for i in data]
        ys = [i[2] for i in data]
        # idx = [i[0] for i in data]
        xs = pad_sequence(xs, batch_first=True, padding_value=dataset.pad_id)
        return xs, torch.tensor(ys)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=collate_fn
    )

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(args.epoch):
        for xs, ys in loader:
            model.train()
            xs = xs.to(args.device)
            ys = ys.to(args.device)
            optimizer.zero_grad()
            preds = model(xs)
            loss = loss_fn(preds, ys)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()
            if current_step % args.log_steps == 0:
                logger.info(f'step:{current_step}, loss:{(total_loss-logging_loss)/args.log_steps}')
                logging_loss = total_loss
                logger.info(f'sentence: {dataset.decode_sentence(xs[2].tolist())}, predicate:{dataset.labels[int(preds[2].topk(1)[1])]}, gold: {dataset.labels[int(ys[2])]}')
            
            current_step += 1
    return total_loss, current_step


def train_state_model(args, model: StateModel, dataset):
    current_step = 0
    total_loss = 0.0
    logging_loss = 0.0

    def collate_fn(data):
        x = [i[0] for i in data]
        x_len = [len(i) for i in x]
        x_flat = reduce(lambda x, y: x + y, x)
        y = [i[1] for i in data]
        y_len = [len(i) for i in y]
        y_flat = reduce(lambda x, y: x + y, y)
        label2word = {
            0: 13,# ready
            1: 3, # table
            2: 2, # on
            3: 40, # holding
            4: 16, #'nothing'
        }
        y_flat = [torch.tensor(label2word(i.item())) for i in y_flat]
        others = [i[2] for i in data]
        gold_params = [i['gold_params_logits'] for i in others]
        gp_flat = reduce(lambda x, y: x + y, gold_params)
        extract_params = [i['extract_params_logits'] for i in others]
        ep_flat = reduce(lambda x, y: x + y, extract_params)
        return x_flat, y_flat, ep_flat, gp_flat, x_len, y_len
        #return x, y, gold_params, extract_params
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    for _ in range(args.epoch):
        #for xs, ys, g, e in loader:
        for xs, ys, e, g, x_len, y_len in loader:
            model.train()
            xs = [[i.to(args.device) for i in j]for j in xs]
            ys = [[i.to(args.device) for i in j]for j in ys]
            e = [[i.to(args.device) for i in j]for j in e]
            g = [[i.to(args.device) for i in j]for j in g]
            optimizer.zero_grad()
            x, y, pred_out, params_out = model(xs, ys, e, g)
            loss = model.calculate_loss(x, y)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()
            if current_step % args.log_steps == 0:
                logger.info(f'epoch: {_} total step:{current_step}, loss:{(total_loss-logging_loss)/args.log_steps}')
                logging_loss = total_loss
                # logger.info(f'sentence: {dataset.decode_sentence(xs[2].tolist())}, predicate:{dataset.labels[int(preds[2].topk(1)[1])]}, gold: {dataset.labels[int(ys[2])]}')
            
            current_step += 1
    return total_loss, current_step
    
    
