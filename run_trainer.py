from trainer import predict, train_state_model
from model.models import MappingModel, StateModel
from model.dataset.data_reader import SimpleData, StateData
import torch
from config.config import model_config, training_config, args_type
import argparse
parser = argparse.ArgumentParser()
for k, v in model_config.items():
    parser.add_argument('--'+k, default=v, required=False)
for k, v in training_config.items():
    parser.add_argument('--'+k, default=v, required=False)
args = parser.parse_args()
dataset = StateData('data/Newblocks/train.pkl', args_type=args_type)
args.vocab_size = dataset.vocab_size
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.args_type = args_type
args.dataset = dataset


if __name__ == '__main__':
    model = StateModel(args)
    model.to(args.device)
    ret = train_state_model(args, model, dataset)

    # predict(args, model, dataset)