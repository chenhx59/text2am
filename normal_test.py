
import os

from nltk.sem.logic import ENTITY_TYPE
# from train import collate_fn
from plan_generator.generator import Plan, ActionModel, parse_result
from config.config import args_type
from model.dataset.data_reader import SimpleData, StateData
from model.models import StateModel
#from run_trainer import args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence


def compare(file1, file2):
    with open(file1, 'r') as f1:
        with open(file2, 'r') as f2:
            str1 = f1.read()
            str2 = f2.read()
            if str1 == str2:
                return True
            # print(file1, file2)
            return False

def test():
    path = 'solution'
    for file in [os.path.join(path, i) for i in os.listdir(path)]:
        plan = Plan.parse(file)
        plan.output(f'{file}_')
        if not compare(file, file+'_'):
            print(file, file+'_')
def test_action_parse():
    inp = '''(:action PUT-DOWN
:parameters (?a -TYPE0 ?b -TYPE1 )
:precondition (and (THERE ?b ?a) (THERE ?a ?b) (OF ?b ?a) (READY ?b ?a))
:effect (and (not(THERE ?b ?a)) (not(THERE ?a ?b)) (not(OF ?b ?a)) (not(READY ?b ?a)) (THERE ?a) (ROBOT ?b) (THERE ?b) (BLOCK1 ?a) (BLOCK5 ?b) (NONE ?a))
)'''
    
    am = ActionModel.parse(inp)
    pass

def test_res_parse():
    res_file = 'HTNML/am1/Result.out'
    soln_dir = 'HTNML/am1'
    parse_result(res_file, soln_dir)
    pass

# test_res_parse()
# soln_file = 'solution/Soln11'
# plan = Plan.parse(soln_file)
# plan.output('solution/Soln0')
def collate_fn(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    label2word = {
        0: 13,# ready
        1: 3, # table
        2: 2, # on
        3: 40, # holding
        4: 16, #'nothing'
    }
    y = [[torch.tensor(label2word[j.item()]) for j in i] for i in y]
    others = [i[2] for i in data]
    gold_params = [i['gold_params_logits'] for i in others]
    extract_params = [i['extract_params_logits'] for i in others]

    return x, y, gold_params, extract_params

def test_map_list_speed(i):
    import time
    l = list(range(1000))
    t = time.time()
    print(f'map start: {t}')
    for _ in range(i):
        l_map = list(map(lambda x: x, l))
    end_t = time.time()
    print(f'map end: {end_t}')
    print(f'map duration: {end_t-t}')

    t = time.time()
    print(f'list start: {t}')
    for _ in range(i):
        l_map = [i for i in l]
    end_t = time.time()
    print(f'list end: {end_t}')
    print(f'list duration: {end_t-t}')

def foo(*w, **kw):

    pass

foo(1, 2, k=1, l=2)
pass