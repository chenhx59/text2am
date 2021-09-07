from plan_generator import generator
from plan_generator.generator import Plan, State, Predicate, Goal
import os
import pickle as pkl
from config.config import args_type


def generate(prediction_file, raw_data_file, dir_out, file_out_prefix='Soln'):
    os.makedirs(dir_out, exist_ok=True)
    with open(prediction_file, 'rb') as pred_f:
        global_idx = 0
        predictions = pkl.load(pred_f)
        with open(raw_data_file, 'rb') as raw_f:
            raw_data = pkl.load(raw_f)
            raw_data.sort(key=lambda x: x['id'])
            for data in raw_data:
                states = []
                plan_id = data['id']
                actions = [None] + data['plan'][:-1] + [None]
                info = {}
                info['plan_id'] = data['id']
                info['actions'] = data['plan'][:-1]
                state_num = len(data['text_trace'])
                init = None
                gold_goal = data['goal_state']
                goal = []
                for state_idx in range(state_num):
                    state_info = {}
                    predicates = []
                    pred_num = len(data['text_trace'][state_idx].split('.')) - 1
                    for pred_idx in range(pred_num):
                        one_prediction = predictions[global_idx]
                        global_idx += 1
                        predicate = Predicate(one_prediction)
                        assert one_prediction.id[0] == info['plan_id']
                        assert one_prediction.id[1] == state_idx
                        assert one_prediction.id[2] == pred_idx
                        if one_prediction.is_goal:
                            goal.append(predicate)
                        predicates.append(predicate)
                    state_info['predicates'] = predicates
                    state_info['plan_id'] = info['plan_id']
                    state_info['state_id'] = state_idx
                    state_info['pre_action'] = actions[state_idx]
                    state_info['post_action'] = actions[state_idx+1]
                    state = State(state_info)
                    states.append(state)
                goal_info = {
                    'predicates': goal, 
                    'plan_id': plan_id
                }
                
                info['state_list'] = states
                info['goal'] = Goal(goal_info)
                plan = Plan(info, args_type)
                plan.output(os.path.join(dir_out, f'{file_out_prefix}{plan_id}'))
                print('done!')



if __name__ == '__main__':
    prediction_file = 'tmp/prediction.pkl'
    raw_data_file = 'data/Newblocks/train.pkl'
    generate(prediction_file, raw_data_file, 'solution')