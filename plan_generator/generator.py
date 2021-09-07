from model.models import ModelOutputForPlan
from typing import Dict, List
import os
import re
import copy


def check_AC(am):
            '''
            check action constraint
            '''
            add_and_pre = am.add_list & am.precondition
            if not add_and_pre.is_empty():
                print(f'{am.name} action constraint: add list in precondition {add_and_pre}')
            del_sub_pre = am.del_list - am.precondition
            if not del_sub_pre.is_empty():
                print(f'action constraint: del list not in precondition {del_sub_pre}')
            if add_and_pre.is_empty() and del_sub_pre.is_empty():
                print(f'{am.name} action constraint fulfill.')

def check_PC(am):
    '''
    check plan constraint
    '''
    pass

class ActionModel():
    '''
    action schema
    '''
    def __init__(self, data: dict) -> None:
        self.name = data.get('name')
        self.parameters = data.get('parameters')
        self.precondition = data.get('precondition')
        self.del_list = data.get('del_list')
        self.add_list = data.get('add_list')



    def regular(self, key='ad'):
        '''
        修改action model使其符合action constraint
        '''
        n_s = copy.deepcopy(self)
        if key == 'p':# 修改precondition
            # 处理矛盾，即add list 与del list的交集
            # TODO
            # 删除与add list的交集
            n_s.precondition = self.precondition - (self.precondition & self.add_list)
            # 添加del list 与precondition的差集
            n_s.precondition = self.precondition | (self.del_list - self.precondition)
        elif key == 'ad' or key == 'da':
            # add list去除与precondition的交集
            n_s.add_list = self.add_list - (self.add_list & self.precondition)
            # del list去除与precondition的差集
            n_s.del_list = self.del_list - (self.del_list - self.precondition)

        return n_s
    @classmethod
    def parse_param(cls, stream):
        '''
        (?a -TYPE0 ?b -TYPE1 )
        '''
        params = {}
        pattern = re.compile(r'\?[a-zA-Z\s]+\-[a-zA-Z0-9]+\s')
        al = pattern.findall(stream)
        for item in al:
            item = item.replace('-', '')
            item = item.split(' ')
            params[item[0]] = item[1].lower()
        return params


    @classmethod
    def parse_precondition(cls, stream):
        pattern = re.compile(r'\([A-Z0-9]+[\s\?a-z]+\)')
        al = pattern.findall(stream)
        precondition = State.parse('\n'.join(al)+'\n')
        return precondition
        

    @classmethod
    def parse_effect(cls, stream):
        del_pattern = re.compile(r'not\([A-Z0-9]+[\s\?a-z]+\)')
        add_pattern = re.compile(r'\([A-Z0-9]+[\s\?a-z]+\)')
        del_al = del_pattern.findall(stream)
        del_al = [i.replace('not', '') for i in del_al]
        add_al = add_pattern.findall(stream)
        del_list = State.parse('\n'.join(del_al)+'\n')
        add_list = State.parse('\n'.join(add_al)+'\n')
        add_list = add_list - del_list
        return del_list, add_list

    @classmethod
    def parse(cls, stream):
        data = {}
        stream = stream.split('\n')
        name = ''
        param_stream = ''
        pre_stream = ''
        eff_stream = ''
        for item in stream:
            if ':action' in item:
                name = item.split(' ')[1]
            elif ':parameters' in item:
                param_stream = ' '.join(item.split(' ')[1:])
            elif ':precondition' in item:
                pre_stream = ' '.join(item.split(' ')[1:])
            elif ':effect' in item:
                eff_stream = ' '.join(item.split(' ')[1:])
        data['name'] = name
        data['parameters'] = cls.parse_param(param_stream)
        data['precondition'] = cls.parse_precondition(pre_stream)
        data['del_list'], data['add_list'] = cls.parse_effect(eff_stream)
        return cls(data)

class Action(ActionModel):
    
    @classmethod
    def instantiate_from(cls, am: ActionModel, params_map):
        data = {'parameters': {}, 'name': am.name}
        for p, t in am.parameters.items():
            data['parameters'][params_map[p]] = t
        
        # precondition
        data['precondition'] = am.precondition.params_change(params_map)
        data['del_list'] = am.del_list.params_change(params_map)
        data['add_list'] = am.add_list.params_change(params_map)
        return cls(data)
        # del_list

        # add_list


class Predicate():
    '''
    由命题名称和参数组成
    '''
    def __init__(self, input_data: ModelOutputForPlan) -> None:
        self.params = input_data.extract_params
        self.name = input_data.predicate
    
    def params_change(self, params_map):
        '''
        改变该谓词的参数(self.params)，在将action model实例化时用到
        :param params_map: 原参数到目标参数的映射，如{'?a': 'Block1'}
        '''
        n_p = copy.deepcopy(self)
        for idx, p in enumerate(n_p.params):
            n_p.params[idx] = params_map[p]
        return n_p

    def __eq__(self, o: object) -> bool:
        if not len(self.params) == len(o.params):
            return False
        if not o.name == self.name:
            return False
        for i, j in zip(self.params, o.params):
            if not i == j:
                return False
        
        return True

    def __str__(self) -> str:
        return f'({self.name} {self.params})'

    def __repr__(self) -> str:
        return f'({self.name} {self.params})'
            

    @classmethod
    def parse(cls, stream: str):
        '''
        stream looks like '(predicate param1 param2 ...)'
        '''
        stream = stream.replace('(', '')
        stream = stream.replace(')', '')
        
        return cls(ModelOutputForPlan({
            'extract_params': stream.split(' ')[1:],
            'predicate': stream.split(' ')[0],
            'id': None
        }))


class State():
    '''
    由多个命题组成
    '''
    def __init__(self, info: dict) -> None:
        self.content = info['predicates'] # 命题列表
        self.plan_id = info['plan_id']
        self.state_id = info.get('state_id')
        self.is_goal = False
        self.precondition_of = None if self.is_goal else info.get('post_action')
        self.postcondition_of = None if self.is_goal else info.get('pre_action')
        self.is_init = False
        
    def __getitem__(self, idx):
        return self.content[idx]

    def __setitem__(self, idx, value):
        if not isinstance(value, Predicate):
            raise ValueError(f'{value} is not a Predicate.')
        self.content[idx] = value

    def __len__(self):
        return len(self.content)

    def __call__(self, am: Action):
        '''
        将action作用到state上，返回下一个state
        有可能state缺少am.del_list的内容，
        有可能state存在am.add_list的内容
        :return {
            'next_state': next state,
            'to_be_add': 原状态需要增加的predicate列表（这些predicate在del_list出现
            但是没有在state中出现,
            'to_be_del': 原状态需要删除的predicate列表（这些predicate在add_list出现
            但是也在state中出现
        }
        
        '''
        
            
        n_s = copy.deepcopy(self)
        ret = {'next_state': self, 'to_be_add': [], 'to_be_del': []}
        
        '''
        action constraint 不满足，改del list\\add list还是改precondition
        plan constraint不满足（这个应该普遍存在），如何处理
        '''
        ret['to_be_add'] = am.precondition - n_s
        ret['to_be_del'] = am.add_list & n_s
        n_s = n_s | am.add_list
        n_s = n_s - am.del_list
        ret['next_state'] = n_s

                
        return ret

    def __and__(self, s):
        n_s = copy.deepcopy(self)
        n_s.content = []
        for pred in self:
            for comp_pred in s:
                if pred == comp_pred:
                    n_s.content.append(pred)
                    break
        return n_s

    def __or__(self, s):
        n_s = copy.deepcopy(self)
        d = {}
        for item in n_s.content:
            key = ' '.join([item.name]+item.params)
            d[key] = item
        for item in s.content:
            key = ' '.join([item.name]+item.params)
            d[key] = item
        n_s.content = list(d.values())
        return n_s
    
    def __sub__(self, s):
        n_s = copy.deepcopy(self)
        for pred in self:
            for comp_pred in s:
                if pred == comp_pred: # pred in s
                    n_s.content.remove(pred)
                    break
        return n_s

    def __repr__(self) -> str:
        if self.is_empty():
            return 'empty state'
        return ','.join([i.__repr__() for i in self.content])

    def is_empty(self):
        if len(self) == 0:
            return True
        return False

    def params_change(self, params_map):
        '''
        返回参数改变后的State，将action model实例化时用到
        :param params_map 形如{'?a': 'Block1'}
        '''
        n_s = copy.deepcopy(self)
        n_s.content = [pred.params_change(params_map) for pred in self.content]
        return n_s



    @classmethod
    def parse(cls, stream, plan_id=-1):
        '''
        stream looks like '()\n()\n...()\n'
        '''
        info = {
            'predicates': [],
            'plan_id': plan_id
        }
        
        for p_stream in stream.split('\n')[:-1]:
            info['predicates'].append(Predicate.parse(p_stream))

        return cls(info)

class Goal(State):
    def __init__(self, info: dict):
        self.content = info['predicates']
        self.plan_id = info['plan_id']
        self.is_goal = True

    def __len__(self):
        return len(self.content)

    


class Plan():
    def __init__(self, info: dict, args_type: dict={}) -> None:
        self.states = info['state_list']
        self.plan_id = info['plan_id']
        self.actions = info['actions']
        self.init = None
        self.goal = info['goal']
        self.args_type = args_type

    def __getitem__(self, idx):
        return self.states[idx]

    def __setitem__(self, idx, value):
        if not isinstance(value, State):
            raise ValueError(f'{value} is not type State.')
        self.states[idx] = value

    def __len__(self):
        return len(self.states)


    def output(self, path):
        '''
        将一个Plan对象写入到目标文件中。
        '''

        def _write_state(handler, state: State, prefix='observations'):
            handler.write(f'(:{prefix}\n')
            for predicate in state.content:
                handler.write(f'({predicate.name} ')
                handler.write(' '.join(predicate.params))
                handler.write(')\n')
            handler.write(')\n')


        dir, file_name = os.path.split(path)
        os.makedirs(dir, exist_ok=True)

        with open(path, 'w') as f:
            f.write('(solution\n')
            f.write('(:objects')
            for k, v in self.args_type.items():
                f.write(f' {k} - {v}')
            f.write(')\n') #object end
            _write_state(f, self.states[0], 'init')
            f.write('\n')
            for action, state in zip(self.actions[:], self.states[:]):
                _write_state(f, state)
                f.write(action)
                f.write('\n')
            _write_state(f, self.states[-1], 'goal')
            # _write_state(f, self.states[-1])
            # f.write('\n')
            # _write_state(f, self.goal, 'goal')
            

            f.write(')') # solution end
        
    @classmethod
    def parse(cls, soln_file):

        '''
        解析一个plan文件（一般在solution文件夹中）
        '''
        # read file
        plan_id = re.match(r'([a-zA-Z/]+)([0-9]+)', soln_file).groups()[1]
        state_stream = []
        action_stream = []
        current_process = ''
        state = ''
        args_type_steam = ''
        with open(soln_file, 'r') as f:
            while True:
                stream = f.readline()
                if not stream:
                    break
                if ':objects' in stream:
                    stream = stream.replace('(:objects ', '')
                    stream = stream.replace(')', '')
                    args_type_steam = stream
                if ':observations' in stream:
                    current_process = ':observations'
                    break
            while True:
                stream = f.readline()
                if not stream:
                    break
                
                if stream == '\n':
                    continue

                if current_process == 'action':
                    if ':observations' in stream or ':goal' in stream:
                        current_process = ':observations'
                        state = ''
                        continue
                    else:
                        action_stream.append(stream)
                elif current_process == ':observations':
                    if stream == ')\n':
                        current_process = 'action'
                        state_stream.append(state)
                        continue
                    else:
                        state += stream
        info = {}
        info['actions'] = action_stream[:-1]
        info['plan_id'] = plan_id
        info['state_list'] = []
        args_type = {}
        a_t_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9]+\s\-\s[a-zA-Z][a-zA-Z0-9]+')
        args_type_group = a_t_pattern.findall(args_type_steam)
        for a_t in args_type_group:
            args_type[a_t.split(' - ')[0]] = a_t.split(' - ')[1]
        # print(args_type_group, args_type)
        # state
        for s_stream in state_stream:
            info['state_list'].append(State.parse(s_stream, plan_id))

        info['goal'] = info['state_list'][-1]

        return cls(info, args_type)

def parse_AM(res_file, regular=False) -> Dict:
    '''
    解析Result.out文件，得到action model
    :return: {am_name: am}，am_name为小写
    '''
    ams = {}
    '''
    parse action model
    '''
    with open(res_file, 'r') as f:
        action_stream = ''
        while True:
            stream = f.readline()
            if ':action' in stream:
                action_stream += stream
                break
        while True:
            stream = f.readline()
            if not stream:
                break
            if ':action' in stream:
                action_stream = ''
                action_stream += stream
            elif stream == '\n':
                am = ActionModel.parse(action_stream)
                if regular:
                    am = am.regular()
                ams[am.name.lower()] = am
                action_stream = ''
            else:
                action_stream += stream
    return ams

         
def parse_result(res_file, soln_dir, soln_num=10):
    '''
    解析HTNML生成的Result.out文件得到action model，将action model作用于Soln文件生成
    新的plan
    :param res_file: result.out文件
    :param soln_dir: 存放soln文件的路径，该路径下的soln文件名形如 Soln0，数字后缀由0
    递增到soln_num-1
    :param soln_num: soln文件的个数
    '''
    ams = parse_AM(res_file)
    for file in [os.path.join(soln_dir, f'Soln{i}') for i in range(soln_num)]:
        plan = Plan.parse(file)
        '''
        plan的第一个动作根据action model需要满足一个前置条件，该前置条件有可能与
        plan的初始状态相异：
            若前置条件存在初始状态所没有的命题，那么在新生成的plan中，在plan的初
            始状态增加该命题；
        然后根据该初始状态和action model，获得整条新的plan
        '''
        init_flag = 1
        for idx, action_stream in enumerate(plan.actions):
            action_stream = action_stream.replace('\n', '')
            action_stream = action_stream.replace(')', '')
            action_stream = action_stream.replace('(', '')
            action_name = action_stream.split(' ')[0].lower()
            am = ams[action_name]
            params = action_stream.split(' ')[1:]
            params_map = {}
            for k, p in zip(am.parameters.keys(), params):
                params_map[k] = p
            am = Action.instantiate_from(ams[action_name], params_map)
            '''
            am的precondition与plan.states[0]并集
            '''
            if init_flag:
                # 创建init state
                plan[0] = plan[0] | am.precondition
                init_flag = 0
            '''
            更新
            由于动作模型是不正确的，所以由初始状态和动作生成整条plan的时候，
            action与其前面的状态会产生矛盾，因而action需要要求状态进行增删，
            显而易见，最后增删的内容都要作用到初始状态上
            但是可能出现第i+1个动作要求删除命题p，但是p又存在于第i个动作的
            add_list中，那么如何处理呢？TODO
            '''
            '''
            由plan[idx]与action得到next_state、to_be_add和to_be_del
            '''
            ret = plan[idx](am)
            next_state = ret['next_state']
            to_be_del = ret['to_be_del']
            to_be_add = ret['to_be_add']
            plan[0] = plan[0] | to_be_add
            plan[0] = plan[0] - to_be_del
            plan[idx+1] = next_state


    

    pass




