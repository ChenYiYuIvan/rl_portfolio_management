import json
import yaml
from types import SimpleNamespace


def read_json_config(file_name):
    f = open(f'config/{file_name}.json')
    return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def read_yaml_config(file_name):
    class Struct:
        def __init__(self, **entries): 
            self.__dict__.update(entries)

    f = open(f'config/{file_name}.yaml')
    dic = yaml.safe_load(f)

    return Struct(**dic)


def get_checkpoint_folder(agent, env):
    folder = './checkpoints/'
    if agent.pre:
        folder += 'pre_'
    folder += f'{agent.name}_{agent.network_type}_{agent.reward_type}_stocks{env.observation_space.shape[0]}_batch{agent.batch_size}_window{env.window_length}'
    if agent.preprocess == 'divide_close':
        folder += '_close'
    if hasattr(agent, 'reward_scale'):
        scale = str(agent.reward_scale)
        scale = scale.replace('.', '_')
        folder += f'_scale{agent.reward_scale}'
    if agent.active_il:
        folder += '_ail'

    return folder
    