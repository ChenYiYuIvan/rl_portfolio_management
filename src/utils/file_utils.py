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


def get_checkpoint_folder(agent_name, env_name):
    folder = f'./checkpoints/{agent_name}_{env_name}'
    return folder


class Dict2Class(object):
    # Turns a dictionary into a class
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])