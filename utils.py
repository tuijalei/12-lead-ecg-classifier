import json
import yaml

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)    
    
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def load_yaml(yaml_file):
    ''' Loading the configurations from a yaml file'''  

    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)          
    
    return dict2obj(args)