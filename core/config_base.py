import yaml
from collections import OrderedDict

base = OrderedDict()

# ================================
# Model
base_model = {'name': '',
              'weights': '',
              'params': OrderedDict()}
base['model'] = base_model

# ================================
# Functions

def merge_config(config, base_config=base, pprint=True, unknowns=[]):
    '''
    Merge the config file with the base config. Additionally modifies
        the merged config with the "unknowns" variables for easy experimentation.
        (e.g. modify only one variable without need to create n config files)
        unkowns format in parser:
                --var1 v1 type1
                --var2.d1 v21,v22 type2
            --var1 modifies config['var1'] = type1(v11)
            --var2 modifies config['var2']['d1'] = [type2(v21), type2(v22)]
            Note that for boolean types, false is '' while anything else is True (e.g. bool('False')) 
    '''
    for k, v in base_config.items():
        
        if k not in config.keys():
            config[k] = v
        else:
            if isinstance(config[k], dict):
                merge_config(config[k], v, False)

    if unknowns != []:

        adict = []
        for k, v, t in zip(unknowns[:-1:3], unknowns[1::3], unknowns[2::3]):
            # process keys
            k = k[2:]  # removes the '--'
            k = k.split('.')
            keys = "['" + "']['".join(k) + "']"

            # process values
            v = v.split(',')
            if len(v) == 1:
                v = f"{t}('{v[0]}')"
            else:
                if len(v) == 2 and v[1] == '':
                    v = '[' + f"{t}('{v[0]}')" + ']'
                else:
                    v = '[' + ', '.join([f"{t}('{vv}')" for vv in v]) + ']'

            # save all to the config
            string = 'config' + keys + ' = ' + v
            print('New values:', string)
            exec(string)

    if pprint:
        print_config(config)

    return config


def print_config(config):
    print('---------- PARAMETERS ----------')
    print(yaml.dump(config))
    print('-------- END PARAMETERS --------')
