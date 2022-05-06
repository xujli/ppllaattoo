import yaml
import os

yml_name = 'fedprox_FashionMNIST_mlp.yml'

def modify_random_seed(seed):
    with open(yml_name, encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['data']['random_seed'] = int(seed)
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)

def modify_sampler(sampler):
    with open(yml_name, encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['data']['sampler'] = sampler
        if sampler == 'orthogonal':
            content['trainer']['rounds'] = 80
        else:
            content['trainer']['rounds'] = 50
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)

def modify_concentration(concentration):
    with open(yml_name, encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['data']['concentration'] = concentration
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)


if __name__ == '__main__':
    modify_sampler('noniid')
    for concentration in [0.1, 0.5]:
        modify_concentration(concentration)
        for seed in range(1, 11):
            modify_random_seed(seed)
            os.system('python fedprox.py')
    for sampler in ['orthogonal']:
        modify_sampler(sampler)
        for seed in range(1, 11):
            modify_random_seed(seed)
            os.system('python fedprox.py')