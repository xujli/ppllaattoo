import yaml
import os

yml_name = 'momentum_FashionMNIST_mlp.yml'

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
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)


if __name__ == '__main__':
    for sampler in ['iid', 'noniid_fedavg', 'orthogonal_fedavg']:
        modify_sampler(sampler)
        for seed in range(0, 10):
            modify_random_seed(seed)
            os.system('python momentum.py')