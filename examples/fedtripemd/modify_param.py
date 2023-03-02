import yaml
import os

yml_name = 'fedtrip_MNIST_mlp.yml'

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
            content['trainer']['rounds'] = 100
        else:
            content['trainer']['rounds'] = 100
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)

def modify_concentration(concentration):
    with open(yml_name, encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['data']['concentration'] = concentration
        content['trainer']['rounds'] = 100
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)

def modify_mu(alpha):
    with open(yml_name, encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['trainer']['alpha'] = alpha
        content['trainer']['rounds'] = 100
    with open(yml_name, 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)


if __name__ == '__main__':
    for alpha in [0.1]:
        modify_concentration(alpha)
        for seed in range(1, 11):
            modify_random_seed(seed)
            os.system('python fedtrip.py')

