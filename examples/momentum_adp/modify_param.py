import yaml
import os

def modify_random_seed(seed):
    with open('./momentum_MNIST_lenet5.yml', encoding='UTF-8') as fp:
        content = yaml.load(fp, Loader=yaml.FullLoader)
        content['data']['random_seed'] = int(seed)
    with open('./momentum_MNIST_lenet5.yml', 'w', encoding='UTF-8') as fp:
        yaml.dump(content, fp)


if __name__ == '__main__':
    for seed in range(0, 10):
        modify_random_seed(seed)
        os.system('python momentum.py')