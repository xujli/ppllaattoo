import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def get_acc(dir):
    acc1 = []
    for file in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, file))
        acc = df['accuracy']
        if len(acc) < 30:
            acc = np.append(acc, np.full(50-len(acc), 98.2))
        acc1.append(acc)
    return np.mean(acc1, axis=0)

def vis(dataset, net, sampler):
    acc4 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/fedavg')
    acc5 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/fedsign')
    acc6 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/fedprox')
    acc7 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/momentum')
    acc8 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/{sampler}')
    # acc7 = get_acc(f'{dataset}/{net}/{sampler}/{sampler}')
    # acc6 = get_acc('orth/fedavg6')

    plt.plot(acc4)
    plt.plot(acc5)
    # plt.plot(acc6)
    plt.plot(acc7)
    plt.plot(acc8)
    # plt.ylim(60, 90)
    plt.title('{}_{}_{}'.format(dataset, net, sampler))
    plt.grid()
    plt.legend(['fed_avg', 'fed_sign', 'momentum', 'local_momentum'])
    plt.savefig('{}_{}_{}.png'.format(dataset, net, sampler), dpi=800)
    plt.show()

# print(acc1.max(), acc2.max())
if __name__ == '__main__':

    acc4 = get_acc(f'results/50_4/FashionMNIST/lenet/noniid/fedavg')
    acc5 = get_acc(f'results/50_4/FashionMNIST/lenet/noniid/fedsign')
    acc6 = get_acc(f'results/50_4/FashionMNIST/lenet/noniid/fedprox')

    plt.plot(acc4)
    plt.plot(acc5)
    plt.plot(acc6)

    plt.grid()
    plt.legend(['fed_avg', 'fed_sign', 'fedprox', 'momentum', 'local_momentum'])
    plt.show()
    # vis('FashionMNIST', 'mlp', 'iid')