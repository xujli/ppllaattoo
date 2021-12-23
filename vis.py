import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def get_acc(dir):
    acc1 = []
    for file in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, file))
        acc = df['accuracy']
        acc1.append(acc)
    return np.mean(acc1, axis=0)

def vis(dataset, net, sampler, target_acc=0, vis=True):
    label_list = ['FedAM', 'FedProx', 'FedAvg', 'Local Momentum', 'momentum', 'Server Momentum']
    for label in label_list:
        acc4 = get_acc(f'results/10_4/{dataset}/{net}/{sampler}/{label}')

        plt.plot(acc4)
        print(np.sum(acc4 < target_acc))
        print((np.sum(np.abs(acc4[1:] - acc4[:-1])) - (np.max(acc4) - np.min(acc4))) /
              (np.max(acc4) - np.min(acc4)))

    if vis:
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(label_list, fontsize=15)

        # plt.savefig('vis/{}_{}_{}.png'.format(dataset, net, sampler), dpi=800)
        plt.show()

def boxplot(dataset, net, sampler, target_acc=0):
    label_list = ['FedAM', 'FedProx', 'FedAvg', 'Local Momentum', 'Server Momentum']
    plt.figure()
    plt.tight_layout()
    accs = []
    for label in label_list:
        acc1 = []
        for file in os.listdir(f'results/10_4/{dataset}/{net}/{sampler}/{label}'):
            df = pd.read_csv(os.path.join(f'results/10_4/{dataset}/{net}/{sampler}/{label}', file))
            acc = df['accuracy']
            acc1.append(np.sum(acc < target_acc))
        accs.append(acc1 / np.mean(acc1))
    plt.boxplot(accs, labels=label_list)
    plt.show()

# print(acc1.max(), acc2.max())
if __name__ == '__main__':
    vis('MNIST', 'mlp', 'orthogonal', 70)