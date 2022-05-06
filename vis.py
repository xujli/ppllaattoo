import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Times New Roman'

cmap_list = [
    '#0780cf',
    '#765005',
    '#fa6d1d',
    '#0e2c82',
    '#b6b51f',
    '#da1f18',
    '#701866',
    '#f47a75',
    '#009db2',
    '#024b51',
    '#0780cf',
    '#765005',
]

cmap_list2 = [
    '#45c8dc',
    '#854cff',
    '#5f45ff',
    '#47aee3',
    '#d5d6d8',
    '#96d7f9',
    '#f9e264',
    '#f47a75',
    '#009db2',
    '#024b51',
    '#0780cf',
    '#765005'
]

cmap_list3 = [
    '#9489fa',
    '#f06464',
    '#f7af59',
    '#f0da49',
    '#71c16f',
    '#2aaaef',
    '#5690dd',
    '#bd88f5',
    '#009db2',
    '#024b51',
    '30780cf',
    '3765005'
]

def EMA(curve, alpha=0.2):
    state = 0
    smoothed_curve = []
    for item in curve:
        if len(smoothed_curve) == 0:
            state = item
        else:
            state = alpha * item + state * (1 - alpha)
        smoothed_curve.append(state)
    return np.array(smoothed_curve)

def get_acc(dir):
    acc1 = []
    for file in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, file))
        acc = df['accuracy'].values
        if 'orthogonal' not in dir:
            acc = acc[:100]
        acc1.append(acc)

    return np.mean(acc1, axis=0), np.max(acc1, axis=1), np.std(acc1, axis=0)


def vs_prox(data_setting, dataset, net, sampler, target_accs=[0], vis=True):
    label_list = ['FedTrip', 'FedProx', 'FedAvg', 'MOON']
    for i, label in enumerate(label_list):
        acc4, max_acc, std = get_acc(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')
        acc = EMA(acc4)
        plt.plot(acc, c=cmap_list3[i], linewidth=2, alpha=0.8)
        # plt.fill_between(np.arange(0, len(acc4)), acc4-std, acc4+std, alpha=0.5)
        print('{:.3f} {:.3f} {}'.format(np.mean(max_acc), np.std(max_acc),
                                        [np.sum(acc4 <= target_acc) for target_acc in target_accs]))

    if vis:
        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(label_list, fontsize=15)
        plt.xlabel('# Rounds', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)

        plt.tight_layout(pad=0.1)
        plt.savefig('vis/acc_plot/{}/prox_{}_{}_{}.png'.format(data_setting, dataset, net, sampler), dpi=800)
        plt.show()

def vis_acc(data_setting, dataset, net, sampler, target_accs=[0], vis=True):
    label_list = ['FedTripM', 'FedAvgM', 'SlowMo', 'FedDyn', 'FedAGM']
    # label_list = ['FedTripOpt', 'FedCM', 'FedGbo']
    plt.figure(figsize=(9, 6))
    for i, label in enumerate(label_list):
        acc4, max_acc, std = get_acc(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')
        acc = EMA(acc4)
        plt.plot(np.arange(len(acc))+1, acc, c=cmap_list3[i], linewidth=1)
        # plt.fill_between(np.arange(0, len(acc4)), acc4-std, acc4+std, alpha=0.5)
        print('{:.3f} {:.3f} {}'.format(np.mean(max_acc), np.std(max_acc),
                                        [np.sum(acc4 <= target_acc) for target_acc in target_accs]))

    if vis:

        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(label_list, fontsize=15)
        plt.xlabel('# Rounds', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlim(0, 100)
        # plt.ylim(70, 100)
        plt.tight_layout(pad=0.1)
        plt.savefig('vis/acc_plot/{}/{}_{}_{}.pdf'.format(data_setting, dataset, net, sampler), dpi=400)
        plt.show()

def vis_acc2(data_setting, dataset, net, sampler, target_accs=[0], vis=True, legend=True):
    label_list = ['FedTrip', 'FedAvg', 'FedProx']

    plt.figure(figsize=(9, 6))
    for i, label in enumerate(label_list):
        acc4, max_acc, std = get_acc(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')
        acc = EMA(acc4)
        plt.plot(np.arange(len(acc))+1, acc, c=cmap_list3[i], linewidth=2)
        # plt.fill_between(np.arange(0, len(acc4)), acc4-std, acc4+std, alpha=0.5)
        print('{:.3f} {:.3f} {:.3f} {}'.format(np.mean(max_acc), np.std(max_acc), acc4[49],
                                        [np.sum(acc4 <= target_acc) for target_acc in target_accs]))

    if vis:
        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if legend:
            plt.legend(label_list, fontsize=15)
        plt.xlabel('# Rounds', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlim(0, 100)
        # plt.tight_layout(pad=0.1)
        plt.savefig('vis/acc_plot/{}/{}_{}_{}.pdf'.format(data_setting, dataset, net, sampler), dpi=400)
        plt.show()

def get_loss(dir):
    loss1 = []
    for file in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, file))
        loss = df['loss']
        loss1.append(loss)
    return np.mean(loss1, axis=0), np.max(loss1, axis=1), np.std(loss1, axis=0)

def vis_loss(data_setting, dataset, net, sampler, target_acc=0, vis=True):
    label_list = ['FedAvg', 'FedAdp']# ['FedAM', 'Local Momentum', 'Server Momentum', 'FedGbo', 'FedSign_ad']

    plt.figure()
    for label in label_list:
        loss4, max_acc, std = get_loss(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')

        plt.plot(loss4)
        # plt.fill_between(np.arange(0, len(loss4)), loss4-std, loss4+std, alpha=0.5)
        print(np.mean(max_acc), np.std(max_acc))

    if vis:
        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(label_list, fontsize=15)

        plt.tight_layout(par=0.1)
        plt.savefig('vis/loss_plot/{}_{}_{}.pdf'.format(dataset, net, sampler))
        plt.show()
    plt.close()

def boxplot(data_setting, dataset, net, sampler, target_acc=0):
    label_list = ['FedTripOpt', 'Local Momentum', 'Server Momentum', 'FedProx Momentum', 'FedAvg', 'FedGbo']
    plt.figure()
    accs = []
    for idx, label in enumerate(label_list):
        acc1 = []
        for file in os.listdir(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}'):
            df = pd.read_csv(os.path.join(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}', file))
            acc = df['accuracy']
            acc1.append(np.max(acc))
        accs.append(acc1)

    box = plt.boxplot(accs, notch=True, patch_artist=True,
                      medianprops={'color': 'white'})

    for patch, color in zip(box['boxes'], cmap_list3):
        patch.set(color=color, alpha=0.8)

    plt.xticks(np.arange(len(label_list)) + 1, label_list, rotation=-13)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout(pad=0.1)
    plt.savefig('vis/boxplot/{}/{}_{}_{}.pdf'.format(data_setting, dataset, net, sampler), dpi=400)
    plt.show()

def tranverse(data_setting):
    nets = ['mlp', 'lenet']
    datasets = ['MNIST', 'FashionMNIST']
    samplers = ['noniid_0.1', 'noniid_0.5', 'orthogonal']
    for net in nets:
        for dataset in datasets:
            for sampler in samplers:
                sample(data_setting, dataset, net, sampler, target=75)

def sample(data_setting, dataset, net, sampler, target):
    vis_acc(data_setting, dataset, net, sampler, target)
    boxplot(data_setting, dataset, net, sampler, target)

def get_acc1(dir):
    acc1 = []
    times = []
    for file in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, file))
        acc = df['accuracy'].values
        time1 = df['round_time'].values
        acc1.append(acc)
        times.append(time1)

    return np.mean(acc1, axis=0), np.max(acc1, axis=1), np.std(acc1, axis=0), np.mean(times, axis=0)

def time_acc(data_setting, dataset, net, sampler=0, target_acc=0):
    label_list = ['FedTripOpt', 'MimeLite']
    marker = ['-', '--', ':']
    for i, label in enumerate(label_list):
        for j, sampler in enumerate(['noniid_0.1',  'noniid_0.5', 'orthogonal']):
            acc4, max_acc, std, time1 = get_acc1(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')
            acc = EMA(acc4)
            plt.plot(np.cumsum(time1), acc, linestyle=marker[j], c=cmap_list3[i], linewidth=2, alpha=0.8, label='{}_{}'.format(label_list[i], sampler))
            # plt.fill_between(np.arange(0, len(acc4)), acc4-std, acc4+std, alpha=0.5)
            print('{:.3f} {:.3f} {}'.format(np.mean(max_acc), np.std(max_acc), np.sum(acc4 <= target_acc)))

    plt.grid(axis='y', linestyle='--', linewidth=1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.xlabel('Seconds', fontsize=18)
    plt.ylabel('Test Accuracy', fontsize=18)

    plt.tight_layout(pad=0.1)
    plt.savefig('vis/time_acc/{}_{}.pdf'.format(dataset, net), dpi=400)
    plt.show()

def vis_test(data_setting, dataset, net, sampler, target_accs=[0], vis=True):
    label_list = ['cpu', 'gpu']
    for i, label in enumerate(label_list):
        acc4, max_acc, std = get_acc(f'results/{data_setting}/{dataset}/{net}/{sampler}/{label}')
        acc = EMA(acc4)
        plt.plot(acc, c=cmap_list3[i], linewidth=2, alpha=0.8)
        # plt.fill_between(np.arange(0, len(acc4)), acc4-std, acc4+std, alpha=0.5)
        print('{:.3f} {:.3f} {}'.format(np.mean(max_acc), np.std(max_acc),
                                        [np.sum(acc4 <= target_acc) for target_acc in target_accs]))

    if vis:
        plt.grid(axis='y', linestyle='--', linewidth=1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(label_list, fontsize=15)
        plt.xlabel('# Rounds', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)

        plt.tight_layout(pad=0.1)
        plt.savefig('vis/acc_plot/{}/{}_{}_{}.pdf'.format(data_setting, dataset, net, sampler), dpi=400)
        plt.show()

# print(acc1.max(), acc2.max())
if __name__ == '__main__':
    sampler = ['noniid_0.1', 'noniid_0.5', 'orthogonal']
    # vis_test('10_4', 'MNIST', 'test', 'test', )
    # vis_acc('10_4', 'MNIST', 'lenet', sampler[2], [88, 93])
    # print()
    # vis_acc('10_4', 'FashionMNIST', 'lenet', sampler[1], [88, 93])
    # tranverse('10_4')
    vs_prox('10_4', 'FashionMNIST', 'mlp', sampler[1], )
    # time_acc('10_4', 'MNIST', 'mlp', 'noniid_0.1', 80)