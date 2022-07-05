import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'Times New Roman'
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


concentration = 0.5
sampler = 'iid'
cliend_id = 2
patterns = [['iid', 0.5], ['noniid', 0.1], ['noniid', 0.5], ['orthogonal', 0.5]]
intervals = [-0.3, -0.1, 0.1, 0.3]
length = 10
dir = 'MNIST'


def bar_plot():
    plt.figure()
    for idx, (c, pattern) in enumerate(zip(cmap_list3, patterns)):
        array = np.load('{}/label_{}_{}_{}.npy'.format(dir, cliend_id, *pattern), allow_pickle=True)
        dic = {}
        for i in range(length):
            dic[i] = np.sum(array == i)
        plt.bar(np.array(list(dic.keys())) + intervals[idx], dic.values(), alpha=0.8, color=c, width=0.2)

    plt.grid(axis='y', linestyle='--', linewidth=1)
    plt.xlim(-0.5, length - 0.5)
    plt.xticks(np.arange(0, length), fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Class', fontsize=18)
    plt.ylabel('# Samples', fontsize=18)
    # plt.legend([r'IID', r'Non-IID, $\alpha$=0.1', r'Non-IID, $\alpha$=0.5', r'Orthogonal'], fontsize=18)
    plt.tight_layout(pad=0.1)
    plt.savefig('data distribution {}.pdf'.format(dir))
    plt.show()

def heatmap_plot():
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.tight_layout()
    for ax, pattern in zip(axes.flat, patterns):
        dics = []
        for idx, id in enumerate(range(1, 11)):
            array = np.load('{}/label_{}_{}_{}.npy'.format(dir, id, *pattern), allow_pickle=True)
            dic = []

            for i in range(length):
                dic.append(np.sum(array == i))

            dics.append(dic)
        dics = np.array(dics)

        im = ax.imshow(dics, cmap='Purples')
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10), ['Client {}'.format(i) for i in range(1, 11)], rotation=30)
        ax.set_title('{}_{}'.format(pattern[0].upper(), pattern[1]) if pattern[0] != 'iid' else '{}'.format(pattern[0].upper()))
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig('./1.pdf', pad_inches=0)

if __name__ == '__main__':
    heatmap_plot()