import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['font.sans-serif'] = 'Times New Roman'
cmap_list = [
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


def plot(filename):
    with open(filename, 'r') as file:
        divergence_dic = {}
        name = ''
        for line in file.readlines():
            if len(line.split(' ')) < 2:
                name = line.strip()
                divergence_dic[name] = {}
            else:
                divergence_dic[name][line.split(' ')[0]] = float(line.split(' ')[1]) / 16
    shift_list = [-0.3, -0.1, 0.1, 0.3]
    plt.figure()
    for i, (name, item) in enumerate(divergence_dic.items()):
        plt.bar(np.arange(len(item.values())) - shift_list[i],
                item.values(), alpha=0.8, width=0.2, color=cmap_list[i], label=name)

    plt.xlabel('Layers', fontsize=18)
    plt.ylabel('Divergence', fontsize=18)
    plt.xticks(np.arange(len(divergence_dic['IID'].keys())),
               divergence_dic['IID'].keys(), rotation=-25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(fontsize=20)
    plt.tight_layout(pad=0.1)
    plt.savefig('Divergence {}-{}.pdf'.format(filename.split('_')[0].upper(),
                                              filename.split('_')[1].upper()))
    plt.show(dpi=400)

if __name__ == '__main__':
    plot('mnist_lenet')