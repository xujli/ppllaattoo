import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


dir = 'results/10_4/MNIST/lenet/fedavg/noniid/'

plt.figure(figsize=(9, 6))
max_acc, max_std = [], []
mean_acc, mean_std = [], []
for subdir in os.listdir(dir):
    if subdir.endswith('0.5'):
        continue
    acc1 = []
    for file in os.listdir(os.path.join(dir, subdir)):
        df = pd.read_csv(os.path.join(dir, subdir, file))
        acc = df['accuracy'].values
        acc1.append(acc)
    try:
        acc1 = np.array(acc1)
        max_acc.append(np.mean(np.max(acc1, axis=1)))
        max_std.append(np.std(np.max(acc1, axis=1)))
        indices = np.where(acc1 >= 90)
        accs = [indices[1][0]]
        for index in range(1, len(indices[0])):
            if indices[0][index] > indices[0][index - 1]:
                accs.append(indices[1][index])
        mean_acc.append(np.mean(accs))
        mean_std.append(np.std(accs))
    except:
        Exception()

max_acc = np.array(max_acc)
max_std = np.array(max_std) / 2
mean_acc = np.array(mean_acc)
mean_std = np.array(mean_std) / 2
print(max_acc)
plt.plot(np.arange(len(max_acc)) / 10 + 0.1, max_acc, color=cmap_list3[0], label='Test Accuracy')
plt.fill_between(np.arange(len(max_acc)) / 10 + 0.1, max_acc-max_std,
                 max_acc+max_std, color=cmap_list3[1], alpha=0.4)
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', linewidth=1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'Value of $\alpha$', fontsize=18)
plt.ylabel('Test Accuracy(%)', fontsize=18)
plt.xlim(0.1, 2.5)
plt.twinx()
plt.plot(np.arange(len(mean_acc)) / 10 + 0.1, mean_acc, color=cmap_list3[2], label='# of Rounds')
plt.fill_between(np.arange(len(mean_acc)) / 10 + 0.1, mean_acc-mean_std,
                 mean_acc+mean_std, color=cmap_list3[3], alpha=0.4)
plt.yticks(fontsize=18)
plt.ylim(20, 75)
plt.ylabel('# of Round that Test Accuracy reaches 90%', fontsize=18)
# plt.legend([r'IID', r'Non-IID, $\alpha$=0.1', r'Non-IID, $\alpha$=0.5', r'Orthogonal'], fontsize=18)
plt.tight_layout(pad=0.1)
plt.savefig('./Test Accuracy via different alpha 0.1.pdf')
plt.show()