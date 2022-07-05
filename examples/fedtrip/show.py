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
alpha = 0.5
sep = 4
plt.figure()
max_acc, max_std = [], []
mean_acc, mean_std = [], []
alphas = []
alphasss = np.arange(1, 26).astype(np.float) / 10

for subdir in os.listdir(dir):
    if (subdir.endswith('{}'.format(alpha))) and (len(subdir.split('_')) == sep):
        acc1 = []
        if float(subdir.split('_')[-2]) in alphasss:
            for file in os.listdir(os.path.join(dir, subdir)):
                df = pd.read_csv(os.path.join(dir, subdir, file))
                acc = df['accuracy'].values
                acc1.append(acc)
            try:
                acc1 = np.array(acc1)
                idx = np.argmax(np.mean(acc1, axis=0))
                print(idx, acc1.shape)
                max_acc.append(np.mean(acc1, axis=0)[idx])
                max_std.append(np.std(acc1[:, idx]))
                indices = np.where(acc1 >= 85)
                accs = [indices[1][0]]
                for index in range(1, len(indices[0])):
                    if indices[0][index] > indices[0][index - 1]:
                        accs.append(indices[1][index])
                mean_acc.append(np.mean(accs))
                mean_std.append(np.std(accs))
                alphas.append(float(subdir.split('_')[-2]))
            except:
                Exception()

max_acc = np.array(max_acc)
max_std = np.array(max_std)
print(max_acc)
mean_acc = np.array(mean_acc)
mean_std = np.array(mean_std)
scatter = plt.scatter(alphas, max_acc, s=max_std * 100,
            alpha=1, color=cmap_list3[0], label='Test Accuracy')
# plt.legend(*scatter.legend_elements(prop='sizes', num = 6))
# plt.ylim(85, 95) # (88, 95)
plt.grid(axis='y', linestyle='--', linewidth=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'Value of $\alpha$', fontsize=18)
plt.ylabel('Test Accuracy(%)', fontsize=18)
plt.twinx()
plt.scatter(alphas, mean_acc, s=mean_std * 40,
         alpha=1, color=cmap_list3[2], label='# of Rounds')

plt.yticks(fontsize=16)
# plt.ylim(22, 60) # (10, 40)
plt.ylabel('# of Round that Test Accuracy reaches 90%', fontsize=16)
# plt.legend([r'IID', r'Non-IID, $\alpha$=0.1', r'Non-IID, $\alpha$=0.5', r'Orthogonal'], fontsize=18)
plt.tight_layout(pad=0.1)
plt.savefig('./Test Accuracy via different alpha {}.png'.format(alpha))
plt.show()