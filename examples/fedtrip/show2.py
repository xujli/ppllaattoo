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


dir = 'results/10_4/MNIST/lenet/fedavg/orthogonal/'
alpha = 0.5
sep = 5
plt.figure()
max_acc, max_std = [], []
mean_acc, mean_std = [], []
alphas = []
alphasss = np.arange(1, 30, 5).astype(np.float) / 10
fig = plt.figure()
for subdir in os.listdir(dir):
    if (subdir.endswith('{}'.format(alpha))) and (len(subdir.split('_')) == sep):
        acc1 = []
        if float(subdir.split('_')[-2]) in alphasss:
            for file in os.listdir(os.path.join(dir, subdir)):
                df = pd.read_csv(os.path.join(dir, subdir, file))
                acc = df['accuracy'].values
                acc1.append(acc)

            plt.plot(np.mean(np.array(acc1), axis=0), label=r'$\mu$=' + subdir.split('_')[-2])
            print(np.mean(np.array(acc1), axis=0)[-1])
plt.legend()
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.savefig('orthogonal.pdf')
plt.show(dpi=300)