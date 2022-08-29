from matplotlib import pyplot as plt
import seaborn as sns

fontsize = 13


def plot_acc(train_acc, val_acc, model):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(val_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.plot(x, val_acc, alpha=0.9, linewidth=2, label='val acc')
    plt.xlabel('Iter')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/{}/acc.png'.format(model), dpi=400)

