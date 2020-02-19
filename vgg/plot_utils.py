import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os.path

def plot_dist(tensor,name):
    sns.set(color_codes=True)
    flat_tensor = tensor.view(-1)
    tensor_distri = sns.distplot(flat_tensor.cpu().numpy(),color="#006284");#D0104C 006284
    fig = tensor_distri.get_figure()
    sns.set_context("poster")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylabel("Frequency",fontsize=20)
    # plt.ylabel("Density")
    # plt.xlim(-1,7.5)
    fig.savefig("./plots/"+ name + ".eps")
    plt.close()
    # plt.show()
    print("ploted")

def plot_heatmap(tensor,name):
    for dim in range(tensor.shape[0]):
        tensor_2d = tensor.cpu().numpy()[dim,:,:]
        f, ax = plt.subplots(figsize=(9, 6))
        heatmap = sns.heatmap(tensor_2d, annot=True, linewidths=.5, ax=ax)
        fig = heatmap.get_figure()
        fig.savefig("./heats/"+ name + "_" + str(dim) + ".eps")
        plt.close()
        print("ploted")
