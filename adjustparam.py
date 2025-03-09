import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import get_bpf
from data import MagCompData, MagCompDataset
from config import config

config = config()
MagCompData = MagCompData(config)


def pca_adjust_param():
    xyzs = MagCompData.xyzs
    train_inds = MagCompData.train_inds
    test_inds = MagCompData.test_inds
    selected_features = config.selected_features
    lpf = get_bpf(pass1=0.0, pass2=0.2, fs=10.0)
    custom_dataset = MagCompDataset(xyzs=xyzs, train_inds=train_inds, test_inds=test_inds, features=selected_features, lpf=lpf,
                                    mode='train', is_pca=False)

    # 训练 PCA
    pca = PCA()
    print(custom_dataset.x.size())
    pca.fit(custom_dataset.x)

    # 计算累积方差贡献率
    explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    pca_adjust_param()
