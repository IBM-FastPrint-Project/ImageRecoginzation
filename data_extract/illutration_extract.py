from constant import DATAPATH, OUTPUTPATH

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import os
import numpy as np


def take_pixel(image):
    width, height, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = gray / 225

    pixels = []
    for i in range(width):
        for j in range(height):
            if im[i][j] > 0.5:
                pixels.append([i, j])

    return pixels


def visualization(datas, pred):
    x_data = []
    y_data = []
    color = []
    choose_colors = ['#4EACC5', '#FF9C34']
    for i in range(len(pred)):
        #     pred[i]
        x, y = datas[i]
        x_data.append(x)
        y_data.append(y)
        color.append(choose_colors[pred[i]])

    plt.scatter(x_data, y_data, color=color)
    plt.show()


def cluster(datas):
    # k-means
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_pred = k_means.fit_predict(datas)
    visualization(datas, k_pred)

    # mean shift no use as k-means
    # bandwidth = estimate_bandwidth(datas, quantile=0.2, n_samples=500)
    # meanshift = MeanShift(n_clusters=2, bandwidth=bandwidth, bin_seeding=True)
    # m_pred = meanshift.fit_predict(datas)
    # visualization(datas, m_pred)

    # spectral_cluster
    # spectral_cluster = SpectralClustering(n_clusters=2)
    # s_pred = spectral_cluster.fit_predict(datas)
    # visualization(datas, s_pred)
    return


def main():
    INPUT_IMG = '尺寸012Gerber 文件.jpg'
    input_img_path = os.path.join(DATAPATH, INPUT_IMG)
    image = cv2.imread(input_img_path)

    datas = take_pixel(image)
    cluster(datas)


if __name__ == '__main__':
    main()