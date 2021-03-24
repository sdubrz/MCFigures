import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image
import os


def small_image(eta=0.5, in_path="", out_path=""):
    """
    将图片缩小
    :param eta: 缩小的倍数
    :param in_path: 读取图片路径
    :param out_path: 保存缩小后的图片的路径
    :return:
    """
    for root, middle, files in os.walk(in_path):
        for file_name in files:
            im = Image.open(in_path+file_name)
            (x, y) = im.size
            s_img = im.resize((int(x*eta), int(y*eta)), Image.ANTIALIAS)
            s_img.save(out_path+file_name)


def get_image(path):
    return OffsetImage(plt.imread(path))


def mnist_images(path=None, data_path=None, eta=0.4, y_name="PCA.csv", label=None, image_shape=(28, 28), colormap='gray', inv=False, enlarge=1, trans=False):
    """
    用MNIST数据画艺术散点图
    :return:
    """

    small_path = path + "smallImages\\"
    if not os.path.exists(small_path):
        os.makedirs(small_path)
    small_image(eta=eta, in_path=path+"images\\", out_path=small_path)
    Y = np.loadtxt(data_path + y_name, dtype=np.float, delimiter="\t")
    (n, m) = Y.shape
    fig, ax = plt.subplots()
    # plt.colormaps()
    ax.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2])

    if colormap == 'gray':
        plt.set_cmap(cm.gray)

    ptr1 = 2115
    ptr2 = 200
    for i in range(0, 100):
        ab = AnnotationBbox(get_image(small_path + str(i+ptr1)+".png"), (Y[ptr2+i, 0], Y[ptr2+i, 1]), frameon=False)
        ax.add_artist(ab)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    path = "E:\\IRC\\WCFigures\\Figure3\\raw\\image2\\"
    data_path = "E:\\IRC\\WCFigures\\Figure3\\raw\\results\\"
    method = ["dynamic tsne", "equal initialization", "joint tsne", "random initialization"]
    data_path = data_path + method[3] + "\\"
    y_name = "dr_1.txt"  # 1
    mnist_images(path=path, data_path=data_path, eta=0.7, y_name=y_name)


