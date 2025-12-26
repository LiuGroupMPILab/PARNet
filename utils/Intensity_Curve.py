import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from FBP_utils import *
from PIL import Image, ImageOps
def Curve(data_,modelOut,modelname):
    min_h = min(data_.shape[0], modelOut.shape[0])
    min_w = min(data_.shape[1], modelOut.shape[1])

    data_ = np.expand_dims(data_[ :min_h, :min_w], axis=0)
    modelOut = np.expand_dims(modelOut[ :min_h, :min_w], axis=0)
    images = np.concatenate([data_, modelOut], axis=0)  # [64, 26, 13]
    # images = images[0].reshape(1,images[0].shape[0],images[0].shape[1])
    # for i,img in enumerate(images):
    #     images[i] = np.rot90(img, k=1)
    models = ['Phantom', modelname]
    colors = ['red', 'blue']

    fig = plt.figure(figsize=(6, 4))  # 只创建一个画布，不生成多余子图

    # ---- 显示两张灰度图 ----
    # for i, ax in enumerate(axes.flat):
    #     img = np.squeeze(images[i])
    # ax.imshow(img, cmap='gray', aspect='equal')
    # ax.set_title(models[i], fontsize=12)
    # ax.axis('off')

    # ---- 新建一个坐标轴用于曲线 ----
    fig.subplots_adjust(bottom=0.25)  # 给曲线图留出空间
    ax_curve = fig.add_axes([0.15, 0.15, 0.5, 0.75])  # [left, bottom, width, height]

    # 画 data_ 和 modelOut 的强度曲线
    for i, img in enumerate(images):
        line_pos = img.shape[1]-3*img.shape[1]//5 - 1
        middle_row = img[line_pos, :].squeeze()
        intensity_profile = middle_row / middle_row.max()
        x = np.arange(img.shape[1])

        ax_curve.plot(x, intensity_profile, color=colors[i], linewidth=2, label=models[i])

    # 设置曲线图属性
    ax_curve.set_ylim(0, 1.5)
    ax_curve.set_xlim(0, img.shape[1] - 1)
    ax_curve.set_yticks([0, 0.5, 1])
    ax_curve.set_xticks([0, img.shape[1] // 2, img.shape[1] - 1])
    ax_curve.set_xticklabels([0, 50, 100], fontsize=12)
    ax_curve.set_xlabel("Pixel", fontsize=16)
    ax_curve.set_ylabel("Normalized intensity", fontsize=16)
    ax_curve.tick_params(axis='both', labelsize=12)
    ax_curve.grid(True, linestyle='--', alpha=0.5)
    ax_curve.legend(loc='upper left', fontsize=12)

    plt.show()

    output_img = images[1].squeeze()  # [H, W]

    plt.imshow(output_img, cmap='gray')
    plt.title('Model Output')
    plt.axis('off')

    # 中间列索引
    mid_col = line_pos

    # 在灰度图上画红色竖线
    plt.axhline(y=mid_col, color='red', linewidth=2)

    plt.show()


def load_two_images(folder_path, filename1, filename2):
    """
    从指定文件夹中读取两张图片，并返回二维数组形式
    :param folder_path: 图片所在文件夹
    :param filename1: 第一张图片文件名
    :param filename2: 第二张图片文件名
    :return: tuple (img1_array, img2_array)，均为二维 NumPy 数组
    """
    path1 = os.path.join(folder_path, filename1)
    path2 = os.path.join(folder_path, filename2)

    # 读取图片并转为灰度
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')  # jet图是彩色，先确保是RGB

    # 转成二维 NumPy 数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    return img1_array, img2_array


# 使用示例
folder = "datasets/ground/curve_M"
file1 = "1.png"
file2 = "6.png"

img1_array, img2_array = load_two_images(folder, file1, file1)
Curve(img1_array, img2_array, 'PAR')