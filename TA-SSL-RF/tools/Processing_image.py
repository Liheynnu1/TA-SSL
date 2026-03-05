# 导入需要的库
import os
import cv2

import argparse
# from skimage import io
# from osgeo import gdal
from PIL import Image
import scipy.io as scio
import numpy as np
import tifffile as tiff
from sklearn.decomposition import PCA
from util import standardization_org, add_zero
Image.MAX_IMAGE_PIXELS = None

# 定义一个函数，将传入的数据标准化
def standardization(image):
    # 将图片的数据类型转换为float32
    image = image.astype(np.float64)
    w, h, c = image.shape
    out = []
    for i in range(c):
        image2 = image[:, :, i]
        # image[:19, 0] = -8.5
        # image[image > 100] = 100 # lidar 数据需要控制在+-50以内
        # image[image < -100] = -100
        # image[image > 10000] = 10000  # lidar 数据需要控制在+-50以内

        # 计算平均值和标准差

        mean = np.mean(image2)
        std = np.std(image2)
        # # 定义一个异常值的范围，这里假设是平均值加减三倍标准差，可以根据实际情况修改
        # lower_bound = mean - 3 * std
        # upper_bound = mean + 3 * std
        #
        # # 检测并处理异常值，将超过范围的值替换为平均值，也可以用中位数或其他方法替换
        # image2[image2 < lower_bound] = mean
        # image2[image2 > upper_bound] = mean
        # 标准化图片
        normalized = (image2 - mean) / std
        # 线性变换图片，将最小值映射到0，最大值映射到255
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        scaled = (normalized - min_val) / (max_val - min_val) * 255
        out.append(scaled.reshape((w, h, 1)))
    out = np.concatenate(out, axis=2)
    return out


# 定义一个函数，将单通道图片合并为三通道图片
def merge_images(images):
    # 获取图片的宽度和高度
    # w = images[0].shape[1]
    # h = images[0].shape[0]
    h, w, c = images.shape
    # 创建一个空的三通道图片
    merged = np.zeros((h, w, 3), dtype=np.float64)
    # 将单通道图片按顺序赋值给三通道图片的每个通道
    for i in range(images.shape[-1]):
        # 获取单通道图片
        image = images[:, :, i]
        # image = standardization(image)
        # 将图片的数据类型转换为uint8
        # scaled = scaled.astype(np.uint8)
        # 赋值给三通道图片
        merged[:, :, i] = image

    # 返回合并后的图片
    return merged



# 定义一个函数，将图片的局部剪切，并保存为png格式
def split_and_save(image, output_dir):
    # for i in range(4):
    image = Image.fromarray(np.uint8(image[1202:2404, 1192:5960, :]))
    # cropped_img = image.crop(((i + 1) * 1192, 1202, (i + 2) * 1192, 2404))
    # 构造小块的文件名，格式为row_column.png
    filename = "lidar{}.png".format("")
    # 拼接小块的完整路径
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)


# 定义一个函数，将图片分割为224x224的小块，并保存为png格式
def split224_and_save(args, image, output_dir, c=0):
    # 获取图片的宽度和高度
    image = np.array(image)
    h = image.shape[1]
    w = image.shape[0]
    image = add_zero(args, image)
    # 计算分割后的小块的数量
    n_w = w // args.Shear_stride_x
    n_h = h // args.Shear_stride_y
    # imageze = np.zeros(((n_w+1)*224, (n_h+1)*224, 3))
    # imageze[:w, :h, :] = image
    # 遍历每个小块
    if args.suffix == 'png':
        for i in range(n_w+1):
            for j in range(n_h+1):
                # 获取小块的左上角和右下角的坐标
                x1 = j * args.Shear_stride_y
                y1 = i * args.Shear_stride_x
                x2 = x1 + 224
                y2 = y1 + 224
                # 截取小块
                patch = image[y1:y2, x1:x2, :]
                # 构造小块的文件名，格式为row_column.png
                filename = "{}_{}_{}_{}.png".format(args.dataset, i, j, c)
                # 拼接小块的完整路径
                filepath = os.path.join(output_dir, filename)
                # 将小块保存为png格式
                cv2.imwrite(filepath, patch)
    elif args.suffix == 'tif':
        image = np.transpose(image, (2, 0, 1))
        for i in range(n_w+1):
            for j in range(n_h+1):
                # 获取小块的左上角和右下角的坐标
                x1 = j * args.Shear_stride_y
                y1 = i * args.Shear_stride_x
                x2 = x1 + 224
                y2 = y1 + 224
                # 截取小块
                patch = image[:, y1:y2, x1:x2]
                # 构造小块的文件名，格式为row_column.png
                filename = "{}_{}_{}_{}.tif".format(args.dataset, i, j, c)
                filepath = os.path.join(output_dir, filename)
                # 将小块保存为tif格式
                tiff.imwrite(filepath, patch)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Backbone')
    parser.add_argument("--image_size", default=224, type=int, help="切块的数量")
    parser.add_argument("--suffix", default='tif', type=str, help="tif, png")
    # parser.add_argument("--dataset", default='PaviaU', type=str,
    #                     help="使用的数据集：PaviaU， Salinas, Trento, 'Indian', 'Houston'")
    # parser.add_argument("--out_channel", default=3, type=int, help="2或者3")
    # parser.add_argument("--Shear_stride_x", default=15, type=int, help="15或者50")
    # parser.add_argument("--Shear_stride_y", default=20, type=int, help="10或者50")
    parser.add_argument("--out_path", default='out', type=str)
    args = parser.parse_args()
    return args

# img = tiff.imread(r'E:\SVM\tools\data_org\HSI_out\Houston_0_0_0.tif')
# img = np.transpose(img, (1, 2, 0)).astype('uint8')
# img = Image.fromarray(img)
# img = Image.open(r'E:\SVM\tools\data_org\HSI_out\Houston_0_0_0.png')
# img = np.array(img)
args = parse_args()
# 定义输入文件夹和输出文件夹的路径，可以根据实际情况修改
# input_dir = r"D:\gw\GW-main\SVM\data\dataset\Trento.mat"
output_dir = r"HSI_out"
is_image = True
# 如果输出文件夹不存在，就创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# data = gdal.Open(input_dir)
# # num_bands = data.RasterCount     # 获取波段数
# # print(num_bands)
# tmp_img = data.ReadAsArray()      #将数据转为数组
# image1s = tmp_img.transpose(1, 2, 0)     #由波段、行、列——>行、列、波段


for dataset in ['Indian']:
    args.dataset = dataset
    if args.dataset == 'Salinas':
        args.dataset_img = 'salinas_corrected'
        args.dataset_shape = [512, 217, 204]
        test_image1_paths = r'data/dataset/Salinas_corrected.mat'
    elif args.dataset == 'PaviaU':
        args.dataset_img = 'paviaU'
        args.dataset_shape = [610, 340, 103]
        test_image1_paths = r'data/dataset/PaviaU.mat'
    elif args.dataset == 'Trento':
        args.dataset_img = 'HSI'
        args.dataset_shape = [166, 600, 63]
        test_image1_paths = r'data/dataset/Trento.mat'
    elif args.dataset == 'Indian':
        args.dataset_img = 'indian_pines_corrected'
        args.dataset_shape = [145, 145, 200]
        test_image1_paths = r'data/dataset/Indian_pines_corrected.mat'
    else:
        args.dataset_img = 'img'
        args.dataset_label = 'labels'
        args.dataset_shape = [349, 1905, 144]
        test_image1_paths = r'data/dataset/Houston.mat'
    file_path = os.getcwd()[:-6]
    image1_paths = os.path.join(file_path, test_image1_paths)
    image1s = scio.loadmat(image1_paths)[args.dataset_img]
    imgs = []
    # HSI = np.concatenate(image1s, axis=2) if isinstance(image1s, list) else image1s
    # h, w, c = HSI.shape
    # HSI1 = HSI.copy().reshape(h * w, c)
    # pca = PCA(n_components=args.out_channel)
    # for i, c_idx in enumerate(range(0, c, c//args.out_channel)):
    #     if i+1 > args.out_channel:break
    #     c_idx_2 = c_idx+int(c/args.out_channel) if c-c_idx >= int(c/args.out_channel) else c
    #     HSI2 = pca.fit_transform(HSI1[:, c_idx:c_idx_2])  # 10000，3
    #     HSI2 = HSI2.reshape(h, w, args.out_channel)
    #     HSI2 = standardization_org(HSI2, remove_exception=True)
        imgs.append(HSI2)  # list [

    img = np.concatenate(imgs, axis=2)
    # c = img.shape[-1]
    # for c_idx in range(0, c, args.out_channel):  # 将数据拆分为三通道
    #     # if img.shape[-1] - c_idx <= 3:
    #     #     img2 = merge_images(img[:, :, c_idx:img.shape[-1]])
    #     #     # out.append(img2)
    #     #     split224_and_save(img2, output_dir, c=c_idx, view_idx=view_idx)
    #     # else:
    #     # if c-c_idx-3 <= 2:  # 舍弃最后一个视图
    #     #     print(c_idx)
    #     #     continue
    split224_and_save(args, img[:, :, c_idx:c_idx + args.out_channel], output_dir, c=c_idx)
    print(f"完成{args.dataset}数据处理！")

# # 定义输入文件夹和输出文件夹的路径，可以根据实际情况修改
# input_dir = r"D:\gw\GW-main\SVM\data_org\img"
# output_dir = r"D:\gw\GW-main\SVM\data_org\img_out"
# is_image = True
# # 如果输出文件夹不存在，就创建它
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# # 获取输入文件夹中的所有文件名，并按字母顺序排序  处理高光谱和rgb数据
# filenames = sorted(os.listdir(input_dir))
#
# # 创建一个空列表，用于存储单通道图片
# images = []
# name = 3 if is_image else 0
# # 遍历每个文件名
# for filename in filenames:
#     # 拼接文件的完整路径
#     filepath = os.path.join(input_dir, filename)
#     # 判断文件是否是图片格式，如果不是就跳过
#     if not filepath.endswith((".jpg", ".png", ".bmp", "tif", "mat")):
#         continue
#
#     # 使用cv2读取图片，注意cv2默认读取为BGR格式，所以需要转换为灰度格式（单通道）
#     image = io.imread(filepath)  # 读取RGB原图像
#     # 将图片添加到列表中
#     if is_image:
#         width, height, c = image.shape  # 获取原始图片的宽高
#         new_width = width // 20  # 计算新的宽度
#         new_height = height // 20  # 计算新的高度
#         new_size = (new_height, new_width)  # 构造新的尺寸元组
#         image = Image.fromarray(image)
#         print(new_width, new_height, c)
#         img_resized = image.resize(new_size, Image.BILINEAR)  # 调用resize()函数
#         # img_resized.show()
#         split224_and_save(img_resized, output_dir, name, 0)
#         print(name)
#         name += 1
#         continue
#     images.append(image)
#
#     # 如果列表中有三张图片，就合并为三通道图片，并分割并保存为小块，然后清空列表，继续下一组
#     if len(images) == 3:
#         merged = merge_images(images)
#         split_and_save(merged, output_dir)
#         images.clear()
#         name += 1
#
# # 如果列表中还有剩余的图片，就用0补齐三张图片，然后合并为三通道图片，并分割并保存为小块，然后清空列表，结束程序
# if len(images) > 0 and is_image == False:
#     while len(images) < 3:
#         images.append(np.zeros_like(images[0]))
#
#     merged = merge_images(images)
#     split_and_save(merged, output_dir)
#     images.clear()
#
# print("完成数据处理！")
