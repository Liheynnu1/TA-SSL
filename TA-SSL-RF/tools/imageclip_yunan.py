# -*- coding: UTF-8 -*-
from osgeo import gdal, ogr, gdalconst
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
import time
# 读取数据，并对数据进行标准化处理
def read_img(img_path):
    """读取遥感数据信息"""
    start = time.perf_counter()
    dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    adf_GeoTransform = dataset.GetGeoTransform()
    im_Proj = dataset.GetProjection()
    img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=np.float64)  # 将数据写成数组，对应栅格矩阵
    # 标准化数据
    mean = np.mean(img_data)
    std = np.std(img_data)
    normalized = (img_data - mean) / std
    # 线性变换图片，将最小值映射到0，最大值映射到255
    min_val = np.min(normalized)
    max_val = np.max(normalized)
    scaled = (normalized - min_val) / (max_val - min_val) * 255
    # print(img_width, img_height)
    del dataset
    end = time.perf_counter()
    print('readimage执行时间：', end-start)
    return scaled
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
    n_w = w // 224
    n_h = h // 224
    # n_w = w // args.Shear_stride_x
    # n_h = h // args.Shear_stride_y
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
    parser.add_argument("--Shear_stride_x", default=5, type=int, help="15或者50")
    parser.add_argument("--Shear_stride_y", default=5, type=int, help="10或者50")
    # parser.add_argument("--out_path", default=10, type=int)
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # 读取所有特征
    # 地理环境特征
    elevlation = read_img(r"E:\Forest3\Feature\Feature_clip\elevlation.tif")
    raw31 = np.zeros((elevlation.shape[0], 1))
    elevation = np.insert(elevlation, [0], raw31, axis=1)
    col31 = np.zeros(elevation.shape[1])
    elevation = np.insert(elevation, [0], col31, axis=0)
    slope = read_img(r"E:\Forest3\Feature\Feature_clip\slope.tif")
    # Precmean = read_img(r"E:\Forest3\Feature\Feature_clip\Precmean.tif")
    # # 光谱和极化特征
    # B1 = read_img(r"E:\Forest3\Feature\Feature_clip\B1_Rej_YN.tif")
    # B5 = read_img(r"E:\Forest3\Feature\Feature_clip\B5_Rej_YN.tif")
    # B6 = read_img(r"E:\Forest3\Feature\Feature_clip\B6_Rej_YN.tif")
    # B9 = read_img(r"E:\Forest3\Feature\Feature_clip\B9_Rej_YN.tif")
    # EVI = read_img(r"E:\Forest3\Feature\Feature_clip\EVI_Rej_YN.tif")
    # NDVI = read_img(r"E:\Forest3\Feature\Feature_clip\NDVI_Rej_YN.tif")
    # VH = read_img(r"E:\Forest3\Feature\Feature_clip\VH_Rej_YN.tif")
    # ratio = read_img(r"E:\Forest3\Feature\Feature_clip\ratio_Rej_YN.tif")
    # mRVI = read_img(r"E:\Forest3\Feature\Feature_clip\mRVI_Rej_YN.tif")
    # # 物候特征
    # mRVI_cv = read_img(r"E:\Forest3\Feature\Feature_clip\mRVIcv_Rej_YN.tif")
    # mRVI_zf = read_img(r"E:\Forest3\Feature\Feature_clip\mRVIz_Rej_YN.tif")
    # NDI_std = read_img(r"E:\Forest3\Feature\Feature_clip\NDIstdDev_Rej_YN.tif")
    # REPI_maxd = read_img(r"E:\Forest3\Feature\Feature_clip\REPImaxd_Rej_YN.tif")
    # REPI_mind = read_img(r"E:\Forest3\Feature\Feature_clip\REPImind_Rej_YN.tif")
    # NDVI_zf = read_img(r"E:\Forest3\Feature\Feature_clip\NDVIz_Rej_YN.tif")
    # # 纹理特征
    # B5_contract = read_img(r"E:\Forest3\Feature\Feature_clip\B5contras_Rej_YN.tif")
    # # 垂直结构特征
    # FH2023 = read_img(r"E:\Forest3\Feature\Feature_clip\FH2023_Rej_YN.tif")
    # FH_ZFL = read_img(r"E:\Forest3\Feature\Feature_clip\FHZFL_Rej_YN.tif")
    # FH_speedy2023L = read_img(r"E:\Forest3\Feature\Feature_clip\FHspeedy2023L_Rej_YN.tif")
    # FH_meanL = read_img(r"E:\Forest3\Feature\Feature_clip\FHmeanL_Rej_YN.tif")
    # # 生长速度特征
    # NDVI_meanL = read_img(r"E:\Forest3\Feature\Feature_clip\NDVImeanL_Rej_YN.tif")
    # NDVI_speedy2023L = read_img(r"E:\Forest3\Feature\Feature_clip\NDVIspeedy2023L_Rej_YN.tif")
    # NDVI_ZFL = read_img(r"E:\Forest3\Feature\Feature_clip\NDVIZFL_Rej_YN.tif")
    # Texture_ZFL = read_img(r"E:\Forest3\Feature\Feature_clip\TextureZFL_Rej_YN.tif")
    # Texture_speedyL = read_img(r"E:\Forest3\Feature\Feature_clip\TexturespeedyL_Rej_YN.tif")
    # 合并所有特征
    # consta1 = np.stack((elevation, slope, Precmean, B1, B5, B6, B9, EVI, NDVI, VH, ratio, mRVI, mRVI_cv, mRVI_zf,
    #                     NDI_std, REPI_maxd, REPI_mind, NDVI_zf, B5_contract,
    #                     FH2023, FH_ZFL, FH_speedy2023L, FH_meanL, NDVI_meanL, NDVI_speedy2023L, NDVI_ZFL, Texture_ZFL,
    #                     Texture_speedyL))
    consta1 = np.stack((elevation, slope))
    image1s = consta1.transpose(1, 2, 0)# 由波段、行、列——>行、列、波段
    # 批量将多维特征裁剪为224*224的图像
    # 定义输出文件夹的路径
    output_dir = r"D:\ForestClass2\self_supeervLeaning\feature224"
    is_image = True
    # 如果输出文件夹不存在，就创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = parse_args()
    split224_and_save(args, image1s, output_dir, 0)
    print(f"完成{args.dataset}数据处理！")


