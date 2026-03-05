import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage import io
# from osgeo import gdal
from PIL import Image
# 读取txt文件中的标签信息
with open('2013_IEEE_GRSS_DF_Contest_Samples_TR.txt', 'r') as f:
    lines = f.readlines() # 读取所有行
    a = str(lines[3].split(":")[1])[:3]
    num_rois = int(a) # 读取ROI的个数

    labels = np.zeros((349, 1905), dtype=np.uint8) # 初始化标签矩阵为全零
    i = 9  # 跳过前六行，从第七行开始处理
    for k in range(num_rois):  # 对每个ROI进行处理
        a = str(lines[i+2].split(":")[1])[:-1]
        roi_npts = int(a)  # 读取ROI的点数
        i += 4 # 跳过空行、名称、RGB值和表头，从第五行开始处理
        for j in range(roi_npts): # 对每个点进行处理
            point = list(map(int, lines[i+j].split()[1:3])) # 读取点的坐标
            labels[point[1]-1, point[0]-1] = k+1 # 将对应位置的标签设为k+1，注意坐标从1开始，而索引从0开始
        i += roi_npts + 2 # 跳过点的行数和一个空行，进入下一个ROI

# 保存标签矩阵为mat文件
savemat('Houston_gt.mat', {'labels': labels})

# 读取tif文件中的图像数据
img = io.imread(r'C:\Users\DELL\Downloads\2013_DFTC\2013_DFTC\2013_IEEE_GRSS_DF_Contest_CASI.tif')  # 读取tif文件

# 保存图像数据为mat文件
savemat('Houston.mat', {'img': img})