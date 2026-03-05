# 导入需要的库
import os
import cv2
import numpy as np
from skimage import io
from PIL import Image




# 定义一个函数，将单通道图片合并为三通道图片
def merge_images(images):
    # 获取图片的宽度和高度
    w = images[0].shape[1]
    h = images[0].shape[0]
    # 创建一个空的三通道图片
    merged = np.zeros((h, w, 3), dtype=np.float64)
    # 将单通道图片按顺序赋值给三通道图片的每个通道
    a = False
    for i in range(len(images)):
        # 获取单通道图片
        image = images[i]
        # 将图片的数据类型转换为float32
        image = image.astype(np.float64)
        # image[image > 10000] = 10000
        image[:19, 0] = -8.5
        image[image > 100] = 100  # lidar 数据需要控制在+-50以内
        image[image < -100] = -100
        # 计算平均值和标准差
        mean = np.mean(image)
        std = np.std(image)
        # 定义一个异常值的范围，这里假设是平均值加减三倍标准差，可以根据实际情况修改
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        # 检测并处理异常值，将超过范围的值替换为平均值，也可以用中位数或其他方法替换
        image[image < lower_bound] = mean
        image[image > upper_bound] = mean
        # 标准化图片
        normalized = (image - mean) / std
        # 线性变换图片，将最小值映射到0，最大值映射到255
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        scaled = (normalized - min_val) / (max_val - min_val) * 255
        # 将图片的数据类型转换为uint8
        # scaled = scaled.astype(np.uint8)
        # 赋值给三通道图片
        merged[:, :, i] = scaled
        # img = images[i]
        # img = img.astype(np.float64)
        # img[img > 10000] = 10000
        #
        # mean = np.mean(img)
        # std = np.std(img)
        # if mean>1000:
        #   img = img/10000
        # normalization = (img)/std*255
        # min_val = np.min(img)
        # max_val = np.max(img)
        # # img= img/mean
        # scaled = (img - min_val)/(max_val - min_val)*255
        # # scaled = scaled.astype(np.uint8)
        # merged[:, :, i] = scaled
    # 返回合并后的图片
    return merged


# 定义一个函数，将图片分割为256x256的小块，并保存为png格式
def split_and_save(image, output_dir):
    # # for i in range(4):
    # image = Image.fromarray(np.uint8(image[1202:2404, 1192:5960, :]))
    # # cropped_img = image.crop(((i + 1) * 1192, 1202, (i + 2) * 1192, 2404))
    # # 构造小块的文件名，格式为row_column.png
    # filename = "lidar{}.png".format(name)
    # # 拼接小块的完整路径
    # filepath = os.path.join(output_dir, filename)
    # image.save(filepath)

    # 获取图片的宽度和高度
    w = image.shape[1]
    h = image.shape[0]
    # 计算分割后的小块的数量
    n_w = w // 224
    n_h = h // 224
    # 遍历每个小块
    for i in range(n_h):
        for j in range(n_w):
            # 获取小块的左上角和右下角的坐标
            x1 = j * 224
            y1 = i * 224
            x2 = x1 + 224
            y2 = y1 + 224
            # 截取小块
            patch = image[y1:y2, x1:x2, :]
            # 构造小块的文件名，格式为row_column.png
            filename = "lidar{}_{}_{}.png".format(name, i, j)
            # 拼接小块的完整路径
            filepath = os.path.join(output_dir, filename)
            # 将小块保存为png格式
            cv2.imwrite(filepath, patch)


# 定义输入文件夹和输出文件夹的路径，可以根据实际情况修改

input_dir = r"D:\gw\GW-main\SVM\data_org\HSI"
output_dir = r"D:\gw\GW-main\SVM\data\a_org2"
is_image = False
# 如果输出文件夹不存在，就创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取输入文件夹中的所有文件名，并按字母顺序排序
filenames = sorted(os.listdir(input_dir))

# 创建一个空列表，用于存储单通道图片
images = []
name = 3 if is_image else 0
# 遍历每个文件名
for filename in filenames:
    # 拼接文件的完整路径
    filepath = os.path.join(input_dir, filename)
    # 判断文件是否是图片格式，如果不是就跳过
    if not filepath.endswith((".jpg", ".png", ".bmp", "tif")):
        continue

    # 使用cv2读取图片，注意cv2默认读取为BGR格式，所以需要转换为灰度格式（单通道）
    image = io.imread(filepath)  # 读取RGB原图像
    # 将图片添加到列表中
    if is_image:
        split_and_save(image, output_dir)
        print(name)
        name += 1
        continue
    images.append(image)

    # 如果列表中有三张图片，就合并为三通道图片，并分割并保存为小块，然后清空列表，继续下一组
    if len(images) == 3:
        merged = merge_images(images)
        split_and_save(merged, output_dir)
        images.clear()
        name += 1

# 如果列表中还有剩余的图片，就用0补齐三张图片，然后合并为三通道图片，并分割并保存为小块，然后清空列表，结束程序
if len(images) > 0 and is_image == False:
    while len(images) < 3:
        images.append(np.zeros_like(images[0]))

    merged = merge_images(images)
    split_and_save(merged, output_dir)
    images.clear()

print("完成数据处理！")
