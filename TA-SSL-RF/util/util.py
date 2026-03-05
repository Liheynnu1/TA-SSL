import torch
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from PIL import Image
import random
from util.sample import aligned_sample
from sklearn.svm import SVC
from scipy.signal import wiener

def convert_unit(flops, params):
    # 定义一个单位列表，从小到大排列
    units = [' ', 'K', 'M', 'G', 'T']
    # 初始化一个空列表，用于存放转换后的结果
    results = []
    # 对于FLOPs和Params，分别进行转换
    for value in [flops, params]:
        # 初始化一个指数，用于记录单位的位置
        index = 0
        # 如果值大于等于1000，就除以1000，并将指数加一，直到值小于1000为止
        while value >= 1000:
            value /= 1000.0
            index += 1
        # 将转换后的值和对应的单位拼接成一个字符串，并添加到结果列表中
        results.append('{:.2f} {} '.format(value, units[index]))
    # 返回结果列表
    return results


# 定义一个函数，计算混淆矩阵
def confusion_matrix(y_true, y_pred):
    y_true = y_true - 1
    y_pred = y_pred - 1
    # 获取类别数
    n_classes = len(np.unique(y_true))
    # 初始化混淆矩阵
    matrix = np.zeros((n_classes, n_classes))
    # 遍历每个样本
    for i in range(len(y_true)):
        # 累加真实类别和预测类别对应的位置
        matrix[y_true[i], y_pred[i]] += 1
    # 返回混淆矩阵
    return matrix


# 定义一个函数，计算总体精度
def overall_accuracy(y_true, y_pred):
    # 计算混淆矩阵
    matrix = confusion_matrix(y_true, y_pred)
    # 计算正确分类的样本数
    correct = np.trace(matrix)
    # 计算总样本数
    total = np.sum(matrix)
    # 计算并返回总体精度
    return correct / total


# 定义一个函数，计算平均精度
def average_accuracy(y_true, y_pred):
    # 计算混淆矩阵
    matrix = confusion_matrix(y_true, y_pred)
    # 获取类别数
    n_classes = len(np.unique(y_true))
    # 初始化平均精度
    average = 0
    # 遍历每个类别
    for i in range(n_classes):
        # 计算该类别的精度
        accuracy = matrix[i, i] / np.sum(matrix[i, :])
        # 累加到平均精度
        average += accuracy
        # 计算并返回平均精度
    return average / n_classes


# 定义一个函数，计算Kappa系数
def kappa_coefficient(y_true, y_pred):
    # 计算混淆矩阵
    matrix = confusion_matrix(y_true, y_pred)
    # 计算正确分类的样本数
    correct = np.trace(matrix)
    # 计算总样本数
    total = np.sum(matrix)
    # 计算随机分类的期望值
    expected = np.sum(matrix, axis=0) * np.sum(matrix, axis=1) / total
    # 计算并返回Kappa系数
    return ((correct - expected) / (total - expected)).mean()


def print_time(filename, last=""):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_str = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{time_str} {filename}]：', end=last)


def fix_random_seeds(seed=42):
    """
    Fix random seeds. 31
    """
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# def add_zero(args, image: np.array, no_c=False):
#     # image = np.array(image)
#     if no_c:
#         w, h = image.shape
#     else:
#         w, h, c = image.shape
#     piece = args.image_size
#     margin = int(args.piece_size/2-1)
#     # 计算分割后的小块的数量
#     n_w = w // piece
#     n_h = h // piece
#     if no_c:
#         imageze = np.zeros(((n_w + 1) * piece, (n_h + 1) * piece))
#         imageze[:w, :h] = image
#         out = np.zeros(((n_w + 1) * piece+margin*2, (n_h + 1) * piece+margin*2))
#         out[margin:(n_w + 1) * piece+margin, margin:(n_h + 1) * piece+margin] = imageze
#     else:
#         imageze = np.zeros(((n_w + 1) * piece, (n_h + 1) * piece, c))
#         imageze[:w, :h, :] = image
#         out = np.zeros(((n_w + 1) * piece+margin*2, (n_h + 2) * piece+margin*2, c))
#         out[margin:(n_w + 1) * piece+margin, margin:(n_h + 1) * piece+margin, :] = imageze
#     return out

def add_zero(args, image: np.array, no_c=False):
    if no_c:
        w, h = image.shape
    else:
        w, h, c = image.shape
    piece = int(args.image_size / 2)

    if no_c:
        imageze = np.zeros((w + piece * 2 - 1, h + piece * 2 - 1))
        imageze[piece:w + piece, piece:h + piece] = image
    else:
        imageze = np.zeros((w + piece * 2 - 1, h + piece * 2 - 1, c))
        imageze[piece:w + piece, piece:h + piece, :] = image
        # Fill top and bottom edges
    for i in range(piece):
        imageze[i, :, :] = imageze[2 * piece - i, :, :]
        if i == piece - 1:
            continue
        imageze[w + piece + i, :, :] = imageze[w + piece - i - 1, :, :]

    # Fill left and right edges
    for i in range(piece):
        imageze[:, i, :] = imageze[:, 2 * piece - i, :]
        if i == piece - 1:
            continue
        imageze[:, h + piece + i, :] = imageze[:, h + piece - i - 1, :]
    return imageze


def standardization_org(image, cat=False, remove_exception=False):
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
        if remove_exception:
            # 定义一个异常值的范围，这里假设是平均值加减三倍标准差，可以根据实际情况修改
            lower_bound = mean - 1 * std
            upper_bound = mean + 1 * std

            # 检测并处理异常值，将超过范围的值替换为平均值，也可以用中位数或其他方法替换
            image2[image2 < lower_bound] = lower_bound
            image2[image2 > upper_bound] = upper_bound
        # 标准化图片
        normalized = (image2 - mean) / std
        # 线性变换图片，将最小值映射到0，最大值映射到255
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        scaled = (normalized - min_val) / (max_val - min_val) * 255
        out.append(scaled.reshape((w, h, 1)))
    out = np.concatenate(out, axis=2)
    if cat:
        return out[601:1202, 596:2980].reshape((601, 2384, c))
    else:
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

# def Composite_feature_map(args, data, data2=None):
#     data = np.concatenate(data, axis=2)
#     h, w, c = data.shape
#     data = data.reshape(h * w, c)
#     out = []
#     out_org = []
#     imgs = []
#     for view_idx in args.views:
#         pca = PCA(n_components=view_idx)
#         for c_idx in range(0, c, args.views_number):
#             if c - c_idx >= args.views_number:
#                 c_idx_2 = c_idx + args.views_number
#             else:
#                 continue
#             HSI2 = pca.fit_transform(data[:, c_idx:c_idx_2])
#             HSI2 = HSI2.reshape(h, w, view_idx)
#             HSI2 = standardization_org(HSI2, remove_exception=True)
#             imgs.append(HSI2)
#         if view_idx == 0:
#             imgs.append(data.reshape(h, w, c))
#         img = np.concatenate(imgs, axis=2)
#         for c_idx in range(0, img.shape[-1], 3):  # 将数据拆分为三通道
#             if img.shape[-1] - c_idx <= 3:
#                 img2 = merge_images(img[:, :, c_idx:img.shape[-1]])
#                 out.append(Image.fromarray(np.array(add_zero(args, img2), dtype=np.uint8)))
#             else:
#                 out.append(Image.fromarray(np.array(add_zero(args, img[:, :, c_idx:c_idx + 3]), dtype=np.uint8)))
#     return out
# def Composite_feature_map(args, data, data2=None):
#     data = np.concatenate(data, axis=2)
#     h, w, c = data.shape
#     data = data.reshape(h * w, c)
#     out = []
#     out_org = []
#     imgs = []
#     imgs2 = []
#     for view_idx in args.views:
#         if view_idx == 0:
#             imgs2.append(data.reshape(h, w, c))
#             continue
#         pca = PCA(n_components=view_idx)
#         for c_idx in range(0, c, args.views_number):
#             if c - c_idx >= args.views_number:
#                 c_idx_2 = c_idx + args.views_number
#             else:
#                 continue
#             HSI2 = pca.fit_transform(data[:, c_idx:c_idx_2])
#             # if args.used_views:
#             HSI2 = HSI2.reshape(h, w, view_idx)
#             HSI2 = standardization_org(HSI2, remove_exception=True)
#             imgs.append(HSI2)
#
#         img = np.concatenate(imgs, axis=2)
#         out_org = np.concatenate(imgs2, axis=2) if 0 in args.views else []
#         for c_idx in range(0, img.shape[-1], 3):  # 将数据拆分为三通道
#             if img.shape[-1] - c_idx <= 3:
#                 img2 = merge_images(img[:, :, c_idx:img.shape[-1]])
#                 # out.append(img2)
#                 out.append(Image.fromarray(np.array(add_zero(args, img2), dtype=np.uint8)))
#             else:
#                 # out.append(img[:, :, c_idx:c_idx+3])
#                 out.append(Image.fromarray(np.array(add_zero(args, img[:, :, c_idx:c_idx + 3]), dtype=np.uint8)))
#     return out, out_org if 0 in args.views else []

def Composite_feature_map(args, data):
    data = np.concatenate(data, axis=2)
    h, w, c = data.shape
    data = data.reshape(h * w, c)
    out = []
    imgs = []
    imgs2 = []
    for view_idx in args.views:
        if view_idx == 0:
            continue
        pca = PCA(n_components=view_idx)
        for c_idx in range(0, c, args.views_number):
            if c - c_idx >= args.views_number:  # 去除最后一个视图
                c_idx_2 = c_idx + args.views_number
            else:
                continue
            HSI2 = pca.fit_transform(data[:, c_idx:c_idx_2])
            # if args.used_views:
            HSI2 = HSI2.reshape(h, w, view_idx)
            HSI2 = standardization_org(HSI2, remove_exception=True)
            imgs.append(HSI2)

        img = np.concatenate(imgs, axis=2)
        for c_idx in range(0, img.shape[-1], 3):  # 将数据拆分为三通道

            if img.shape[-1] - c_idx <= 3:
                img2 = merge_images(img[:, :, c_idx:img.shape[-1]])
                out.append((np.array(add_zero(args, img2), dtype=np.uint8)))
            else:
                # out.append(img[:, :, c_idx:c_idx+3])
                out.append((np.array(add_zero(args, img[:, :, c_idx:c_idx + 3]), dtype=np.uint8)))
    # if 0 in args.views:
    #     out_org = []
    #     data = data.reshape(h, w, c)
    #     for c_idx in range(0, data.shape[-1], 3):  # 将数据拆分为三通道
    #         if data.shape[-1] - c_idx <= 3:
    #             img2 = merge_images(data[:, :, c_idx:data.shape[-1]])
    #             out_org.append(Image.fromarray(np.array(img2, dtype=np.uint8)))
    #         else:
    #             out_org.append(Image.fromarray(np.array(data[:, :, c_idx:c_idx + 3], dtype=np.uint8)))
    #     return out, out_org
    if 0 in args.views:
        out_org = np.array(data.reshape(h, w, c), dtype=np.uint8)
        return out, out_org
    else:
        return out, []


def soft_composite_feature_map(args, data):
    data = np.concatenate(data, axis=2)
    h, w, c = data.shape
    data = data.reshape(h * w, c)
    out = []
    imgs = []
    in_channel = args.in_channel
    for view_idx in args.views:
        if view_idx == 0:
            continue
        pca = PCA(n_components=view_idx)
        for idx in range(args.views_group):
            c_idx = args.dataset_shape[2]//args.views_group*idx
            scope = min(80, args.dataset_shape[2]//2)  # 80
            if c - c_idx >= scope:  # 去除最后一个视图
                c_idx_2 = c_idx + scope
                HSI2 = pca.fit_transform(data[:, c_idx:c_idx_2])
            else:
                data1 = data[:, c_idx:c]
                data2 = data[:, 0:scope-(c-c_idx)]
                HSI2 = pca.fit_transform(np.concatenate([data1,data2], axis=1))
            # HSI2 = pca.fit_transform(data[:, c_idx:c_idx_2])
            # if args.used_views:
            HSI2 = HSI2.reshape(h, w, view_idx)
            HSI2 = standardization_org(HSI2, remove_exception=True)
            # if "wiener" in args.checkpoint_path:
            HSI2 = np.array(HSI2, dtype=np.float)
            HSI2 = [wiener(HSI2[:, :, i], mysize=(3, 3)) for i in range(HSI2.shape[-1])]
            # 合并滤波后的各个通道
            HSI2 = np.stack(HSI2, axis=-1)
            # 归一化到 [0, 255] 范围
            HSI2 = (HSI2 / HSI2.max())*255
                # image2_crop = torch.tensor(image2_crop, dtype=torch.float)
            imgs.append(HSI2)

        img = np.concatenate(imgs, axis=2)
        for c_idx in range(0, img.shape[-1], in_channel):  # 将数据拆分为三通道

            if img.shape[-1] - c_idx <= in_channel:
                img2 = merge_images(img[:, :, c_idx:img.shape[-1]])
                out.append((np.array(add_zero(args, img2), dtype=np.uint8)))
            else:
                # out.append(img[:, :, c_idx:c_idx+3])
                out.append((np.array(add_zero(args, img[:, :, c_idx:c_idx + in_channel]), dtype=np.uint8)))

    if 0 in args.views:
        out_org = np.array(add_zero(args, data.reshape(h, w, c)), dtype=np.uint8)
        return out, out_org
    else:
        return out, []


def grid_search_svm(args, logger, params, result_train, label_train, number, repeat=1, class_weight=None):
    # params: a dictionary of SVM parameters to search
    # data: a pandas dataframe with features and labels
    # repeat: the number of times to repeat each parameter combination
    # number: the number of samples to draw from each class for training
    best_score = 0
    best_params = None
    result, label = result_train, label_train
    X_all, Y_all = aligned_sample(args, result, label, number=300000)
    X, Y = aligned_sample(args, result, label, number=number)
    for gamma in params['svc__gamma']:
        for C in params['svc__C']:
            svm = SVC(gamma=gamma, C=C, kernel='rbf', class_weight=class_weight)
            score = []
            score2 = []
            for i in range(repeat):
                svm.fit(X[i], Y[i])
                # predict on the test set and compute the accuracy score
                yfit = svm.predict(X_all)
                score.append(accuracy_score(Y_all, yfit))
                score2.append(balanced_accuracy_score(Y_all, yfit))
            score = np.array(score)
            score2 = np.array(score2)
            # update the best score and best parameters if necessary
            if score.mean() > best_score:
                best_score = score.mean()
                best_params = [gamma, C]
                best_svm = svm
            logger.info(f'gamma={gamma}, C={C},score:{score.mean()}, AA:{score2.mean()}')
        # return the best score and best parameters
    return best_score, best_params, best_svm
