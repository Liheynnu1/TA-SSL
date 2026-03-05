import pandas as pd
import numpy as np


def aligned_sample(args, result, label, number=2000, repeat=10, seed=0):  # 对数据的各个类别分别进行采样
    c = result.shape[-1]
    np.random.seed(args.seed)
    # 把标签数据转换成一维数组，方便后续的操作
    label_data = label.reshape(-1)
    # 把标签数据转换成pandas的Series对象，方便使用groupby方法
    label_series = pd.Series(label_data)
    # 对标签数据按照类别进行分组，得到一个GroupBy对象
    label_groups = label_series.groupby(label_series)
    # 创建一个空的列表，用来存放每个类别抽样后的索引


    imgout, labelout = [], []
    # 遍历每个分组，对每个类别进行随机抽样
    # for i in range(repeat):
    sample_indice = []
    for name, group in label_groups:
        if name == 0:
            continue
        # 获取该类别的样本数量
        group_size = len(group)
        # print(f'label_{name}_sum : {group_size}')
        # 如果样本数量大于2000，就随机抽取2000个索引
        if group_size > number:
                sample_indice.extend(group.sample(number, random_state=seed).index.tolist())
        # 如果样本数量小于等于2000，就取出所有的索引
        else:
            sample_indice.extend(group.index.tolist())
    # np.random.shuffle(sample_indice)  # 将元素打乱
    # 把索引列表转换成numpy数组，方便后续的操作
    sample_indices = np.array(sample_indice)
    # # 根据索引，从图像数据中取出相应的元素，得到抽样后的图像数据
    result = result.reshape(-1, c)
    sample_image_data = result[sample_indices, :]
    # 根据索引，从标签数据中取出相应的元素，得到抽样后的标签数据
    sample_label_data = label_data[sample_indices]
    if repeat==None:
        return sample_image_data, sample_label_data
    imgout.append(sample_image_data)
    labelout.append(sample_label_data)
    if number>500:
        return imgout[0], labelout[0]
    # return sample_image_data[sample_label_data != 0, :], sample_label_data[sample_label_data != 0]
    return imgout, labelout
