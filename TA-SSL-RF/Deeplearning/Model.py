import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import tifffile as tf
from torch.utils.data import DataLoader, Dataset
from util import setup_logger, Splicing_result, fix_random_seeds, aligned_sample
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  # 分类效果报告模块
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import argparse
import numpy as np
import warnings
import os
import logging
import torch
import tifffile as tf
class myMLP(nn.Module):
    def __init__(self):
        super(myMLP, self).__init__()
        # 简化网络结构
        self.feature = nn.Sequential(
            # 隐藏层1
            nn.Linear(in_features=102, out_features=500, bias=True),
            nn.ReLU(),
            # 隐藏层2
            nn.Linear(500, 500),
            nn.Tanh(),
            # 隐藏层3
            nn.Linear(500, 400),
            nn.Dropout(0.2),
            # 隐藏层4
            nn.Linear(400, 400),
            nn.Tanh(),
            # 隐藏层5
            nn.Linear(400, 400),
            nn.Tanh(),
            # 隐藏层6
            nn.Linear(400, 360),
            nn.Dropout(0.1),
            # 隐藏层7
            nn.Linear(360, 200),
            nn.ReLU(),
            # 隐藏层8
            nn.Linear(200, 200),
            nn.Tanh(),
            # 隐藏层9
            nn.Linear(200, 180),
            nn.Dropout(0.1),
            # 隐藏层10
            nn.Linear(180, 100),
            nn.ReLU(),
            # 隐藏层11
            nn.Linear(100, 100),
            nn.Tanh(),
            # 隐藏层12
            nn.Linear(100, 90),
            nn.Dropout(0.1),
            # 隐藏层13
            nn.Linear(90, 50),
            nn.ReLU(),
            # 隐藏层14
            nn.Linear(50, 50),
            nn.Tanh(),
            # 隐藏层15
            nn.Linear(50, 45),
            nn.Dropout(0.1)
        )
        # 分类
        self.classify = nn.Sequential(
            nn.Linear(45, 10),  # 假设有4个类别
            # nn.Softmax(dim=1)  # 使用Softmax作为激活函数
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.feature(x)
        output = self.classify(x)
        return output
    def extract_features(self, x):
        x = self.feature(x)
        return x
class MyDataSet(Dataset):
    def __init__(self, make_txt_file, dataset_type, transform=None, update_dataset=False):
        self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(make_txt_file)
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()
    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self, index):
        item = self.sample_list[index]
        tifpath = item.split(' _')[0]
        img = tf.imread(item.split(' _')[0])
        # img = Image.open(item.split(' _')[0])  # Use Image.open to read images
        if self.transform is not None:
            img = self.transform(img)
        label = int(item.split(' _')[-1])
        # 设置一个跟图像大小相一致的标签图像
        label1 = np.full((img.shape[0], img.shape[1]), label)
        return {'image': img, 'label': label1}
def getimg(args, logger, txt_train, dataset_type="train", position_offset=0):
    """
    Args:
        args: 其他参数
        loaders: 加哉哪种数据-train、val、test
        number: 加载数据的数量（0 - 54X54X4）
        position_offset: 加载数据从何处开始

    Returns: data、label
    """
    ds = MyDataSet(txt_train, dataset_type="train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    data1 = tqdm(loader)
    image = []
    label = []
    for data in tqdm(loader):
        image1 = data["image"]
        # image1 = image1.to(args.device)
        image1 = torch.tensor(image1)
        image.append(image1)
        label.append(data["label"])
    result = image
    label1 = label
    torch.cuda.empty_cache()
    return result, label1