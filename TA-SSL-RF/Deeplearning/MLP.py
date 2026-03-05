from PIL import Image
import tifffile as tf
from torch.utils.data import DataLoader, Dataset
from util import setup_logger, Splicing_result, fix_random_seeds, aligned_sample
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
import hiddenlayer as hl
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from osgeo import gdal, ogr, gdalconst
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import os
import logging
import argparse
from Model import myMLP, MyDataSet, getimg
import tifffile as tf
from sklearn.ensemble import RandomForestClassifier

Image.MAX_IMAGE_PIXELS = None
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def parse_args():
    parser = argparse.ArgumentParser(description='Convert Backbone')
    parser.add_argument("--model", default='swin', type=str, help="swin, vit, SVM, t2t,DMVL, resnet，Dinov2,simMIM")
    parser.add_argument("--arch", default='t', type=str)
    parser.add_argument("--batch_size", default=4000, type=int, help="vit=64, swim=64")
    parser.add_argument("--in_channel", default=3, type=int, help="2 or 3")
    parser.add_argument("--stip", default=32, type=int, help="32 or 16")
    parser.add_argument("--dim", default=768, type=int, help="768 or 1024")
    parser.add_argument("--repeat", default=10, type=int, help="实验重复次数")
    parser.add_argument("--image_size", default=224, type=int, help="切块的数量")
    parser.add_argument("--dataset", default='PaviaU', type=str,
                        help="使用的数据集：PaviaU， Salinas, Trento, Houston")
    parser.add_argument("--addfeature", default=False, type=bool, help="是否加入空间碎片特征")
    parser.add_argument("--addspectral", default=True, type=bool, help="是否加入光谱特征")
    parser.add_argument("--piece_size", default=10, type=int, help="切块的大小")
    parser.add_argument("--stride", default=2, type=int, help="原始图像缩放倍数")
    parser.add_argument("--seed", default=20, type=int)
    parser.add_argument("--Complete_feature", default=False, type=bool, help="是否使用完整特征图")
    parser.add_argument("--views", default=[0, 3], type=int, help="将要使用的视图，单独0为原始图像")
    parser.add_argument("--views_group", default=2, type=int, help="将波段分为n组，c/n个波段压缩为一个视图")
    parser.add_argument("--sample_number", default=5, type=int, help="每个类别抽取的样本数量")
    parser.add_argument("--device", default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument("--user_feature", default=False, type=str, help="是否使用加载好了的特征")
    args = parser.parse_args()
    return args
def train_model(model, train_loader, test_loader, optimizer, loss_func, device, epochs=50):
    # 使用Canvas可视化
    historyT = hl.History()
    canvasT = hl.Canvas()
    # 记录训练过程的指标
    print_step = 1
    history1 = hl.History()
    canvas1 = hl.Canvas()
    best_accuracy = 0.0
    correct = 0
    for epoch in range(epochs):
        model.train()
        for step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            output = model.forward(data)  # 前向传播
            # pred1 = output.argmax(dim=1, keepdim=True)
            train_loss = loss_func(output, target)  # 计算损失
            optimizer.zero_grad()  # 梯度初始化为0
            train_loss.backward()  # 反向传播
            optimizer.step()  # 使用梯度进行优化
            niter = epoch * len(train_loader) + step + 1
            # if niter % print_step == 0:
            # 验证精度
            model.eval()
            correct = 0
            with torch.no_grad():
                for datav, targetv in test_loader:
                    datav, targetv = datav.to(device), targetv.to(device)
                    outputV = model.forward(datav)
                    pred = outputV.argmax(dim=1, keepdim=True)
                    correct += pred.eq(targetv.view_as(pred)).sum().item()
                    test_accuracy = 100. * correct / len(test_loader.dataset)
                    # 添加epoch，损失和精度
                    history1.log(niter, train_loss=train_loss, test_accuracy=test_accuracy)
                    # 使用两个图像可视化损失函数和精度
                    # with canvas1:
                    #     canvas1.draw_plot(history1['train_loss'])
                    #     canvas1.draw_plot(history1['test_accuracy'])
                    # 保存最佳模型的权重
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        torch.save(testnet.state_dict(), best_model_path)
                        print(f"Saved better model with accuracy: {test_accuracy}")
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, test_accuracy: {test_accuracy:.2f}%')
if __name__ == "__main__":
    args = parse_args()
    args.out_path = f'out/{args.dataset}'
    logger = logging.getLogger("train")
    setup_logger("train", output=os.path.join(args.out_path, "log"), rank=0, log_level=logging.INFO)
    fix_random_seeds(args.seed)
    # model = mymodel(args, maxpool_10=False, pcas=None).to(args.device)
    logger.info(str(args).replace(",", "\n").replace("Namespace(", ""))
    # 训练样本
    txt_train = r'G:\trainsampleImg\featureMergeall\Trainingpath.txt'
    # 验证样本
    txt_val = r'G:\ValidatesampleImg\featureMergeall\Validatpath.txt'
    # 数据读取args, logger, txt_train, data_path="train", position_offset=0
    train_nots_loader = getimg(args, logger, txt_train, dataset_type="train", position_offset=0)
    test_nots_loader = getimg(args, logger, txt_val, dataset_type="val", position_offset=0)

    # 初始化网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testnet = myMLP().to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(testnet.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    # 记录训练过程的指标
    history1 = hl.History()
    # 使用Canvas可视化
    canvas1 = hl.Canvas()
    best_accuracy = 0.0
    best_model_path = "best_MLP_featurelearn001.pth"
    # 训练特征提取算法
    train_model(testnet, train_nots_loader, test_nots_loader, optimizer, loss_func, device, epochs=50)
    # validate(testnet, device, test_nots_loader, loss_func)
    # 保存模型权重
    # 训练分类模型
    model = myMLP().to(device)
    model.load_state_dict(torch.load(best_model_path))
    input_data = torch.from_numpy(X_train_s).float().to(device)
    Xfeatures = model.extract_features(input_data)
    Xfeatures = Xfeatures.cpu().detach()
    # 加载数据集
    args = parse_args()
    logger = logging.getLogger("train")
    # 获取图像
    traindata,label = getimg(args, logger, data_path=txt_train, position_offset=0)
    ds = MyDataSet(txt_train, dataset_type='train')  # Initialize the dataset
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

    for data in tqdm(loader):
        image1 = []
        image = data["image"].permute(0, 3, 1, 2)
        image = image.to(args.device)
        # image = model1(image).detach().to('cpu')
        image = torch.tensor(image)
    # result_train, label_train = getimg(args, logger, data_path=txt_train, models= None, position_offset=0)
    trainimg, trainlabel = MyDataSet(txt_train)
    args = parse_args()
    args.out_path = f'out/{args.dataset}'