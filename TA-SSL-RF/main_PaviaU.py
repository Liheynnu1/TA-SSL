from util import setup_logger, Splicing_result_paviaU, fix_random_seeds, aligned_sample
from Dataset.Dataset_swin_paviaU import mymodel, getimg, train_pca
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  # 分类效果报告模块
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import warnings
import os
import logging
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')



def train(args, result_train, label_train, number=2000):
    logger.info(f"开始训练！\n")
    svc = SVC(kernel='rbf', class_weight='balanced')  # kernel='rbf', class_weight='balanced'
    model = make_pipeline(StandardScaler(), svc)  # 打包管道
    # # 网格搜索：通过不断调整参数C，和参数gamma（控制径向基函数核的大小），确定最优模型
    param_grid = {'svc__C': [200],  # 50, 100, 150, 200, 250, 300
                  'svc__gamma': [0.0003]}  # gamma = 1/ 样本数  0.00003, 0.0003, 0.003, 0.03, 0.3
    grid = GridSearchCV(model, param_grid)
    result, label = result_train, label_train
    X, Y = aligned_sample(args, result, label, number=number)
    logger.info(f"训练集采样前{result.shape} {label.shape}")
    logger.info(f"训练集采样后{X.shape} {Y.shape}")

    grid.fit(X, Y)
    logger.info(f"{grid.best_params_}\n")
    # 最优参数落在了网格的中间位置。如果落在边缘位置，我们还需继续拓展网格搜索范围。接下来，我们可以对测试集的数据进行预测了
    return grid


def val(args, grid, result_val, label_val, number=1000):  # 对数据的各个类别分别进行采样from sklearn.preprocessing import StandardScaler

    result, label = result_val, label_val
    logger.info(f"开始评估！ {result.shape}  {label.shape}\n")
    X, Y = aligned_sample(args, result, label, number=number, ft_number=2550)  # 对每个类别分别进行2000次采样 ,并且打乱顺序
    X = np.array(X)
    for i in range(21):
        if i == 0:
            continue
        model = grid.best_estimator_  # 向支持向量机中加载最好的参数
        try:
            yfit = model.predict(X[Y == i, :])
            logger.info(f'类别{i}： 数量{len(X[Y==i,:])}; 准确率{accuracy_score(Y[Y==i], yfit)*100}%')
        except:
            pass
    # logger.info(f"评估集交叉验证: {cross_val_score(model, X[Y != 0, :], Y[Y != 0], cv=5, scoring='accuracy').mean()}\n")
    yfit = model.predict(X)
    logger.info(f"评估集尺寸{Y.shape}")
    logger.info(f'准确率{accuracy_score(Y, yfit)*100}%')
    logger.info(f"{classification_report(Y, yfit)}")
    mat = confusion_matrix(Y, yfit)
    sns.heatmap(mat.T, square=False, annot=False, fmt='d', cbar=True)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


def test(args, grid, result_test, label_test):
    model = grid.best_estimator_  # 向支持向量机中加载最好的参数
    result, label = result_test, label_test
    logger.info(f"开始测试！ {result[label!=0].shape}  {label[label!=0].shape}\n")
    n, h, w, c = result.shape  # 将标签reshape成(n*h*w,)的形状
    result2 = result[label != 0]
    yfit = []
    Y_val = label.reshape(n * h * w)
    length = result2.shape[0]
    for i in tqdm(range(0, length, 10000)):
        i_e = i+10000 if i+10000 <= result2.shape[0] else length
        X_val = result2[i:i_e, ...]  # .reshape(h*w, c)
        yfit.append(model.predict(X_val))  # 使用支持向量机进行预测
    yfit = np.concatenate(yfit)
    out = np.zeros_like(label.reshape(-1))
    out[label.reshape(-1) != 0] = yfit
    print(out.sum())
    # logger.info(f"支持向量机分类准确率： {accuracy_score(Y_val[Y_val!=0], yfit[Y_val!=0])*100}\n")
    logger.info(f"支持向量机分类准确率： {accuracy_score(Y_val[Y_val != 0], yfit) * 100}%\n")
    Splicing_result_paviaU(args, out.reshape(n * h * w), Y_val)  # 还原出结果图
    return yfit


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Backbone')
    parser.add_argument("--checkpoint_path", default='checkpoint/swin_400.pth', type=str)
    parser.add_argument("--model", default='swin', type=str, help="swin, intern")
    parser.add_argument("--out_path", default='out', type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--addfeature", default=True, type=bool)
    parser.add_argument("--seed", default=154, type=int)
    parser.add_argument("--n_components", default=[8, 16, 32, 64], type=int,
                        help="PCA 降维后的特征数，总特征数=4 x n_components x len(featureidx) + 39")
    parser.add_argument("--featureidx", default=[0, 3], type=int, help="PCA 起作用的层级")
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str)
    args = parser.parse_args()
    return args
args = parse_args()
fix_random_seeds(args.seed)
logger = logging.getLogger("train")
setup_logger("train", output=os.path.join(args.out_path, "log"), rank=0, log_level=logging.INFO)

models = []
pcas = train_pca(args, number=3)  # 20
models.append(mymodel(args, maxpool_10=False,  pcas=pcas).to(args.device))


logger.info(str(args).replace(",", "\n").replace("Namespace(", ""))

result_train, label_train = getimg(args, logger, loaders="test", number=6, models=models)
# result_val, label_val = getimg(args, logger, loaders="val", number=1, models=models, position_offset=1)
# result_test, label_test = getimg(args, logger, loaders="test", number=54*54*4, models=models, position_offset=0)

grid = train(args, result_train, label_train, number=20)  # 在均衡样本上进行训练
val(args, grid, result_train, label_train, number=2000)  # 在均衡的部分样本上测试
test(args, grid, result_train, label_train)  # 在全部数据上进行测试，输出结果图


