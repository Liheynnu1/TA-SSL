from util import setup_logger, Splicing_result, fix_random_seeds, aligned_sample, grid_search_svm
# from Dataset.Dataset_swin import mymodel, getimg, get_pca
from Dataset.Dataset_swin_paviaU2 import mymodel, getimg

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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# np.set_printoptions(suppress=True)

# class_weight = [1.941, 6.438, 0.135, 2.693, 0.995, 0.895, 0.053, 7.878, 44.32, 9.085,
#                 6.74, 0.301, 9.18, 1.954, 1.374, 2.278, 0.029, 1.297, 1.063, 1.352]
class_weight = [0.35, 1, 0.104, 0.163, 0.074, 0.278, 0.073, 0.204, 0.052]
# class_weight = np.ceil(np.array(class_weight))
class_weight = np.array(class_weight)
print(class_weight)
class_weight = class_weight / class_weight.max()
class_weight = {k + 1: v for k, v in enumerate(class_weight)}

print(class_weight)


# def train(args, result_train, label_train, number=2000):
#     logger.info(f"开始训练！\n")
#     svc = SVC(kernel='rbf', class_weight=class_weight)  # kernel='rbf', class_weight='balanced'
#     model = make_pipeline(StandardScaler(), svc)  # 打包管道
#     # # 网格搜索：通过不断调整参数C，和参数gamma（控制径向基函数核的大小），确定最优模型
#     # param_grid = {'svc__C':np.logspace(5, 12, 7, base=2),  # 50, 100, 150, 200, 250, 300
#     #               'svc__gamma': np.logspace(-8, -6, 6, base=2)}  # gamma = 1/ 样本数  0.00003, 0.0003, 0.003, 0.03, 0.3
#     param_grid = {'svc__C': [45, 64, 80],  # 50, 100, 150, 200, 250, 300
#                   'svc__gamma': [0.01]}  # gamma = 1/ 样本数  0.00003, 0.0003, 0.003, 0.03, 0.3
#     # 1, 0.3,0.1,0.03,0.01,0.003,0.001,0.0003
#     # grid = GridSearchCV(model, param_grid)
#     result, label = result_train, label_train
#     best_score, best_params, best_svm = grid_search_svm(args, logger, param_grid, result, label, number,
#                                                         repeat=10, class_weight=class_weight)
#     logger.info(f"训练集采样前{result.shape} {label.shape}")
#     # logger.info(f"训练集采样后{X.shape} {Y.shape}")
#     logger.info(f"best_params:{best_params}\n best_score:{best_score}")
#     # 最优参数落在了网格的中间位置。如果落在边缘位置，我们还需继续拓展网格搜索范围。接下来，我们可以对测试集的数据进行预测了
#     return {"best_params_": best_params, "best_estimator_": best_svm}

def train(args, X, Y, number=2000):
    # logger.info(f"开始训练！\n")
    svc = SVC(kernel='rbf', class_weight=args.dataset_class_weight)  # kernel='rbf'
    model = make_pipeline(StandardScaler(), svc)  # 打包管道
    # # 网格搜索：通过不断调整参数C，和参数gamma（控制径向基函数核的大小），确定最优模型
    param_grid = {'svc__C': np.logspace(3, 9, 7, base=2),  # 50, 100, 150, 200, 250, 300
                  'svc__gamma': np.logspace(-12, -4, 6, base=2)}  # gamma = 1/ 样本数  0.00003, 0.0003, 0.003, 0.03, 0.3
    # C是惩罚系数，理解为调节优化方向中两个指标（间隔大小，分类准确度）偏好的权重，即对误差的宽容度，
    # C越高，说明越不能容忍出现误差,容易过拟合，C越小，容易欠拟合，C过大或过小，泛化能力变差。
    # gamma参数定义了“单个训练样本对整个模型的影响程度”，gamma值很低表示“影响深远”，
    # gamma值高却表示“影响不大”。gamma参数可以看作是模型选出的那些支持向量的影响半径的倒数
    grid = GridSearchCV(model, param_grid,cv=3)

    # logger.info(f"训练集采样后{X.shape} {Y.shape}")
    grid.fit(X, Y)
    # logger.info(f"{grid.best_params_}\n")
    return grid

def val(args, grid, result_val, label_val, number=1000, seed=0):  # 对数据的各个类别分别进行采样from sklearn.preprocessing import StandardScaler
    score = []
    result, label = result_val, label_val
    # logger.info(f"开始评估！ {result.shape}  {label.shape}\n")
    X, Y = aligned_sample(args, result, label, number=number)  # 对每个类别分别进行2000次采样 ,并且打乱顺序
    number = []
    for i in range(21):
        if i == 0:
            continue
        # model = grid["best_estimator_"]  # 向支持向量机中加载最好的参数
        model = grid.best_estimator_  # 向支持向量机中加载最好的参数
        try:
            yfit = model.predict(X[Y == i, :])
            score1 = accuracy_score(Y[Y==i], yfit)*100
            # logger.info(f'类别{i}： 数量{len(X[Y==i,:])}; 准确率{score1}%')
            score.append(score1)
            number.append(len(X[Y==i,:]))

        except:
            pass
    number = np.array(number)
    # print(number/number.max())
    # logger.info(f"评估集交叉验证: {cross_val_score(model, X[Y != 0, :], Y[Y != 0], cv=5, scoring='accuracy').mean()}\n")
    yfit = model.predict(X)
    OA = accuracy_score(Y, yfit)*100
    AA = balanced_accuracy_score(Y, yfit)*100
    Kappa = cohen_kappa_score(Y, yfit)*100
    # logger.info(f"评估集尺寸{Y.shape}")
    logger.info(f'准确率{OA}%, seed={seed} ')
    # logger.info(f"{classification_report(Y, yfit)}")
    logger.info(f"text:OA={OA}, AA={AA}, Kappa={Kappa}")
    # mat = confusion_matrix(Y, yfit)
    # sns.heatmap(mat.T, square=False, annot=False, fmt='d', cbar=True)
    # plt.xlabel('true label')
    # plt.ylabel('predicted label')
    # plt.show()
    return [OA, AA, Kappa, *score]


def test(args, grid, result_test, label_test):
    # model = grid["best_estimator_"]  # 向支持向量机中加载最好的参数
    model = grid.best_estimator_  # 向支持向量机中加载最好的参数
    result, label = result_test, label_test
    logger.info(f"开始测试！ {result[label!=0].shape}  {label[label!=0].shape}\n")
    h, w = 610,340  # 将标签reshape成(n*h*w,)的形状
    result2 = result[label != 0]
    yfit = []
    Y_val = label.reshape(h, w)
    length = result2.shape[0]
    for i in tqdm(range(0, length, 10000)):
        i_e = i+10000 if i+10000 <= result2.shape[0] else length
        X_val = result2[i:i_e, ...]  # .reshape(h*w, c)
        yfit.append(model.predict(X_val))  # 使用支持向量机进行预测
    yfit = np.concatenate(yfit)
    out = np.zeros_like(label.reshape(-1))
    out[label.reshape(-1) != 0] = yfit
    # logger.info(f"支持向量机分类准确率： {accuracy_score(Y_val[Y_val!=0], yfit[Y_val!=0])*100}\n")
    OA = accuracy_score(Y_val[Y_val != 0], yfit)*100
    AA = balanced_accuracy_score(Y_val[Y_val != 0], yfit)*100
    Kappa = cohen_kappa_score(Y_val[Y_val != 0], yfit)*100
    print(f"text:OA={OA}, AA={AA}, Kappa={Kappa}")
    logger.info(f"支持向量机分类准确率： {OA}%\n")
    Splicing_result(args, out.reshape(h * w), Y_val)  # 还原出结果图
    return [OA, AA, Kappa]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Backbone')
    parser.add_argument("--checkpoint_path", default='checkpoint/swin_192_0-4.pth', type=str)
    parser.add_argument("--model", default='swin', type=str, help="swin, vit, SVM, t2t,DMVL")
    # parser.add_argument("--out_path", default='out', type=str)
    parser.add_argument("--batch_size", default=64, type=int, help="vit=128, swim=64")
    parser.add_argument("--in_channel", default=3, type=int, help="2 or 3")
    parser.add_argument("--repeat", default=10, type=int, help="实验重复次数")
    parser.add_argument("--image_size", default=224, type=int, help="切块的数量")
    parser.add_argument("--dataset", default='PaviaU', type=str,
                        help="使用的数据集：PaviaU， Salinas, Trento, Houston")
    parser.add_argument("--embed_dim", default=192, type=int, help="网络输出的特征维度128,192,384")
    parser.add_argument("--depths", default=[2, 2, 6, 2], type=list, help="网络输出的特征维度128,192,384")
    parser.add_argument("--num_heads", default=[3, 3, 6, 12], type=list, help="[4, 8, 16, 32]网络输出的特征维度128,192,384")
    parser.add_argument("--addfeature", default=True, type=bool)
    parser.add_argument("--addspectral", default=False, type=bool)
    parser.add_argument("--piece_size", default=10, type=int, help="切块的大小")
    parser.add_argument("--stride", default=1, type=int, help="原始图像缩放倍数")
    parser.add_argument("--seed", default=20, type=int)
    parser.add_argument("--views", default=[0, 2], type=int, help="将要使用的视图，单独0为原始图像")
    parser.add_argument("--views_group", default=3, type=int, help="将波段分为n组，c/n个波段压缩为一个视图")
    parser.add_argument("--sample_number", default=5, type=int, help="每个类别抽取的样本数量")
    parser.add_argument("--device", default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument("--user_feature", default=False , type=str, help="是否使用加载好了的特征")
    args = parser.parse_args()
    return args







# 定义一个函数，利用特征降维方法对特征分离度进行可视化
def visualize_feature_separation(args, features, labels, n_components=2, perplexity=30, random_state=10):
    """
    Args:
        features: 一个二维数组，表示一万个数据每个数据n个特征。
        labels: 一个一维数组或列表，表示每个数据对应的标签。
        n_components: 一个整数，表示降维后的目标维度，默认为2。
        perplexity: 一个浮点数，表示t-SNE算法中的困惑度参数，默认为30。
        random_state: 一个整数或None，表示随机数生成器的种子，默认为0。
    Returns:
        None
    """

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    features_tsne = tsne.fit_transform(features)
    print('aaa')
    # 将降维后的特征和标签合并为一个DataFrame
    df_tsne = pd.DataFrame(np.hstack((features_tsne, labels.reshape(-1, 1))), columns=['x', 'y', 'label'])
    # 绘制散点图并显示
    plt.figure(figsize=(6, 6))
    plt.title(args.info)

    # 设置颜色映射和图例
    cmap = plt.get_cmap('tab20', labels.max())  # 尝试使用不同的颜色映射
    norm = plt.Normalize(labels.min(), labels.max())  # 手动设置颜色映射范围
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 绘制t-SNE结果
    plt.scatter(df_tsne['x'], df_tsne['y'], s=10, c=df_tsne['label'], cmap=cmap, norm=norm)  # 绘制散点图，使用颜色映射和范围
    cbar = plt.colorbar(sm, ax=plt.gca())
    # 设置颜色条的位置
    # cbar.ax.set_position([1.05, 0.1, 0.05, 0.8])  # 调整位置参数根据需要
    plt.xticks([])  # 去掉x轴的刻度线
    plt.yticks([])  # 去掉y轴的刻度线
    plt.savefig(f'save/{args.info}.png')
    plt.show()  # 显示图形


def ones(result_test, label_test):

    score = []
    beest_OA = 0
    seed_list = args.seed_list
    for seed in seed_list:
        fix_random_seeds(seed)
        X, Y = aligned_sample(args, result_test, label_test, number=args.sample_number, repeat=None, seed=seed)
        grid = train(args, X, Y)  # 在均衡样本上进行训练
        score_one = val(args, grid, result_test, label_test, number=400000, seed=seed)
        score.append(score_one)  # 在均衡的部分样本上测试
        if score_one[0] >= beest_OA:
            beest_OA = score_one[0]
            # test(args, grid, result_test, label_test)  # 输出最好的结果图

    score = np.array(score)
    a = list(score[:, 0])
    sorted_indices = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:10]
    # print(sorted_indices)
    # print(np.array(a)[sorted_indices])
    logger.info(f"AA_mean:{score[sorted_indices,:3].mean(0)}")
    logger.info(f"AA_max:{score[sorted_indices,:3].max(0)}")
    logger.info(f"AA_std:{score[sorted_indices,:3].std(0)}")
    logger.info(f"class_score:{score[sorted_indices,3:].mean(0)}")

if __name__ == '__main__':

    args = parse_args()
    logger = logging.getLogger("train")
    if args.dataset == 'Salinas':
        args.dataset_img = 'salinas_corrected'
        args.dataset_label = 'salinas_gt'
        args.dataset_shape = [512, 217, 204]
        class_weight = [0.18, 0.33, 0.18, 0.12, 0.24, 0.35, 0.32, 1., 0.55, 0.29, 0.09, 0.17, 0.08, 0.09,  0.64, 0.16]
        args.seed_list = [i for i in range(args.repeat * 5)]
        # args.seed_list = [26, 13, 1, 40, 29, 10, 12, 31, 7, 27, 25, 46, 34, 8, 38, 16, 20, 21, 23, 37]
    elif args.dataset == 'PaviaU':
        args.dataset_img = 'paviaU'
        args.dataset_label = 'paviaU_gt'
        args.dataset_shape = [610, 340, 103]
        class_weight = [0.35, 1, 0.104, 0.163, 0.074, 0.278, 0.073, 0.204, 0.052]
        args.seed_list = [i for i in range(args.repeat * 5)]
        # args.seed_list = [445, 406, 352, 90, 265, 243, 256, 136, 168, 293, 13, 34, 42, 40, 21, 7, 33, 15, 43, 17]
    elif args.dataset == 'Trento':
        args.dataset_img = 'HSI'
        args.dataset_label = 'gt'
        args.dataset_shape = [166, 600, 63]
        class_weight = [0.38, 0.28, 0.05, 0.87, 1., 0.30]
        args.seed_list = [i for i in range(args.repeat * 5)]
        # args.seed_list = [44, 20, 1, 28, 16, 12, 36, 35, 4, 33, 7, 5, 10, 19, 34, 41, 42, 9, 29, 27]
    else:
        args.dataset_img = 'img'
        args.dataset_label = 'labels'
        args.dataset_shape = [349, 1905, 144]
        class_weight = [0.36, 1., 0.11, 0.16, 0.07, 0.27, 0.07,  0.2, 0.05]
        args.seed_list = [i for i in range(args.repeat * 5)]
        # args.seed_list = [44, 20, 1, 28, 16, 12, 36, 35, 4, 33, 7, 5, 10, 19, 34, 41, 42, 9, 29, 27]
    class_weight = np.array(class_weight)
    class_weight = class_weight / class_weight.max()
    args.dataset_class_weight = {k + 1: v for k, v in enumerate(class_weight)}
    args.views_number = args.dataset_shape[2] // args.views_group
    fix_random_seeds(args.seed)
    for checkpoint_path in ['checkpoint2/unified.pth',
                            'checkpoint2/CMID.pth',
                            'checkpoint2/Distillation_plus_meanadd.pth',
                            'checkpoint2/Distillation_plus_meanadd_avg.pth',
                            "checkpoint/Distillation_plus_zeroreplace.pth",
                            "checkpoint2/MIM.pth"
                            ]:
        args.checkpoint_path = checkpoint_path
        model = mymodel(args, maxpool_10=False).to(args.device)

        logger.info(str(args).replace(",", "\n").replace("Namespace(", ""))
        # 33 6
        result_test, label_test, labels_org = getimg(args, logger, loaders="test", number=1000, models=model, position_offset=0)
        args.info = f"{checkpoint_path[11:-4]}"
        file_path = rf'save/{args.info}.h5'
        if not args.user_feature:
            result_test, label_test, labels_org = getimg(args, logger, loaders="test", number=1000, models=model,
                                                         position_offset=0)
            print("保存数据")
            if not os.path.exists(file_path):
                # 创建文件
                with open(file_path, 'w') as f:
                    pass
                print(f"File '{file_path}' created.")
                f.close()
            f = h5py.File(file_path, 'w')
            f['data'] = result_test
            f['label'] = label_test
            f['labels_org'] = labels_org
            f.close()
            print("保存完成")
        else:
            print("加载数据")
            f = h5py.File(file_path, 'r')
            result_test = f['data'][:]
            label_test = f['label'][:]
            labels_org = f['labels_org'][:]
            f.close()
        visualize_feature_separation(args, result_test[label_test != 0], label_test[label_test != 0])

