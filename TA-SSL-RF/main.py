from util import setup_logger, Splicing_result, fix_random_seeds, aligned_sample, convert_unit
# from Dataset.Dataset_swin import mymodel, getimg, get_pca
from Dataset.Dataset_swin_paviaU2 import mymodel, getimg
from thop import profile
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
CUDA_LAUNCH_BLOCKING=1

def train(args, X, Y, number=2000):
    # logger.info(f"开始训练！\n")
    svc = SVC(kernel='rbf')  # kernel='rbf' , class_weight=args.dataset_class_weight
    model = make_pipeline(StandardScaler(), svc)  # 打包管道
    # # 网格搜索：通过不断调整参数C，和参数gamma（控制径向基函数核的大小），确定最优模型
    param_grid = {'svc__C': np.logspace(3, 9, 7, base=2),  # 50, 100, 150, 200, 250, 300
                  'svc__gamma': np.logspace(-12, -4, 6, base=2)}  # gamma = 1/ 样本数  0.00003, 0.0003, 0.003, 0.03, 0.3
    # C是惩罚系数，理解为调节优化方向中两个指标（间隔大小，分类准确度）偏好的权重，即对误差的宽容度，
    # C越高，说明越不能容忍出现误差,容易过拟合，C越小，容易欠拟合，C过大或过小，泛化能力变差。
    # gamma参数定义了“单个训练样本对整个模型的影响程度”，gamma值很低表示“影响深远”，
    # gamma值高却表示“影响不大”。gamma参数可以看作是模型选出的那些支持向量的影响半径的倒数
    grid = GridSearchCV(model, param_grid, cv=3 if args.sample_number < 5 else 5)

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
    # logger.info(f'准确率{OA}%, seed={seed} ')
    # logger.info(f"{classification_report(Y, yfit)}")
    # logger.info(f"text:OA={OA}, AA={AA}, Kappa={Kappa}")
    # mat = confusion_matrix(Y, yfit)
    # sns.heatmap(mat.T, square=False, annot=False, fmt='d', cbar=True)
    # plt.xlabel('true label')
    # plt.ylabel('predicted label')
    # plt.show()
    return [OA, AA, Kappa, *score]


def test(args, grid, result_test, label_test):
    # model = grid["best_estimator_"]  # 向支持向量机中加载最好的参数
    model = grid.best_estimator_  # 向支持向量机中加载最好的参数
    result2, label = result_test, label_test
    # logger.info(f"开始测试！ {result[label!=0].shape}  {label[label!=0].shape}\n")
    h, w = args.dataset_shape[0], args.dataset_shape[1]  # 将标签reshape成(n*h*w,)的形状
    # result2 = result[label != 0]
    yfit = []
    Y_val = label.reshape(h, w)
    length = result2.shape[0]
    for i in range(0, length, 10000):
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
    # print(f"text:OA={OA}, AA={AA}, Kappa={Kappa}")
    # logger.info(f"支持向量机分类准确率： {OA}%\n")
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
    parser.add_argument("--num_heads", default=[3, 6, 12, 24], type=list, help="[3, 3, 6, 12][3, 6, 12, 24]网络输出的特征维度128,192,384")
    parser.add_argument("--addfeature", default=True, type=bool)
    parser.add_argument("--addspectral", default=False, type=bool)
    parser.add_argument("--piece_size", default=10, type=int, help="切块的大小")
    parser.add_argument("--stride", default=1, type=int, help="原始图像缩放倍数")
    parser.add_argument("--seed", default=20, type=int)
    parser.add_argument("--views", default=[0, 3], type=int, help="将要使用的视图，单独0为原始图像")
    parser.add_argument("--views_group", default=2, type=int, help="将波段分为n组，c/n个波段压缩为一个视图")
    parser.add_argument("--sample_number", default=5, type=int, help="每个类别抽取的样本数量")
    parser.add_argument("--device", default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument("--user_feature", default=False , type=str, help="是否使用加载好了的特征")
    args = parser.parse_args()
    return args




def save_feature(args):
    if args.dataset == 'Salinas':
        args.dataset_img = 'salinas_corrected'
        args.dataset_label = 'salinas_gt'
        args.dataset_shape = [512, 217, 204]
        class_weight = [0.18, 0.33, 0.18, 0.12, 0.24, 0.35, 0.32, 1., 0.55, 0.29, 0.09, 0.17, 0.08, 0.09,  0.64, 0.16]
        args.seed_list = [i for i in range(args.repeat * 5)]
    elif args.dataset == 'PaviaU':
        args.dataset_img = 'paviaU'
        args.dataset_label = 'paviaU_gt'
        args.dataset_shape = [610, 340, 103]
        class_weight = [0.35, 1, 0.104, 0.163, 0.074, 0.278, 0.073, 0.204, 0.052]
        args.seed_list = [i for i in range(args.repeat * 5)]
    elif args.dataset == 'Trento':
        args.dataset_img = 'HSI'
        args.dataset_label = 'gt'
        args.dataset_shape = [166, 600, 63]
        class_weight = [0.38, 0.28, 0.05, 0.87, 1., 0.30]
        args.seed_list = [i for i in range(args.repeat * 5)]
    elif args.dataset == 'Indian':
        args.dataset_img = 'indian_pines_corrected'
        args.dataset_label = 'indian_pines_gt'
        args.dataset_shape = [145, 145, 200]
        class_weight = [0.02, 0.58, 0.34, 0.10, 0.20, 0.30, 0.01, 0.19, 0.01, 0.40, 1., 0.24, 0.08, 0.52, 0.16, 0.04]
        args.seed_list = [i for i in range(args.repeat * 5)]
    else:
        args.dataset_img = 'img'
        args.dataset_label = 'labels'
        args.dataset_shape = [349, 1905, 144]
        class_weight = [0.36, 1., 0.11, 0.16, 0.07, 0.27, 0.07,  0.2, 0.05]
        args.seed_list = [i for i in range(args.repeat * 5)]
    class_weight = np.array(class_weight)
    class_weight = class_weight / class_weight.max()
    args.dataset_class_weight = {k + 1: v for k, v in enumerate(class_weight)}
    args.views_number = args.dataset_shape[2] // args.views_group

    model = mymodel(args, maxpool_10=False, pcas=None).to(args.device)
    file_path = rf'data/{args.dataset}_feature_{args.views_group}_{args.views[-1]}.h5'
    if not args.user_feature:
        result_test, label_test, labels_org = getimg(args, logger, loaders="test", number=1000, models=model, position_offset=0)
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
    return result_test, label_test, labels_org


def ones(result_test, label_test, labels_org):

    score = []
    beest_OA = 0
    idx = 50 if args.sample_number <= 20 else 10
    seed_list = args.seed_list[:idx]
    for seed in seed_list:
        fix_random_seeds(seed)
        X, Y = aligned_sample(args, result_test, label_test, number=args.sample_number, repeat=None, seed=seed)
        grid = train(args, X, Y)  # 在均衡样本上进行训练
        score_one = val(args, grid, result_test, label_test, number=400000, seed=seed)
        score.append(score_one)  # 在均衡的部分样本上测试
        if score_one[0] >= beest_OA:
            beest_OA = score_one[0]
            test(args, grid, result_test, labels_org)  # 输出最好的结果图

    score = np.array(score)
    a = list(score[:, 0])
    sorted_indices = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:idx]
    logger.info(sorted_indices)
    # print(np.array(a)[sorted_indices])
    logger.info(f"AA_mean:{score[sorted_indices,:3].mean(0)}")
    logger.info(f"AA_max:{score[sorted_indices,:3].max(0)}")
    logger.info(f"AA_std:{score[sorted_indices,:3].std(0)}")
    # logger.info(f"{score[sorted_indices,0]}")
    logger.info(f"class_score:{score[sorted_indices,3:].mean(0)}")

# result_test, label_test = save_feature(args)
# ones(result_test, label_test)
def best_args(dataset,):
    if dataset == 'PaviaU':
        args.dataset = 'PaviaU'
        args.checkpoint_path = 'checkpoint/swin_192_0-4.pth'
        args.views[1] = 4
        args.views_group = 2
        args.piece_size = 9
    elif dataset == "Salinas":
        args.dataset = 'Salinas'
        args.checkpoint_path = 'checkpoint/swin_192_0-4.pth'
        args.views[1] = 4
        args.views_group = 4
        args.piece_size = 10
    elif dataset == "Trento":
        args.dataset = 'Trento'
        args.checkpoint_path = 'checkpoint/swin_192_0-5.pth'
        args.views[1] = 3
        args.views_group = 2
        args.piece_size = 12
    elif dataset == "Houston":
        args.dataset = 'Houston'
        args.checkpoint_path = 'checkpoint/swin_192_0-4.pth'
        args.views[1] = 4
        args.views_group = 3
        args.piece_size = 10
# def best_args(dataset,):
#     if dataset == 'PaviaU':
#         args.dataset = 'PaviaU'
#         args.checkpoint_path = 'checkpoint/vit.pth'
#         args.views[1] = 2
#         args.views_group = 2
#         args.piece_size = 10
#     elif dataset == "Salinas":
#         args.dataset = 'Salinas'
#         args.checkpoint_path = 'checkpoint/vit.pth'
#         args.views[1] = 2
#         args.views_group = 2
#         args.piece_size = 10
#     elif dataset == "Trento":
#         args.dataset = 'Trento'
#         args.checkpoint_path = 'checkpoint/vit.pth'
#         args.views[1] = 3
#         args.views_group = 2
#         args.piece_size = 10
#     elif dataset == "Houston":
#         args.dataset = 'Houston'
#         args.checkpoint_path = 'checkpoint/vit.pth'
#         args.views[1] = 2
#         args.views_group = 2
#         args.piece_size = 10

# for dataset in ['PaviaU', 'Salinas', 'Trento', 'Houston']:
#     for group_number in [2, 3, 4]:
#         for view in [2, 3, 4, 5]:
#             args.views[1] = view
#             args.views_group = group_number
#             best_args(dataset)
#             result_test, label_test = save_feature(args)
#             logger.info(f"dataset={dataset}, views_group={group_number}, views={view}")
#             ones(result_test, label_test)
#
#


# for dataset in ['PaviaU', 'Salinas', 'Trento', 'Houston']:
#     for piece_size in [6, 10, 14, 18, 22, 26]:
#         args.piece_size = piece_size
#         args.dataset = dataset
#         best_args(dataset)
#         result_test, label_test, labels_org = save_feature(args)
#         logger.info(f"dataset={dataset}, piece_size={piece_size}")
#         ones(result_test, label_test, labels_org)

if __name__ == '__main__':
    args = parse_args()
    args.out_path = f'out/{args.dataset}'
    logger = logging.getLogger("train")
    setup_logger("train", output=os.path.join(args.out_path, "log"), rank=0, log_level=logging.INFO)

    fix_random_seeds(args.seed)

    # model = mymodel(args, maxpool_10=False, pcas=None).to(args.device)
    logger.info(str(args).replace(",", "\n").replace("Namespace(", ""))
    # input = torch.randn(1, 3, 224, 224).to(args.device)
    # flops, params = profile(model, inputs=(input,), verbose=False)
    # # 调用函数，将FLOPs和Params转换成不同的单位，并打印结果
    # flops_unit, params_unit = convert_unit(flops, params)
    # logger.info(f'model: {args.model} ***#*** FLOPs: {flops_unit}, Params: {params_unit} ***#***')

    for dataset in ['Indian', 'PaviaU', 'Salinas', 'Trento', 'Houston']:
        for checkpoint_path in [
                                'checkpoint/wiener.pth',
                                'checkpoint/MIM_plus_loss_all.pth',
                                # 'checkpoint/swin_192_0-4.pth',
                                # 'checkpoint/Distillation_plus_meanadd.pth',
                                # 'checkpoint/CL_plus_alignment.pth',
                                # 'checkpoint/CL_plus_view_CL.pth',
                                # 'checkpoint/MIM_plus_zero_mask.pth',
                                # 'checkpoint/alignment_and_view.pth',
                                # 'checkpoint/alignment_inversion_view_zero.pth',
                                # 'checkpoint/stage2.pth',

                                ]:
            # best_args(dataset)
            args.dataset = dataset
            # if 'unified' in checkpoint_path:
            #     args.depths = [2, 2, 18, 2]
            #     args.num_heads = [3, 6, 12, 24]
            #     args.embed_dim = 192
            # else:
            #     args.depths = [2, 2, 6, 2]
            #     args.num_heads = [3, 3, 6, 12]
            #     args.embed_dim = 192
            args.checkpoint_path = checkpoint_path
            result_test, label_test, labels_org = save_feature(args)
            args.info = f"{dataset}_{checkpoint_path[11:-4]}"
            logger.info(args.info)
            ones(result_test, label_test, labels_org)

    # for dataset in ['PaviaU', 'Salinas', 'Trento', 'Houston']:
    #     for a in range(1,5):
    #         best_args(dataset)
    #         args.dataset = dataset
    #         if a==1:
    #             args.addfeature = True
    #             args.addspectral = True
    #         elif a==2:
    #             args.addfeature = True
    #             args.addspectral = False
    #         elif a==3:
    #             args.addfeature = False
    #             args.addspectral = True
    #         elif a==4:
    #             args.addfeature = False
    #             args.addspectral = False
    #         result_test, label_test, labels_org = save_feature(args)
    #         logger.info(f"addfeature={args.addfeature}, addspectral={args.addspectral}")
    #         ones(result_test, label_test, labels_org)
    # for dataset in ['PaviaU', 'Salinas', 'Trento', 'Houston']:
    #     best_args(dataset)
    #     # args.checkpoint_path = 'checkpoint/DMVL.pth'
    #     result_test, label_test, labels_org = save_feature(args)
    #     for sample_number in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     # for sample_number in [80,90,100]:
    #         args.sample_number = sample_number
    #         logger.info(f"dataset={dataset}, sample_number={sample_number}")
    #         ones(result_test, label_test, labels_org)
