import torch
import os
import torch.nn as nn
import copy
import numpy as np
import logging
import scipy.io as scio
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
# from models import InternImage
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from mmseg.models.backbones.swin import SwinTransformer

Image.MAX_IMAGE_PIXELS = None  # or a larger value than the default
transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((64, 64)),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomEqualize(),
    transforms.ToTensor(),
])

# 512,217,204
class Dataset(Dataset):
    def __init__(self,
                 image1_paths=r'D:\gw\GW-main\SVM\data\dataset\Salinas_corrected.mat',
                 label_paths=r'D:\gw\GW-main\SVM\data\dataset\Salinas_gt.mat',
                 transform=transform,
                 len=16,
                 loaders="",
                 position_offset=0,
                 **kwargs):
        self.len = len
        self.loaders = loaders
        self.transform = transform
        self.position_offset = position_offset
        self.all_index = np.arange(3)
        np.random.shuffle(self.all_index)
        self.image1s = scio.loadmat(image1_paths)['salinas_corrected']
        self.labels = scio.loadmat(label_paths)['salinas_gt']
        assert self.len <= 3
    def __getitem__(self, index):
        if self.loaders == 'val':  # 如果是测试数据就跳过加载过的加载数据
            index += self.position_offset
            assert index < 3  # 加载的数据过多

        if self.loaders != 'test':
            index = self.all_index[index]

        img = np.zeros((224, 224, 204))
        lable = np.zeros((224, 224))
        c = 0
        for i in range(0, 204, 3):
            column = index % 3
            x = column * 224
            if column in [0, 1]:
                img[:, :217, c:c+3] = self.image1s[x:x+224, :217, c:c+3]
                lable[:, :217] = self.labels[x:x+224, :217]
            else:  # 最后一列特殊考虑
                img[:512-448, :217, c:c+3] = self.image1s[448:512, :217, c:c+3]
                lable[:512-448, :217] = self.labels[448:512, :217]
            c = c + 3
        img = transform(img)
        return [img, lable]

    def __len__(self):
        return self.len


class mymodel(torch.nn.Module):
    def __init__(self,
                 args,
                 pcas=None,
                 patch_size=16,
                 maxpool_10=False,
                 backbone_norm_cfg=dict(type='LN', requires_grad=True),
                 **kwargs):
        super(mymodel, self).__init__(**kwargs)
        self.args = args
        self.patch_size = patch_size
        self.pcas = pcas
        if args.model == 'swin':
            self.backbone = SwinTransformer(
                pretrain_img_size=224,
                embed_dims=128,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                init_cfg=dict(type='Pretrained', checkpoint=args.checkpoint_path),
                norm_cfg=backbone_norm_cfg
            )
        elif args.model == 'intern':
            pass

        self.backbone.init_weights()
        self.maxpool = nn.AvgPool2d(kernel_size=10, stride=10, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=10, stride=10, padding=1)
        self.maxpool_10 = maxpool_10
        self.addfeature = args.addfeature

    def forward(self, img):

        out = self.backbone(img.float())  # (1,3,224,224)
        out = self.dim_reduction(out)
        out = self.feature_scaling(out)
        out = torch.cat([out, img], dim=1)

        return out

    def feature_scaling(self, datas):
        out1 = []
        for data in datas:
            data = torch.tensor(data).to(self.args.device)
            magnification = int(224 /data.shape[-1])
            # data_resized = np.repeat(np.repeat(data, magnification, axis=2), magnification, axis=3)
            data_resized = data.repeat_interleave(magnification, axis=2).repeat_interleave(magnification, axis=3)
            out1.append(data_resized)
        out = torch.cat(out1, dim=1)
        return out

    def dim_reduction(self, x):
        assert self.pcas is not None, "没有训练特征提取器 pca"
        reducted_x = []
        for i, (imgids, pca) in enumerate(self.pcas.items()):
            x1 = x[imgids].detach().to('cpu')
            b, c, w, h = x1.shape
            x2 = pca.transform(x1.reshape((b * w * h, c)))
            reducted_x.append(x2.reshape((b, self.args.n_components[self.args.featureidx[i]], w, h)))
        return reducted_x


def image_split_sample(args, img, split_size, sample_num):
    # 检查图像大小是否为224X224
    n, c, w, h = img.shape
    if w != 224 or h != 224:
        print(f"数据的形状{img.shape}不符合要求，请重新选择")
        return None
    # list1 = [(1,1),(4,4),(7,7),(1,7),(7,1)]
    output = torch.zeros((n, c, 22, 22, sample_num)).to(args.device)  # 输出的大小为22X22X9
    for i in range(0, w - split_size, split_size):
        for j in range(0, h - split_size, split_size):
            # 获取当前小块的索引
            x = i // split_size
            y = j // split_size
            # 获取当前小块的图像数据
            block = img[:, :, i:i + split_size, j:j + split_size]
            # 对当前小块进行9次采样，取每次采样的第一个像素值作为输出
            for k in range(sample_num):
                # 计算每次采样的位置，使之均匀分布在小块中
                pos = (k + 0.5) * split_size / sample_num
                # 取整数部分作为索引
                idx = int(pos)
                output[..., x, y, k] = block[:, :, idx, idx]
            # for k,dot in enumerate(list1):
            #     output[..., dot[0], dot[1], k] = block[:,:,dot[0], dot[1]]
    return output


def train_pca(args, number=400):
    """
    训练特征提取器
    Args:
        number: (int),训练过程中使用的数据量


    Returns:(list),训练好的PCA(只对特定层级的特征有用)

    """
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    backbone_norm_cfg = dict(type='LN', requires_grad=True)
    if args.model == 'swin':
        model = SwinTransformer(
            pretrain_img_size=224,
            embed_dims=128,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            init_cfg=dict(type='Pretrained', checkpoint=args.checkpoint_path),
            norm_cfg=backbone_norm_cfg
        ).to(args.device)
    elif args.model == 'intern':
        pass
    model.init_weights()
    logger = logging.getLogger("train")
    logger.info('开始训练pca.')
    # 加载数据的数量
    if isinstance(args.featureidx, list):

        test_image1_paths = r'data/dataset/Salinas_corrected.mat'
        test_label_paths = r'data/dataset/Salinas_gt.mat'
        trainDataset = Dataset(test_image1_paths, test_label_paths, len=number, loaders='train')
        train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
        images = [[], [], [], []]
        pcas = []

        for data in train_loader:
            model.eval()
            image = data[0]
            image = image.to(args.device)

            for bd in range(0, image.shape[1], 3):
                out = model(image[:, bd:bd+3, ...].float())
                for idx in range(4):  # 对每一级特征分别处理
                    img_out = out[idx].detach().to('cpu')
                    images[idx].append(img_out)

        result = [torch.cat(img, dim=0) for img in images]
        for idx in args.featureidx:
            b, c, w, h = result[idx].shape
            X = np.reshape(result[idx], (b * w * h, c))
            # 创建一个PCA对象，指定要保留的主成分个数为2
            pca = PCA(n_components=args.n_components[idx], copy=True)
            # 调用fit()方法，传入要降维的数据X，训练PCA模型
            pca.fit(X)
            pcas.append(copy.deepcopy(pca))
        logger.info("训练完成.")
        return {k: pcas[i] for i, k in enumerate(args.featureidx)}


# 使用数据加载器加载数据
def getimg(args, logger, loaders="train", number=204, models =None, position_offset=0):
    """
    Args:
        args: 其他参数
        loaders: 加哉哪种数据-train、val、test
        number: 加载数据的数量（0 - 54X54X4）
        addfeature: 是否加入原始特征
        model_name: 使用模型的名字
        position_offset: 加载数据从何处开始

    Returns: data、label

    """

    test_image1_paths = r'data/dataset/Salinas_corrected.mat'
    test_label_paths = r'data/dataset/Salinas_gt.mat'
    trainDataset = Dataset(test_image1_paths, test_label_paths, len=number, loaders=loaders)
    valDataset = Dataset(test_image1_paths, test_label_paths, len=number, loaders=loaders,
                         position_offset=position_offset)
    testDataset = Dataset(test_image1_paths, test_label_paths, len=number, loaders=loaders)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             drop_last=False)

    if loaders == "train":
        loader = train_loader
    elif loaders == "val":
        loader = val_loader
    elif loaders == "test":
        loader = test_loader
    model1 = models[0]
    images = []
    images1 = []
    targets = []
    for data in tqdm(loader):
        model1.eval()
        image = data[0]
        target = data[1]
        image = image.to(args.device)
        for bd in range(0, image.shape[1], 3):
            images.append(model1(image[:, bd:bd+3, ...].float()).detach().to('cpu'))
        targets.append(target)
        images1.append(torch.cat(images, dim=1))
    result = images1
    label = torch.cat(targets, dim=0)

    result = torch.einsum('nchw->nhwc', result)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    logger.info(f"{loaders}数据加载完成 {np.array(result).shape}, {np.array(label).shape}\n")
    return np.array(result), np.array(label)


