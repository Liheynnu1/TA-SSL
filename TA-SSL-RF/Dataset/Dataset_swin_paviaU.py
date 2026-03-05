import torch
import os
import torch.nn as nn
import copy
import numpy as np
import logging
from skimage import io
import scipy.io as scio
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
# from models import InternImage
from util import standardization_org, Composite_feature_map, add_zero
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from typing import Union, List, Any
from mmseg.models.backbones.swin import SwinTransformer
import models.backbones.models_vit as models_vit
from models import t2t_vit_t_24, t2t_vit_14
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


class Dataset(Dataset):
    def __init__(self,
                 args,
                 image1_paths=r'D:\gw\GW-main\SVM\data\dataset\PaviaU.mat',
                 label_paths=r'D:\gw\GW-main\SVM\data\dataset\PaviaU_gt.mat',
                 transform=transform,
                 len=16,
                 loaders="",
                 position_offset=0,
                 **kwargs):

        self.args = args
        self.len = len
        self.loaders = loaders
        self.transform = transform
        self.position_offset = position_offset
        self.all_index = np.arange(54 * 216)
        np.random.shuffle(self.all_index)
        data = scio.loadmat(image1_paths)['paviaU']
        tmp_img = np.array(data)  # 将数据转为数组
        image2s = tmp_img
        image2s = standardization_org(image2s, cat=False)
        self.HSI= Composite_feature_map(self.args, [image2s])
        labels = np.array(scio.loadmat(label_paths)['paviaU_gt'])
        w, h = np.shape(labels)[-2:]
        # self.len = int(w*h)
        self.labels = Image.fromarray(np.array(labels, dtype=np.uint8))
        self.img_size = args.image_size
        self.margin = args.piece_size/2-1
        pass




    def __getitem__(self, index):

        out_crop = []
        w, h = np.shape(self.labels)[-2:]
        Row = h


        # 计算对应的标签块
        x1 = index % Row
        y1 = index // Row
        x2 = x1 + 224
        y2 = y1 + 224
        label_crop = self.labels.crop((x1, y1, x2, y2))
        label_crop = torch.tensor(np.array(label_crop))
        out_crop.append(label_crop)

        if self.args.user_image:
            # 裁剪输入图像1
            x1 = index % Row
            y1 = index // Row
            x2 = x1 + 224
            y2 = y1 + 224
            image1_crop = self.image1s.crop((x1, y1, x2, y2))
            image1_crop = transform(image1_crop)
            out_crop.append(image1_crop.clone().detach())

        for i in range(len(self.HSI)):
            # 裁剪输入图像2
            x1 = index % Row
            y1 = index // Row
            x2 = x1 + 224
            y2 = y1 + 224

            image2_crop = self.HSI[i].crop((x1, y1, x2, y2))
            # image2_crop.show()
            image2_crop = transform(image2_crop)
            out_crop.append(image2_crop.clone().detach())

        return out_crop

    def __len__(self):
        return self.len


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int = 256,
                 hidden_dim: int = 512,
                 out_dim: int = 256,
                 norm_layer=nn.BatchNorm1d,
                 act_layer=nn.ReLU,
                 drop: float = 0.):
        super(MLP, self).__init__()

        self.layer = nn.ModuleList()
        self.layer.append(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm_layer(hidden_dim) if norm_layer is not nn.Identity else norm_layer(),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(drop)
        ))


    def forward(self, x: Union[torch.Tensor, List]):
        out = []
        for feature, layer in zip(x, self.layer):
            out.append(layer(feature))
        return out


def unpatchify_swin(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    patch_size = 4
    imgs = []
    for idx in range(4):
        feature = x[idx]
        p = feature.shape[-1]
        num_patches = p * p
        # feature:64,128,56,56
        h = w = patch_size
        feature = feature.reshape((feature.shape[0], h, w, -1, p, p))  # feature:64,4,4,8,56,56
        feature = torch.einsum('nhwcpq->nchpwq', feature)  # feature:64,8,4,56,4,56
        imgs.append(feature.reshape((feature.shape[0], -1, h * p, w * p)))  # feature:64,8,224,224
    return imgs

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
        self.maxpool_10 = maxpool_10
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
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(msg)
            # self.backbone.init_weights()
        elif args.model == 'vit':
            self.backbone = self.prepare_vitmodel(args.checkpoint_path, 'vit_base_patch16')
        elif args.model == 't2t':
            self.backbone = t2t_vit_14()
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(msg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.addfeature = args.addfeature

    def forward(self, img):
        if self.args.model == 'SVM':
            return img
        out = self.backbone(img.float())  # (1,3,224,224)
        if self.args.model == 'vit':
            out = torch.einsum('bhp->bph', [out])
            out = out.reshape(-1, self.args.embed_dim, 14, 14)
            out = self.maxpool(out)
        elif self.args.model == 't2t':
            out = self.maxpool(out)


        else:
            out = out[3]

        if self.addfeature:
            out = torch.cat([out[:,:,3,3], img[..., 112, 112]], dim=1)
        else:
            out = out[:,:,3,3]

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
            x1 = torch.einsum("nchw->nhwc", [x1])
            x2 = pca.fit_transform(x1.reshape((b * w * h, c)))
            reducted_x.append(x2.reshape((b, self.args.n_components[self.args.featureidx[i]], w, h)))
        return reducted_x

    def prepare_vitmodel(self, chkpt_dir, arch='vit_base_patch16'):
        # build model vit_base_patch16  vit_large_patch16
        model = getattr(models_vit, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        model = nn.Sequential(*list(model.children())[:-3])
        print(msg)
        return model
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

def image_split_sample(args, img, split_size, sample_num):
    # 检查图像大小是否为224X224
    n, c, w, h = img.shape
    if w != 224 or h != 224:
        print(f"数据的形状{img.shape}不符合要求，请重新选择")
        return None
    # list1 = [(1,1),(4,4),(7,7),(1,7),(7,1)]
    output = torch.zeros((n, c, args.piece_size, args.piece_size, sample_num)).to(args.device)  # 输出的大小为22X22X9
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





def get_pca(args, number=400):
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
    # elif args.model == 'intern':
    #     model = InternImage(
    #         core_op='DCNv3',
    #         channels=112,
    #         depths=[4, 4, 21, 4],
    #         groups=[7, 14, 28, 56],
    #         mlp_ratio=4.,
    #         drop_path_rate=0.4,
    #         norm_layer='LN',
    #         layer_scale=1.0,
    #         offset_scale=1.0,
    #         post_norm=True,
    #         with_cp=False,
    #         out_indices=(0, 1, 2, 3),
    #         init_cfg=dict(type='Pretrained', checkpoint=args.checkpoint_path)
    #     ).to(args.device)
    model.init_weights()
    logger = logging.getLogger("train")
    logger.info('开始训练pca.')
    # 加载数据的数量

    if isinstance(args.featureidx, list):

        test_image1_paths = r'data/image/image.png'
        test_image2_paths = r'data/HSI/20170218_UH_CASI_S4_NAD83.tif'
        test_label_paths = r'data/label/labeal.png'
        trainDataset = Dataset(args, test_image1_paths, test_image2_paths, test_label_paths,
                               len=number, loader='train')
        train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,drop_last=False)
        images = [[], [], [], []]
        pcas = []

        for data in train_loader:
            model.eval()
            for i, image in enumerate(data[1:]):
                image = image.to(args.device)
                out = model(image)
                for idx in range(4):  # 对每一级特征分别处理
                    img_out = out[idx].detach().to('cpu')
                    images[idx].append(img_out)
        assert args.dim_reduction in ['PCA', 'MDS', 'Isomap', 'SpectralEmbedding', 'TSNE', 'LinearDiscriminantAnalysis']
        result = [torch.cat(img, dim=0) for img in images]
        for idx in args.featureidx:
            X = result[idx]
            b, c, w, h = X.shape
            X = torch.einsum("nchw->nhwc", [X])
            X = np.reshape(X, (b * w * h, c))
            # 创建一个PCA对象，指定要保留的主成分个数为2
            pca = PCA(n_components=args.n_components[idx], normalized_stress='auto')
            # pca = MDS(n_components=args.n_components[idx], copy=True)
            # 调用fit()方法，传入要降维的数据X，训练PCA模型
            # pca.fit(X)
            pcas.append(copy.deepcopy(pca))
        logger.info("训练完成.")
        return {k: pcas[i] for i, k in enumerate(args.featureidx)}


def get_feature(args, model, image):
    out = model(image).detach().to('cpu')
    out = torch.tensor(out)
    # out = torch.unsqueeze(out, dim=2)
    return out


# 使用数据加载器加载数据
def getimg(args, logger, loaders="train", number=20, models =None, position_offset=0):
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

    test_image1_paths = r'data/dataset/PaviaU.mat'
    test_label_paths = r'data/dataset/PaviaU_gt.mat'

    testDataset = Dataset(args, test_image1_paths, test_label_paths,
                          len=number, loaders=loaders)
    loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             drop_last=False)

    model1 = models
    images = []
    targets = []
    img_size = args.image_size
    images = []
    targets = []
    margin = int(args.piece_size / 2 - 1)
    for data in tqdm(loader):
        model1.eval()
        image1 = []
        target = data[0]
        for i, image in enumerate(data[1:]):
            image = image.to(args.device)
            if 0 in args.views and i >= 1:  # HSI=1 + image=1
                image1.append(image[..., margin:img_size+margin, margin:img_size+margin].to('cpu'))
                continue  # 添加高光谱的原始数据
            image1.append(get_feature(args, model1, image)if args.model != 'SVM' else
                          image[..., margin:img_size+margin, margin:img_size+margin].to('cpu'))
        images.append(image1)
        targets.append(target[:, 0, 0])
    result = [torch.cat(img, dim=1) for img in images]
    result = torch.cat(result, dim=0)
    label = torch.cat(targets, dim=0)

    # result = torch.einsum('nchw->nhwc', result)
    logger.info(f"{loaders}数据加载完成 {np.array(result).shape}, {np.array(label).shape}\n")
    result, label = np.array(result), np.array(label)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return result, label



