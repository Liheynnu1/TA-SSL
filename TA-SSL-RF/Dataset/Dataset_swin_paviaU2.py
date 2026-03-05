import scipy.io as scio
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
# from models import InternImage
from util import standardization_org, Composite_feature_map, soft_composite_feature_map
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from typing import Union, List, Any
from mmseg.models.backbones.swin import SwinTransformer
# from timm.models.swin_transformer import SwinTransformer
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
                 position_offset=0,
                 **kwargs):

        self.args = args
        self.len = len
        self.transform = transform
        self.position_offset = position_offset
        self.all_index = np.arange(54 * 216)
        np.random.shuffle(self.all_index)
        tmp_img = scio.loadmat(image1_paths)[args.dataset_img]
        tmp_img = np.array(tmp_img)  # 将数据转为数组
        image2s = standardization_org(tmp_img, cat=False)
        self.HSI, self.HSI_org = soft_composite_feature_map(self.args, [image2s])
        labels = np.array(scio.loadmat(label_paths)[args.dataset_label])
        w, h = np.shape(labels)[-2:]
        labels[labels == 255] = 0  # 统一背景类标签
        labels[labels == -1] = 0  # 统一背景类标签
        self.labels = (np.array(labels, dtype=np.uint8))
        self.piece_size = args.piece_size
        self.len = labels[labels != 0].reshape((-1)).shape[0]
        self.labels_org = (np.array(labels, dtype=np.uint8))
        self.labels = np.where(self.labels_org!=0)
        self.labels_idx = self.labels_org[self.labels_org!=0]
        self.chip_number = self.args.views[-1]


    def __getitem__(self, index):

        label_crop = self.labels_idx[index]
        label_crop = torch.tensor(np.array(label_crop))
        out_crop = dict(label=torch.tensor(np.array(label_crop)))
        image = []
        for i in range(len(self.HSI)):
            # 裁剪输入图像2
            x1 = self.labels[1][index]
            y1 = self.labels[0][index]
            x2 = x1 + 224
            y2 = y1 + 224

            image2_crop = self.HSI[i][y1:y2, x1:x2, :]
            image2_crop = transform(image2_crop)


            image.append(image2_crop)
            out_crop.update(dict(image=image))
        if 0 in self.args.views:

            out_crop.update(dict(image_org=self.HSI_org[y1 + 112, x1 + 112, :]))
        c = self.args.dataset_shape[2]
        idx = [i + c // (self.chip_number+1) for i in range(0, c, int(c // self.chip_number))]
        idx = idx[:-1]

        out_crop.update(dict(image_patch=self.HSI_org[
                                         y1 + 112 - self.piece_size // 2: y1 + 112 + self.piece_size // 2,
                                         x1 + 112 - self.piece_size // 2: x1 + 112 + self.piece_size // 2,
                                         idx]))

        return out_crop

    def __len__(self):
        return self.len


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int = 1024,
                 hidden_dim: int = 2048,
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
                in_channels=args.in_channel,
                embed_dims=args.embed_dim//8,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=args.depths,
                num_heads=args.num_heads,
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
            a = args.checkpoint_path
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            # print(self.backbone)
            print(msg) # stages.0.blocks.0.norm1.weight
            assert len(msg[0]) < 10
            assert len(msg[1]) < 10

        if args.model == 'DMVL':
            self.backbone = SwinTransformer(
                pretrain_img_size=224,
                embed_dims=args.embed_dim // 8,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=args.depths,
                num_heads=args.num_heads,
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
            # assert len(msg[0]) < 10
            # assert len(msg[1]) < 10
            # self.backbone.init_weights()
            self.MLP = MLP(norm_layer=nn.Identity)
            msg = self.MLP.load_state_dict(checkpoint, strict=False)
            print(msg)
        elif args.model == 'vit':
            self.backbone = self.prepare_vitmodel(args.checkpoint_path, 'vit_base_patch16')
        elif args.model == 't2t':
            self.backbone = t2t_vit_14()
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(msg)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.stride = args.stride
        self.avg_pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride, padding=0)
        self.addfeature = args.addfeature
        self.piece_size = args.piece_size
        self.model = args.model
        self.embed_dim = args.embed_dim

    def forward(self, img):
        if self.args.model == 'SVM':
            return img
        out = self.backbone(img.float())  # (1,3,224,224)
        if self.args.model == 'vit':
            out = torch.einsum('bhp->bph', [out])
            out = out.reshape(-1, self.args.embed_dim, 14, 14)
            out = self.maxpool2(out)
        elif self.args.model == 't2t':
            out = self.maxpool2(out)
        else:
            out = out[3]
        if self.args.model == 'DMVL':
            out = torch.einsum("nchw->nhwc", out)
            out = self.MLP([out])
            out = out[0]
            out = torch.einsum("nhwc->nchw", out)
        if self.addfeature:
            b = img.shape[0]
            if self.piece_size % 2 == 0:
                img2 = self.maxpool(img[...,
                                    112 - self.piece_size // 2:112 + self.piece_size // 2,
                                    112 - self.piece_size // 2:112 + self.piece_size // 2])
            else:
                img2 = self.maxpool(img[...,
                                    112 - self.piece_size // 2:113 + self.piece_size // 2,
                                    112 - self.piece_size // 2:113 + self.piece_size // 2])

            out = torch.cat([out[:, :, 3, 3], img2.reshape((b, 3 * (self.piece_size // self.stride) ** 2))], dim=1)
        else:
            out = out[:, :, 3, 3]
        return out

    def feature_scaling(self, datas):
        out1 = []
        for data in datas:
            data = torch.tensor(data).to(self.args.device)
            magnification = int(224 / data.shape[-1])
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
        msg = model.load_state_dict(checkpoint, strict=False)
        model = nn.Sequential(*list(model.children())[:-3])
        print(msg)
        return model


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
        position_offset: 加载数据从何处开始

    Returns: data、label

    """

    if args.dataset == 'Salinas':
        test_image1_paths = r'data/dataset/Salinas_corrected.mat'
        test_label_paths = r'data/dataset/Salinas_gt.mat'
    elif args.dataset == 'PaviaU':
        test_image1_paths = r'data/dataset/PaviaU.mat'
        test_label_paths = r'data/dataset/PaviaU_gt.mat'
    elif args.dataset == 'Trento':
        test_image1_paths = r'data/dataset/Trento.mat'
        test_label_paths = r'data/dataset/Trento_gt.mat'
    elif args.dataset == 'Indian':
        test_image1_paths = r'data/dataset/Indian_pines_corrected.mat'
        test_label_paths = r'data/dataset/Indian_pines_gt.mat'
    else:
        test_image1_paths = r'data/dataset/Houston.mat'
        test_label_paths = r'data/dataset/Houston_gt.mat'

    testDataset = Dataset(args, test_image1_paths, test_label_paths,
                          len=number, loaders=loaders)
    loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                        drop_last=False)

    model1 = models
    images = []
    targets = []
    for data in tqdm(loader):
        model1.eval()
        image1 = []
        if 1 < len(args.views):
            for i, image in enumerate(data["image"]):
                image = image.to(args.device)
                image1.append(get_feature(args, model1, image)if args.model != 'SVM' else
                              image[:, :, 112, 112].to('cpu'))
        if 0 in args.views:
            image1.append(data["image_org"].to('cpu'))
        if args.addspectral:
            image1.append(torch.flatten(data["image_patch"], start_dim=1).to('cpu'))
        images.append(image1)
        targets.append(data["label"])
    result = [torch.cat(img, dim=1) for img in images]
    result = torch.cat(result, dim=0)
    label = torch.cat(targets, dim=0)

    # result = torch.einsum('nchw->nhwc', result)
    logger.info(f"{loaders}数据加载完成 {np.array(result).shape}, {np.array(label).shape}\n")
    result, label, labels_org = np.array(result), np.array(label), np.array(testDataset.labels_org)
    label[label == 255] = 0  # 统一背景类标签
    labels_org[labels_org == 255] = 0  # 统一背景类标签
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return result, label, labels_org
