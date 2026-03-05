import os
import torch
import functools
import numpy as np
import torchvision
import torch.nn as nn
from scipy.signal import wiener
from collections import abc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from inspect import getfullargspec
from torch.cuda.amp import autocast
from mmcv.utils import TORCH_VERSION, digit_version
from .loss_utils import RegressionLoss, FocalFrequencyLoss
import albumentations as A
np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告

def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, apply_inv=True) -> None:
        trans_cifar = [
            torchvision.transforms.Normalize(
                mean=[0., 0., 0.], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.201]),
            torchvision.transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[1., 1., 1.])]
        trans_in = [
            torchvision.transforms.Normalize(
                mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]
        if apply_inv:
            self.invTrans_cifar = torchvision.transforms.Compose(trans_cifar)
            self.invTrans_in = torchvision.transforms.Compose(trans_in)

    def plot(self,
             img, nrow=4, title_name=None, save_name=None,
             dpi=None, apply_inv=True, overwrite=False):
        assert save_name is not None
        assert img.size(0) % nrow == 0
        ncol = img.size(0) // nrow
        if ncol > nrow:
            ncol = nrow
            nrow = img.size(0) // ncol
        img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)

        cmap = None
        if img.size(1) == 1:
            cmap = plt.cm.gray
        if apply_inv:
            if img.size(2) <= 64:
                img_grid = self.invTrans_cifar(img_grid)
            else:
                img_grid = self.invTrans_in(img_grid)
        img_grid = torch.clip(img_grid * 255, 0, 255).int()
        img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure(figsize=(nrow * 2, ncol * 2))
        plt.imshow(img_grid, cmap=cmap)
        if title_name is not None:
            plt.title(title_name)
        if not os.path.exists(save_name) or overwrite:
            plt.savefig(save_name, dpi=dpi)
        plt.close()


class MIMHead(nn.Module):
    def __init__(self,
                 in_channels: int = 2048,
                 in_chans: int = 3,
                 kernel_size: int = 1,
                 encoder_stride: int = 32,
                 ):
        super(MIMHead, self).__init__()

        self.decoder_pred = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride ** 2 * in_chans,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, feature: torch.Tensor):
        return self.decoder_pred(feature[-1])

class LocalFiltering(nn.Module):
    def __init__(self, input_channels, mysize):
        super(LocalFiltering, self).__init__()
        self.conv_mean = nn.Conv2d(input_channels, input_channels, kernel_size=mysize, padding='same', groups=input_channels)
        self.conv_var = nn.Conv2d(input_channels, input_channels, kernel_size=mysize, padding='same', groups=input_channels)

        # Initialize weights to 1
        nn.init.constant_(self.conv_mean.weight, 1)
        nn.init.constant_(self.conv_var.weight, 1)
    def forward(self, x):
        lMean = self.conv_mean(x) / torch.prod(torch.as_tensor(self.conv_mean.kernel_size))
        lVar = (self.conv_var(x ** 2) / torch.prod(torch.as_tensor(self.conv_var.kernel_size)) - lMean ** 2)
        return lMean, lVar
class MIMLossHead(nn.Module):
    """Head for A2MIM training.

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        fft_weight (float): Loss weight for the fft prediction loss. Default to 0.
        fft_focal (bool): Whether to adopt the focal fft loss. Default to False.
        fft_unmask_replace (str): Mode to replace (detach) unmask patches for the fft
            loss, in {None, 'target', 'prediction', 'mean', 'mixed',}.
        fft_unmask_weight (float): Loss weight to caculate the fft loss on unmask
            tokens. Default to 0.
    """

    def __init__(self,
                 loss=dict(
                     loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 fft_weight=0,
                 fft_focal=False,
                 fft_unmask_replace=None,
                 fft_unmask_weight=0,
                 ):
        super(MIMLossHead, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.fft_weight = fft_weight
        self.fft_focal = fft_focal
        self.fft_unmask_weight = fft_unmask_weight
        self.fft_unmask_replace = fft_unmask_replace
        assert fft_unmask_replace in [None, 'target', 'prediction', 'mean', 'mixed', ]
        assert 0 <= unmask_weight <= 1 and 0 <= fft_unmask_weight <= 1
        assert loss is None or isinstance(loss, dict)

        self.criterion = RegressionLoss(**loss)
        # fft loss
        if fft_focal:
            fft_loss = dict(
                loss_weight=1.0, alpha=1.0,
                ave_spectrum=True, log_matrix=True, batch_matrix=True)
            self.fft_loss = FocalFrequencyLoss(**fft_loss)
        kernel_size = 5
        self.mean_pool = torch.nn.AvgPool3d(kernel_size=(1, kernel_size, kernel_size),
                                       stride=(1, 1, 1),
                                       padding=(0, (kernel_size-1)//2, (kernel_size-1)//2))
        self.transform = A.Compose([
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
        ]
        )
        self.care = (3, 3)
        self.local_filtering_module = LocalFiltering(input_channels=3, mysize=self.care)

    def wiener_filtering_torch(self, im, mysize=None, noise=None):
        im = torch.as_tensor(im, dtype=torch.float32)

        # if mysize is None:
        #     mysize = [3] * im.ndimension()
        # mysize = tuple(torch.as_tensor(size).item() if torch.is_tensor(size) else size for size in mysize)
        #
        # if len(mysize) == 0:
        #     mysize = (im.ndimension(),)

        # # Create an instance of the LocalFiltering module
        # local_filtering_module = LocalFiltering(input_channels=im.shape[1], mysize=mysize)

        # Forward pass through the module
        lMean, lVar = self.local_filtering_module(im)

        # Estimate the noise power if needed.
        if noise is None:
            noise = lVar.mean()

        res = (im - lMean)
        res *= (1 - noise / lVar)
        res += lMean

        out = torch.where(lVar < noise, lMean, res)
        return out

    def wiener_filter(self, image_tensor, filter_size):
        # 对每个颜色通道单独进行 Wiener 滤波
        image = image_tensor.clone().detach()

        # 调用 local_filtering_torch 函数 4.3.244.244
        result_image = self.wiener_filtering_torch(image, mysize=filter_size)

        # 将结果转换为与输入相同的设备和数据类型
        result_image = result_image.to(image_tensor.device)
        # result_image = torch.from_numpy(result_image).to(image_tensor.device).float()

        # result_image = [self.transform(image=result_image[i, ::].detach().to("cpu").numpy().transpose(1, 2, 0))["image"].transpose(2, 0, 1) for i
        #                 in range(result_image.shape[0])]
        out = []
        for i in range(result_image.shape[0]):
            image = self.transform(image=result_image[i, ::].detach().to("cpu").numpy().transpose(1, 2, 0))["image"]
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)
            out.append(image)
        result_image = np.concatenate(out, axis=0)
        return torch.from_numpy(result_image).to(image_tensor.device).float()


    def forward(self, x, x_rec, mask):
        # upsampling mask
        scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)  # 下采样的stride
        if scale_h > 1:
            mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                int(scale_w), 2).unsqueeze(1).contiguous()
        else:
            mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                 scale_factor=(scale_h, scale_w), mode="nearest")
        # x = self.mean_pool(x)
        # x = self.wiener_filter(x, self.care)
        # spatial loss
        if self.unmask_weight > 0.:
            # reweight unmasked patches
            mask_s = mask.clone()
            mask_s = mask_s + (1. - mask_s) * self.unmask_weight
        else:
            mask_s = mask
        loss_rec = self.criterion(x_rec, target=x, reduction_override='none')
        loss_rec = (loss_rec * mask_s).sum() / (mask_s.sum() + 1e-5) / self.encoder_in_channels

        # fourier domain loss
        if self.fft_weight > 0:
            # replace unmask patches (with detach)
            x_replace = None
            if self.fft_unmask_replace is not None:
                if self.fft_unmask_replace == 'target':
                    x_replace = x.clone()
                elif self.fft_unmask_replace == 'prediction':
                    x_replace = x_rec.clone().detach()
                elif self.fft_unmask_replace == 'mean':
                    x_replace = x.mean(dim=[2, 3], keepdim=True).expand(x.size())
                elif self.fft_unmask_replace == 'mixed':
                    x_replace = 0.5 * x_rec.clone().detach() + 0.5 * x.clone()
            if self.fft_unmask_weight < 1:
                mask_f = mask.clone()
                mask_f = mask_f + (1. - mask_f) * self.fft_unmask_weight
                x_rec = (x_rec * mask_f) + (x_replace * (1. - mask_f))  # replace unmask tokens

            # apply fft loss
            if self.fft_focal:
                loss_fft = self.fft_loss(x_rec, x)
                loss_rec += self.fft_weight * loss_fft

        return loss_rec

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

import math
import warnings
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm = PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act