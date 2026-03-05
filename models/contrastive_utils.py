import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Union, List, Any
from timm.models.vision_transformer import trunc_normal_


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.where(sorted_indices_indices < num_matches, True, False)
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def position_match(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


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


class LocalHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 use_bn: bool = True,
                 norm_last_layer: bool = True,
                 num_layers: int = 3,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256):
        super(LocalHead, self).__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Conv1d(in_dim, bottleneck_dim, 1)
        else:
            layers: List[Any] = [nn.Conv1d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv1d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
            # self.mlp1 = nn.Conv1d(192, 2048, kernel_size=(1,), stride=(1,))
            # self.mlp2 = nn.GELU()
            # self.mlp3 = nn.Conv1d(2048, 2048, kernel_size=(1,), stride=(1,))
            # self.mlp4 = nn.GELU()
            # self.mlp5 = nn.Conv1d(2048, 256, kernel_size=(1,), stride=(1,))
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Conv1d(bottleneck_dim, out_dim, 1, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the backbone, the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """

        x = self.mlp(x)
        x = F.normalize(x, dim=1)
        x = self.last_layer(x)
        return x
#
# class iBOTLoss(nn.Module):
#     def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp,
#                  teacher_temp, warmup_teacher_temp2, teacher_temp2,
#                  warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
#                  center_momentum=0.9, center_momentum2=0.9,
#                  lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
#         """
#         初始化 iBOTLoss 模块。
#
#         参数:
#             out_dim (int): 全局裁剪输出的维度。
#             patch_out_dim (int): 补丁裁剪输出的维度。
#             ngcrops (int): 全局裁剪的数量。
#             nlcrops (int): 局部裁剪的数量。
#             warmup_teacher_temp (float): 教师网络的初始温度。
#             teacher_temp (float): 教师网络的最终温度。
#             warmup_teacher_temp2 (float): 补丁居中的初始温度。
#             teacher_temp2 (float): 补丁居中的最终温度。
#             warmup_teacher_temp_epochs (int): 温度热身的时期数。
#             nepochs (int): 总的训练时期数。
#             student_temp (float): 学生网络的温度 (默认: 0.1)。
#             center_momentum (float): 用于更新全局中心的动量 (默认: 0.9)。
#             center_momentum2 (float): 用于更新补丁中心的动量 (默认: 0.9)。
#             lambda1 (float): 全局分类损失的权重 (默认: 1.0)。
#             lambda2 (float): 补丁分类损失的权重 (默认: 1.0)。
#             mim_start_epoch (int): 开始使用 teacher_temp2_schedule 的时期。
#         """
#         super().__init__()
#         self.student_temp = student_temp
#         self.center_momentum = center_momentum
#         self.center_momentum2 = center_momentum2
#         self.ngcrops = ngcrops
#         self.nlcrops = nlcrops
#         self.ncrops = ngcrops + nlcrops
#         self.register_buffer("center", torch.zeros(1, out_dim))
#         self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#
#         # 对教师温度进行热身，因为初始温度过高会导致训练不稳定
#         self.teacher_temp_schedule = np.concatenate((
#             np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         ))
#         self.teacher_temp2_schedule = np.concatenate((
#             np.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
#         )) if mim_start_epoch == 0 else np.concatenate((
#             np.ones(mim_start_epoch) * warmup_teacher_temp2,
#             np.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
#         ))
#
#     def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
#         """
#         计算教师和学生网络之间 softmax 输出的交叉熵。
#
#         参数:
#             student_output (tuple): 包含 student_cls 和 student_patch 输出的元组。
#             teacher_output (tuple): 包含 teacher_cls 和 teacher_patch 输出的元组。
#             student_local_cls (Tensor): 学生模型的局部分类输出。
#             student_mask (Tensor): 学生模型的掩码。
#             epoch (int): 当前训练时期。
#
#         返回:
#             dict: 包含全局和补丁损失组件的字典。
#         """
#         student_cls, student_patch = student_output
#         teacher_cls, teacher_patch = teacher_output
#
#         if student_local_cls is not None:
#             student_cls = torch.cat([student_cls, student_local_cls])
#
#         # 对全局裁剪的 [CLS] 和 patch 进行缩放
#         student_cls = student_cls / self.student_temp
#         student_cls_c = student_cls.chunk(self.ncrops)
#         student_patch = student_patch / self.student_temp
#         student_patch_c = student_patch.chunk(self.ngcrops)
#
#         # 教师居中和锐化
#         temp = self.teacher_temp_schedule[epoch]
#         temp2 = self.teacher_temp2_schedule[epoch]
#         teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
#         teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
#         teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
#         teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)
#
#         total_loss1, n_loss_terms1 = 0, 0
#         total_loss2, n_loss_terms2 = 0, 0
#
#         for q in range(len(teacher_cls_c)):
#             for v in range(len(student_cls_c)):
#                 if v == q:
#                     # 计算 patch 损失
#                     loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
#                     mask = student_mask[v].flatten(-2, -1)
#                     loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
#                     total_loss2 += loss2.mean()
#                     n_loss_terms2 += 1
#                 else:
#                     # 计算 cls 损失
#                     loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
#                     total_loss1 += loss1.mean()
#                     n_loss_terms1 += 1
#         # 计算平均损失并乘以权重
#         total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
#         total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
#         # 组合全局和补丁损失
#         total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
#         # 更新中心
#         self.update_center(teacher_cls, teacher_patch)
#         # 返回损失字典
#         return total_loss
#
#     @torch.no_grad()
#     def update_center(self, teacher_cls, teacher_patch):
#         """
#         更新用于教师输出的中心。
#         """
#         cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
#         dist.all_reduce(cls_center)
#         cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
#         self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)
#
#         patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
#         dist.all_reduce(patch_center)
#         patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
#         self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)