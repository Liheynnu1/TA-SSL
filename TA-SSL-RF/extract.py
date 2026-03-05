import torch
import argparse
from collections import OrderedDict
# from models import InternImage
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# pretrained = r'/home/guwei/GW-main/SVM/checkpoint/intern_200.pth'
# model = InternImage(
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
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)
# ).to(device)
# print(model)
def parse_args():
	parser = argparse.ArgumentParser(description='Convert Backbone')
	# 输入预训练后最后一个网络
	parser.add_argument("--checkpoint_path", default=r'D:\Users\HELI123\PythonWorkspace\pretrain\output\checkpoints\epoch_399.pth', type=str)
	# parser.add_argument("--checkpoint_path", default='D:\Users/HELI123/PythonWorkspace/pretrain/output/checkpoints/epoch_399.pth', type=str)
	parser.add_argument("--out_path", default='checkpoint/stage2-49.pth', type=str)
	args = parser.parse_args()

	return args



def main():
	args = parse_args()
	checkpoint_path = args.checkpoint_path
	out_path = args.out_path
	checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
	new_ckpt = OrderedDict()
	for key in checkpoint.keys():
		if key.startswith("online_encoder.model."):
			new_ckpt[key[len("online_encoder.model."):]] = checkpoint[key]
		# if key.startswith("online_projector."):
		# 	new_ckpt[key[len("online_projector."):]] = checkpoint[key]

	torch.save(new_ckpt, out_path)


if __name__ == "__main__":
	main()