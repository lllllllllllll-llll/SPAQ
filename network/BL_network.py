import torch
import torch.nn as nn
import torchvision


class BL_network(nn.Module):
	def __init__(self):
		super(BL_network, self).__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

	def forward(self, x):
		x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
		result = self.backbone(x)
		return result
