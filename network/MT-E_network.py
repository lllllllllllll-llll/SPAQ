import torch
import torch.nn as nn
import torchvision
from PIL import Image
import argparse
import os

class MTE(nn.Module):
	def __init__(self, config):
		super(MTE, self).__init__()
		self.config = config
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)
		self.exifCNN = nn.Linear(self.config.input_channels, 1, bias=False)

	def forward(self, x, exif):
		generic = self.backbone(x)
		bias = self.exifCNN(exif)
		return generic + bias
