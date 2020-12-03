import torch
import torchvision
import dataloader.dataset_folders as dataset_folders
import numbers
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import torchvision.utils as utils

class Crop_patches(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, input_size, img_shape):

        img = img.astype(dtype=np.float32)
        if len(img_shape) == 2:
            H, W, = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size
        return H, W, stride, hIdxMax, wIdxMax

    def __call__(self, image):
        input_size = 224

        img = self.to_numpy(image)
        img_shape = img.shape
        H, W, stride, hIdxMax, wIdxMax = self.get_params(img, input_size, img_shape)

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()

        return patches_tensor

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

class DataLoader(object):
    def __init__(self, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):
        self.batch_size = batch_size
        self.istrain = istrain

        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512), interpolation=Image.BILINEAR),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512), interpolation=Image.BILINEAR),
                torchvision.transforms.ToTensor(),
                Crop_patches(size=patch_size),
            ])

        self.data = dataset_folders.SPAQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num, status=self.istrain)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
