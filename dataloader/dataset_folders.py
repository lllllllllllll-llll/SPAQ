import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import xlrd

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SPAQFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num, status):
        data = xlrd.open_workbook(os.path.join(root, 'Annotations/MOS and Image attribute scores.xlsx'))
        table = data.sheet_by_index(0)

        imgnames = []
        labels = []
        for rowNum in range(table.nrows):
            if rowNum > 0:
                rowValue = table.row_values(rowNum)
                imgnames.append(rowValue[0])
                labels.append(rowValue[1])

        sample = []
        if status == True:
            '''Train'''
            print('train')
            for i, item in enumerate(index):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'TestImage', imgnames[item]), labels[item]))
        else:
            '''Test'''
            print('test')
            for i, item in enumerate(index):
                sample.append((os.path.join(root, 'TestImage', imgnames[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

