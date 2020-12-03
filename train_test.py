import os
import torch
import argparse
import random
import numpy as np

from BL_Trainer import BL_Trainer

def main(config):
    folder_path = {
        'SPAQ': 'F:/0datasets/SPAQ/',
    }

    img_num = {
        'SPAQ': list(range(0, 11125)),
    }

    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        # Randomly select 80% images for training and the rest for testing

        random.shuffle(sel_num)

        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        print('train_index', train_index)
        print('test_index', test_index)

        if config.model == 'BL':
            solver = BL_Trainer(config, folder_path[config.dataset], train_index, test_index)
        elif config.model == 'MT-E':
            solver = MTE_Trainer(config, folder_path[config.dataset], train_index, test_index)
        elif config.model == 'MT-A':
            solver = MTA_Trainer(config, folder_path[config.dataset], train_index, test_index)
        elif config.model == 'MT-S':
            solver = MTS_Trainer(config, folder_path[config.dataset], train_index, test_index)

        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='SPAQ',
                        help='Support datasets: SPAQ')
    parser.add_argument('--model', dest='model', type=str, default='BL',
                        help='Support model: BL, MT-E, MT-A, MT-S')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=10,
                        help='Number of sample patches from training image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=5, help='Train-test times')

    config = parser.parse_args()
    main(config)
