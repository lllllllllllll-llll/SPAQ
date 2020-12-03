import torch
from scipy import stats
import numpy as np

from network.BL_network import BL_network
import dataloader.data_loader as data_loader

class BL_Trainer(object):
    """training and testing"""
    def __init__(self, config, path, train_idx, test_idx):
        self.epochs = config.epochs
        self.model = BL_network().cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras = [{'params': self.model.backbone.fc.parameters(), 'lr': self.lr}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(path,
                                              train_idx,
                                              config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size,
                                              istrain=True)
        test_loader = data_loader.DataLoader(path,
                                             test_idx,
                                             config.patch_size,
                                             0,
                                             batch_size=1,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:

                img = torch.as_tensor(img.cuda())
                label = torch.as_tensor(label.cuda())

                self.optimizer.zero_grad()

                pred = self.model(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_srcc, test_plcc = self.test(self.test_data)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc

            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Update optimizer
            lr = self.lr / pow(10, t // 10)
            if t <= 10:
                self.paras = [{'params': self.model.backbone.fc.parameters(), 'lr': lr}]
            elif t > 10:
                self.paras = [{'params': self.model, 'lr': lr}]

            self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            img = torch.as_tensor(img.cuda())
            label = torch.as_tensor(label.cuda())

            pred = self.model(img)

            score = np.mean(pred.cpu().detach().numpy())

            pred_scores.append(score)
            gt_scores = gt_scores + label.cpu().tolist()

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)
        return test_srcc, test_plcc
