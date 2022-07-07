import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, lambda_main, lambda_user, lambda_item):
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, pred_user, pred_item, label, label_user, label_item):
        loss_main = self.criterion(pred, label)
        if self.lambda_user:
            loss_user = self.criterion(pred_user, label_user)
        if self.lambda_item:
            loss_item = self.criterion(pred_item, label_item)

        if self.lambda_item and self.lambda_user:
            loss = self.lambda_main * loss_main + self.lambda_user * loss_user\
                + self.lambda_item * loss_item
        elif self.lambda_item:
            loss = self.lambda_main * loss_main + self.lambda_item * loss_item
        elif self.lambda_user:
            loss = self.lambda_main * loss_main + self.lambda_user * loss_user
        else:
            loss = loss_main
        return loss

class BPRLoss(nn.Module):
    def __init__(self, lambda_main, lambda_user, lambda_item):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1+torch.exp(-distance))))
        return loss