# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class activation(torch.nn.Module):
    def __init__(self, type):
        super(activation, self).__init__()
        if type == 'relu':
            self.activation = torch.nn.ReLU()
        elif type == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(x)


class SimConv4(torch.nn.Module):
    def __init__(self, config, feature_size=64):
        super(SimConv4, self).__init__()
        self.feature_size = feature_size
        self.name = "conv4"

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(1, config['model_params']['l1'], config['model_params']['kernel'], 2, int(config['model_params']['kernel']//2), bias=False),
            torch.nn.BatchNorm1d(config['model_params']['l1']),
            activation(config['model_params']['activation']),
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(config['model_params']['l1'], config['model_params']['l2'], config['model_params']['kernel'], 2, int(config['model_params']['kernel']//2), bias=False),
            torch.nn.BatchNorm1d(config['model_params']['l2']),
            torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(config['model_params']['l2'], config['model_params']['l3'], config['model_params']['kernel'], 2, int(config['model_params']['kernel']//2), bias=False),
            torch.nn.BatchNorm1d(config['model_params']['l3']),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(config['model_params']['l3'], config['model_params']['feature'], config['model_params']['kernel'], 2, int(config['model_params']['kernel']//2), bias=False),
            torch.nn.BatchNorm1d(config['model_params']['feature']),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_ = x.view(x.shape[0], 1, -1)
        try:
            h = self.layer1(x_)  # (B, 1, D)->(B, 8, D/2)
            h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
            h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
            h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
            h = self.flatten(h)
            h = F.normalize(h, dim=1)
            return h
        except:
            print(x_)
