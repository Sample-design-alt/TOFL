# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class res_block(torch.nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=1,padding=1):
        super(res_block, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(in_feature, in_feature, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2)),
            torch.nn.BatchNorm1d(in_feature),
            # torch.nn.LayerNorm(300),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(in_feature, out_feature, kernel_size=kernel_size, stride=1,
                      padding=int(kernel_size // 2)),
            torch.nn.BatchNorm1d(out_feature),
            # torch.nn.LayerNorm(300),
            torch.nn.ReLU(),
        )

        self.shortcut = torch.nn.Sequential(
            nn.Conv1d(in_feature, out_feature, kernel_size=kernel_size, stride=1,
                      padding=int(kernel_size // 2)),
            torch.nn.BatchNorm1d(out_feature),
            # torch.nn.LayerNorm(300),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x_ = self.conv2(self.conv1(x))
        shortcut = self.shortcut(x)
        x = x_ + shortcut
        return x


class Resv4(torch.nn.Module):
    def __init__(self, config):
        super(Resv4, self).__init__()
        self.feature_size = config['model_params']['feature']
        self.name = "conv4"

        self.block1 = res_block(1, config['model_params']['l1'], 5, 2)
        self.block2 = res_block(config['model_params']['l1'], config['model_params']['l2'], config['model_params']['kernel'], 2,int(config['model_params']['kernel']//2))
        self.block3 = res_block(config['model_params']['l2'], config['model_params']['l3'], config['model_params']['kernel'],2, int(config['model_params']['kernel']//2))
        self.block4 = res_block(config['model_params']['l3'], config['model_params']['feature'], config['model_params']['kernel'],2, int(config['model_params']['kernel']//2))


        self.avg = torch.nn.AdaptiveAvgPool1d(1)
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

        # h = self.layer1(x_)  # (B, 1, D)->(B, 8, D/2)
        # h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        # h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        # h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)

        h = self.block1(x_)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        output = self.avg(h)

        output = self.flatten(output)
        output = F.normalize(output, dim=1)
        return output
