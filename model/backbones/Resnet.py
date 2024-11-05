from torch import nn
import math
import torch.nn.functional as F
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out', out.size(), 'res', residual.size(), self.downsample)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,configs):
        super(ResNet, self).__init__()
        self.inplanes = configs['model_params']['in_channel']
        self.bottleneck_channels = configs['model_params']['bottleneck_channels']
        self.n_filters = configs['model_params']['n_filters']
        self.layers = configs['model_params']['layers']


        if self.inplanes > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=self.inplanes,
                out_channels=self.bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = lambda x: x
            self.bottleneck_channels = 1


        self.layer1 = self._make_layer(BasicBlock, self.n_filters[0], self.layers[0])
        self.layer2 = self._make_layer(BasicBlock, self.n_filters[1], self.layers[1])  # , stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.n_filters[2], self.layers[2])  # , stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.n_filters[3], self.layers[3])  # , stride=2)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, task='classification'):
        x = x.transpose(1, 2).contiguous()

        x = self.bottleneck(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # X = self.adaptive_avg_pool(x)
        # X = self.flatten(X)
        # output = F.normalize(X, dim=1)

        if task == 'prediction':
            output = F.normalize(x, dim=1)
            # output = output.transpose(1, 2).contiguous()
            return output
        else:
            X = self.adaptive_avg_pool(x)
            X = self.flatten(X)
            output = F.normalize(X, dim=1)
            return output

        return output