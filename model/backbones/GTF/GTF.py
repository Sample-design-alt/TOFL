from .GTFblock import GTFblock
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F


class GTF(pl.LightningModule):
    def __init__(self, in_channels, num_classes, learning_rate=0.0001, n_filters=[32,32], kernel_sizes=[9,19,39],
                 bottleneck_channels=32, activation=nn.ReLU(), use_residual=True, configs=None, opt=None):
        super(GTF, self).__init__()
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy( num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)
        self.GTF_blocks = []
        self.configs = configs
        self.opt = opt
        for i, n_filter in enumerate(n_filters):
            GTF_block = None
            if i == 0:
                GTF_block = GTFblock(
                    in_channels=in_channels,
                    n_filters=n_filter,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    use_residual=use_residual
                )
                # alter = onmi_cnn(in_channels=1, out_channel=n_filter)
            else:
                GTF_block = GTFblock(
                    in_channels=4 * n_filters[i - 1],
                    n_filters=n_filter,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    use_residual=use_residual
                )
            self.GTF_blocks.append(GTF_block)
        self.GTF_blocks = nn.Sequential(*self.GTF_blocks)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * n_filters[-1], num_classes, bias=True)

    def forward(self, X, task='classification'):
        X = X.transpose(1, 2).contiguous()
        X = self.GTF_blocks(X)
        if task == 'prediction':
            output = F.normalize(X, dim=1)
            # output = output.transpose(1, 2).contiguous()
            return output
        else:
            X = self.adaptive_avg_pool(X)
            X = self.flatten(X)
            output = F.normalize(X, dim=1)
            return output
        # X = self.fc(X)

        # output = self.avg(h)
        #
        # output = self.flatten(output)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.train_accuracy(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.val_accuracy(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.test_accuracy(outputs, labels)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy)
