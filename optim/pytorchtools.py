# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, backbone_best='chechpoint.pt',relation_head_best='chechpoint.pt',classification_head_best='chechpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.backbone_best = backbone_best
        self.relation_head_best = relation_head_best
        self.classification_head_best = classification_head_best

    def __call__(self, val_loss, model,h1,h2):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, self.checkpoint_pth)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            self.save_checkpoint(val_loss,model,h1,h2)


            self.counter = 0

    def save_checkpoint(self, val_loss, model,h1,h2):
        '''Saves model when validation loss decrease.'''
        if model is not None:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.backbone_best)
            torch.save(h1.state_dict(), self.classification_head_best)
            if h2!=False:
                torch.save(h2.state_dict(), self.relation_head_best)

            self.val_loss_min = val_loss