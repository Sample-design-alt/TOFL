# -*- coding: utf-8 -*-

import torch
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.models import TOFL
from model.TOFL import Model_TOFL
from model.MTL import Model_MTL
from model.SemiTime import SemiTime
from model.model_backbone import SimConv4
from model.model_res_backbone import Resv4
from model.inception.inceptiontime import InceptionTime
from torch.utils.data.sampler import SubsetRandomSampler
from utils.trainer_strategies import get_backbone
from utils.utils import count_parameters





def SemiTrain(x_train, y_train, x_val, y_val, x_test, y_test, opt, configuration):
    K = configuration['data_params']['K']
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = configuration['model_params']['d_model']
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    cutPF = transforms.CutPF(sigma=configuration['data_params']['alpha'])
    cutPF_transform = transforms.Compose([cutPF])

    tensor_transform = transforms.ToTensor()

    # if opt.model_name == 'SemiTime':
    #     backbone = SimConv4(config=configuration).cuda()
    #     model = TOFL(backbone, configuration, feature_size, configuration['data_params']['nb_class']).cuda()
    # else:
        # backbone = Resv4(config=configuration).cuda()
    # backbone =InceptionTime(1,configuration['data_params']['nb_class']).cuda()
    backbone = get_backbone(opt, configuration)



    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)

    if opt.model_name =='TOFL':
        model = Model_TOFL(backbone, configuration, feature_size, configuration['data_params']['nb_class']).cuda()
        train_set = MultiUCR2018_PF(data=x_train, targets=y_train, K=K,
                                    transform=train_transform,
                                    transform_cuts=cutPF_transform,
                                    totensor_transform=tensor_transform)

    elif opt.model_name =='SemiTime':
        model = SemiTime(backbone, configuration, feature_size, configuration['data_params']['nb_class']).cuda()
        train_set = MultiUCR2018_PF(data=x_train, targets=y_train, K=K,
                                    transform=train_transform,
                                    transform_cuts=cutPF_transform,
                                    totensor_transform=tensor_transform)
    elif opt.model_name == 'MTL':
        model = Model_MTL(backbone, configuration, feature_size, configuration['data_params']['nb_class']).cuda()

        train_set = MTL(data=x_train, targets=y_train, K=K,
                                    transform=train_transform,
                                    transform_cuts=cutPF_transform,
                                    totensor_transform=tensor_transform)
    else:
        raise NotImplementedError
    # val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    count_parameters(model, only_trainable=True)
    count_parameters(model, only_trainable=False)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(configuration['data_params']['label_ratio'] * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                     batch_size=batch_size,
                                                     sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = model.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                                    train_loader_label=train_loader_label,
                                                    val_loader=test_loader,
                                                    test_loader=test_loader,
                                                    opt=opt, config=configuration)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))


    # return acc_max, epoch_max
    return test_acc, acc_unlabel, best_epoch
