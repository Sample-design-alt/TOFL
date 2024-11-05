# -*- coding: utf-8 -*-

import datetime
from optim.pretrain import *
import argparse
import os
from utils.parse import parse_option

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from utils.utils import log_file
from optim.train import supervised_train
from glob import glob

Train_Done = ['Crop', 'Adiac', 'AllGestureWiimoteX', 'GunPointMaleVersusFemale', 'BirdChicken', 'InlineSkate', 'GunPointAgeSpan', 'UWaveGestureLibraryAll', 'ECG200', 'Computers', 'FordA', 'SemiSOP', 'GesturePebbleZ1', 'ScreenType', 'UWaveGestureLibraryZ', 'StarLightCurves', 'CricketY', 'ArrowHead', 'TwoLeadECG', 'FiftyWords', 'Coffee', 'DiatomSizeReduction', 'PowerCons', 'AllGestureWiimoteZ', 'DistalPhalanxOutlineCorrect', 'FreezerSmallTrain', 'BME', 'MelbournePedestrian', 'PigCVP', 'DistalPhalanxOutlineAgeGroup', 'Worms', 'MixedShapesSmallTrain', 'ChlorineConcentration', 'GestureMidAirD1', 'Lightning7', 'PickupGestureWiimoteZ', 'SwedishLeaf', 'BeetleFly', 'InsectEPGRegularTrain', 'SmallKitchenAppliances', 'Strawberry', 'ShapeletSim', 'WordSynonyms', 'FaceFour', 'ItalyPowerDemand', 'EthanolLevel', 'MiddlePhalanxOutlineCorrect', 'Rock', 'Symbols', 'SonyAIBORobotSurface1', 'SemgHandSubjectCh2', 'Mallat', 'ECG5000', 'HouseTwenty', 'MedicalImages', 'MoteStrain', 'MiddlePhalanxTW', 'NonInvasiveFetalECGThorax2', 'PhalangesOutlinesCorrect', 'SyntheticControl', 'Herring', 'Fish', 'AllGestureWiimoteY', 'CinCECGTorso', 'FreezerRegularTrain', 'ShapesAll', 'EOGHorizontalSignal', 'WormsTwoClass', 'UWaveGestureLibraryX', 'Beef', 'SemiTime', 'ProximalPhalanxOutlineAgeGroup', 'NonInvasiveFetalECGThorax1', 'SmoothSubspace', 'GestureMidAirD3', 'RefrigerationDevices', 'SemgHandMovementCh2', 'DodgerLoopWeekend', 'DistalPhalanxTW', 'UWaveGestureLibraryY', 'HandOutlines']


if __name__ == "__main__":
    import os
    import numpy as np
    import yaml

    opt = parse_option()
    opt.dataset_name = 'TSC'

    config_path = './experiments/config/{0}.yaml'.format(opt.model_name)
    configuration = yaml.safe_load(open(config_path, 'r'))
    configuration['data_params']['label_ratio'] = 1
    configuration['data_params']['alpha'] = 0.5

    exp = 'exp-cls'

    Seeds = [0, 1]
    Runs = range(0, 2, 1)

    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']

    if opt.model_name == 'SemiSOP':
        model_paras = 'label{}_{}'.format(configuration['data_params']['label_ratio'],
                                          configuration['data_params']['alpha'])
    else:
        model_paras = 'label{}'.format(configuration['data_params']['label_ratio'])

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './results/{}/{}/{}/{}'.format(
        exp, opt.dataset_name, opt.model_name, model_paras)
    file2print_detail_train, file2print, file2print_detail = log_file(log_dir)

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
            exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
            model_paras, str(seed))

        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        print('[INFO] Running at:', opt.dataset_name)

        # cc = glob('./results/exp-cls/TSC/*')
        # l=[]
        # for i in range(len(cc)):
        #     arch=cc[i].split('/')[-1]
        #     l.append(arch)
        # print(l)
        if opt.dataset_name == 'TSC':
            TSC_archive = glob(opt.ucr_path + '/*')
            for i in range(len(TSC_archive)):
                dataset_name = TSC_archive[i].split('/')[-1]
                if dataset_name in Train_Done:
                    continue
                log_dir = './results/{}/{}/{}/{}/{}'.format(
                    exp, opt.dataset_name, dataset_name, opt.model_name, model_paras)
                file2print_detail_train, file2print, file2print_detail = log_file(log_dir)
                if dataset_name in Train_Done:
                    continue
                x_train, y_train, x_val, y_val, x_test, y_test, configuration['data_params']['nb_class'], _ \
                    = load_ucr2018(opt.ucr_path, dataset_name)
                ACCs_run = {}
                MAX_EPOCHs_run = {}
                for run in Runs:

                    ################
                    ## Train #######
                    ################

                    if 'SemiSOP' in opt.model_name:
                        acc_test, acc_unlabel, epoch_max = train_SemiTime(
                            x_train, y_train, x_val, y_val, x_test, y_test, opt, configuration)

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                        seed, round(acc_test, 2), round(acc_unlabel, 2), epoch_max),
                        file=file2print_detail_train)
                    file2print_detail_train.flush()

                    ACCs_run[run] = acc_test
                    MAX_EPOCHs_run[run] = epoch_max

                ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
                MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                    seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
                    file=file2print_detail)

                file2print_detail.flush()

            ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
            ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
            MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

            print("{}\t{}\t{}\t{}".format(
                opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
                file=file2print)
            file2print.flush()
        else:
            x_train, y_train, x_val, y_val, x_test, y_test, configuration['data_params']['nb_class'], _ \
                = load_ucr2018(opt.ucr_path, opt.dataset_name)
            ACCs_run = {}
            MAX_EPOCHs_run = {}
            for run in Runs:

                ################
                ## Train #######
                ################
                if opt.model_name == 'SemiTime':
                    acc_test, epoch_max = supervised_train(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt)
                    acc_unlabel = 0

                elif 'SemiSOP' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_SemiTime(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt, configuration)

                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                    seed, round(acc_test, 2), round(acc_unlabel, 2), epoch_max),
                    file=file2print_detail_train)
                file2print_detail_train.flush()

                ACCs_run[run] = acc_test
                MAX_EPOCHs_run[run] = epoch_max

            ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
            MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
                file=file2print_detail)

            file2print_detail.flush()

        ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
        ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
        MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

        print("{}\t{}\t{}\t{}".format(
            opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
            file=file2print)
        file2print.flush()
